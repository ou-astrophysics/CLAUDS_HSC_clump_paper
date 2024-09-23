#!/usr/bin/env python

# [START all]
# [START libraries]
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import numpy as np
import os
import sys
import time
sys.path.append('./shared/')
import GalaxyMeasurements

from itertools import repeat
import multiprocessing

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import convolve, Box2DKernel

import photutils
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from photutils.detection import find_peaks
from photutils.centroids import centroid_com
from photutils.utils.exceptions import NoDetectionsWarning
# [END libraries]


# [START initial settings]
DATA_PATH = './data/'
PREDICTIONS_FILE_PATH = './predictions/'
PHOTOMETRY_FILE_PATH = './photometry/'

IMAGE_DATA_PATH = './pngs/'
FITS_FILE_PATH = './fits/'
PSF_FILE_PATH = './psf/'

HSC_ARCSEC_PER_PIXEL = 0.168
SCORE_THRESHOLD = 0.0

NUM_WORKERS = multiprocessing.cpu_count()
# [END initial settings]


# [START classes and functions]
def load_data():
    """
        Function for loading the necessary dataFrame.
        Change accordingly, example below is for reading CLAUDS+HSC crossmatches

        Args: none
        Returns: Pandas DataFrame
    """

    # define your data set, here some dataframes saved as parquet-files
    # final returned dataframe should contain the following fields/columns, but a depending on the filterbands used:
    # 'HSCobjid', 'scores', 'is_central', 'labels', 'petroR90_r', 'useeing', 'gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'clump_id', 'detection_id', 'px_x1_normed', 'px_x2_normed', 'px_y1_normed', 'px_y2_normed'
    df_meta = pd.read_parquet(DATA_PATH + 'galaxy_image_meta_information_file.gzip', engine='pyarrow')
    df_pred = (pd
        .read_parquet(PREDICTIONS_FILE_PATH + 'predictions_file_name.gzip', engine='pyarrow')
        .merge(df_meta, how='inner', on='HSCobjid')
    )
    return df_pred


def peak_finder(df, bands, sigma):
    results = []
    error_list = []
    ff = {}
    ff_wcs = {}

    for objid in df['HSCobjid'].unique():
        _df = df[(df['HSCobjid']==objid) & (df['scores']>=SCORE_THRESHOLD) & (~df['is_central']) & (df['labels'].isin([1,2]))]
    
        if len(_df) > 0:
            # load fits
            for band in bands:
                if band=='U':
                    _fits_file = FITS_FILE_PATH + str(objid)[-3:] + '/' + str(objid) + '_CLAUDS-U.fits'
                else:
                    _fits_file = FITS_FILE_PATH + str(objid)[-3:] + '/' + str(objid) + '_HSC-{}_var.fits'.format(band)
                _ff = fits.open(_fits_file)
                ff_wcs[band] = WCS(_ff[1].header)
                ff[band] = _ff[1].data
                _ff.close()
            
            # size of image for normalised coordinates
            x, y = ff[band].shape
            
            # galaxy mask
            petroR90_asec = _df['petroR90_r'].iloc[0]
            galaxy_seg_mask, _, _, _, _ = GalaxyMeasurements.create_galaxy_mask(
                ff['R'], 
                petroR90_asec, 
                pixel_scale=HSC_ARCSEC_PER_PIXEL
            )
            galaxy_seg_mask = np.logical_not(np.array(galaxy_seg_mask).astype(bool))
    
            min_seeing = int(_df[['useeing', 'gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing']].max(axis=1).max(axis=0) / HSC_ARCSEC_PER_PIXEL * .75)
            footprint = circular_footprint(min_seeing, dtype=bool)
            
            for band in bands:
                # background subtraction
                bkg_estimator = MedianBackground()
                sigma_clip = SigmaClip(sigma=3.)
                coverage_mask = (ff[band] <= 0)
                try:
                    bkg = Background2D(
                        ff[band], (min_seeing, min_seeing), 
                        filter_size=(3, 3), 
                        sigma_clip=sigma_clip, 
                        bkg_estimator=bkg_estimator,
                        coverage_mask=coverage_mask, 
                        fill_value=0.0
                    )
                    
                    science_image_bkgsub = ff[band].copy()
                    science_image_bkgsub = science_image_bkgsub - bkg.background
                    _, median, std = sigma_clipped_stats(science_image_bkgsub)
        
                    # convolve image with simple 4x4 kernel to smooth out noise
                    kernel = Box2DKernel(4)
                    convolved_data = convolve(science_image_bkgsub, kernel)
                    
                    threshold = median + (sigma * std)
                    
                    # mask clumps and shrink bboxes to avoid peaks to close to the borders
                    border_width = 0
                    for idx, data in _df.iterrows():
                        # bbox mask
                        find_peaks_mask = GalaxyMeasurements.mask_image_with_bboxes(
                            px_x1=np.rint(data['px_x1_normed']*x+border_width),
                            px_x2=np.rint(data['px_x2_normed']*x-border_width),
                            px_y1=np.rint(data['px_y1_normed']*y+border_width),
                            px_y2=np.rint(data['px_y2_normed']*y-border_width),
                            img_size=(x, y),
                            invert=True,
                            list_of_boxes=False,
                            debug=False
                        )
            
                        # final mask
                        mask = np.logical_or(galaxy_seg_mask, find_peaks_mask)
            
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', NoDetectionsWarning)
                            tbl = find_peaks(convolved_data, threshold, footprint=footprint, mask=mask, wcs=ff_wcs[band], centroid_func=centroid_com)
            
                        if tbl is not None:
                            results.append([
                                objid,
                                band,
                                data['clump_id'],
                                data['detection_id'],
                                data['labels'],
                                data['scores'],
                                True,
                                tbl['x_peak'],
                                tbl['y_peak'],
                                tbl['x_centroid'],
                                tbl['y_centroid'],
                                tbl['skycoord_peak'].ra.degree,
                                tbl['skycoord_peak'].dec.degree,
                                tbl['skycoord_centroid'].ra.degree,
                                tbl['skycoord_centroid'].dec.degree,
                                tbl['peak_value'],                                       
                            ])
                        else:
                            # check, if clump is inside galaxy mask as this has not been done before
                            # apply inverse mask, switch x-y to y-x because of the shape of the array
                            # use clump bbox midpoints
                            clump_centre_x = (data['px_x1_normed']*x + data['px_x2_normed']*x) / 2.0
                            clump_centre_y = (data['px_y1_normed']*x + data['px_y2_normed']*x) / 2.0
                            if not galaxy_seg_mask[(int(clump_centre_y), int(clump_centre_x))]:
                                # Re-project coords
                                skycoord_clump_centre = ff_wcs[band].wcs_pix2world(clump_centre_x, clump_centre_y, 1)
                                results.append([
                                    objid,
                                    band,
                                    data['clump_id'],
                                    data['detection_id'],
                                    data['labels'],
                                    data['scores'],
                                    False,
                                    clump_centre_x,
                                    clump_centre_y,
                                    clump_centre_x,
                                    clump_centre_y,
                                    skycoord_clump_centre[0],
                                    skycoord_clump_centre[1],
                                    skycoord_clump_centre[0],
                                    skycoord_clump_centre[1],
                                    np.nan,                                       
                                ])
        
                except ValueError:
                    error_list.append([objid])
    
    df_results = (
        pd.DataFrame(
            results, 
            columns=[
                'HSCobjid', 'band',
                'clump_id', 'detection_id', 'labels', 'scores', 'is_peak_detection',
                'x_peak', 'y_peak',
                'x_centroid', 'y_centroid',
                'ra_peak', 'dec_peak',
                'ra_centroid', 'dec_centroid',
                'peak_value',
            ]
        )
        .explode([
            'x_peak', 'y_peak', 
            'x_centroid', 'y_centroid',
            'ra_peak', 'dec_peak',
            'ra_centroid', 'dec_centroid',
            'peak_value',
        ])
    )
    return df_results
# [END classes and functions]


# [START Main]
def main():
    # load data
    # adjust load_data() function accordingly
    print('Loading data...')
    df_pred = load_data()
    cols = [
        'HSCobjid', 'scores', 'is_central', 'labels', 'petroR90_r', 
        'useeing', 'gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing',
        'clump_id', 'detection_id',
        'px_x1_normed', 'px_x2_normed', 'px_y1_normed', 'px_y2_normed',
    ]
    assert(all(col in df_pred.columns for col in cols))

    # split data into chunks based on HSCobjid
    print('Splitting data...')
    chunks = np.array_split(df_pred['HSCobjid'].unique(), NUM_WORKERS)
    df_chunks = [df_pred[df_pred['HSCobjid'].isin(chunk)] for chunk in chunks]

    # [START peak finder]
    start = time.time()
    print('Progressing peak_finder...')
    bands = 'UGRIZY' # for CLAUDS+HSC
    # bands = 'GRIZY' # for HSC only
    sigma = 1.0

    # create list for function call via multiprocessing
    mp_chunks = zip(df_chunks, repeat(bands), repeat(sigma))

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        data = pool.starmap(peak_finder, mp_chunks)
    df_results = pd.concat(data)

    elapsed = (time.time() - start)  
    print('Finished peak_finder in {} seconds.'.format(elapsed))
    # [END peak finder]

    # [START POST-PROCESSING]
    # assign peak_id to all peaks found
    df_results = (df_results
        .reset_index(drop=True)
        .assign(x_peak = lambda _df: _df.x_peak.astype(float))
        .assign(y_peak = lambda _df: _df.y_peak.astype(float))
        .assign(x_centroid = lambda _df: _df.x_centroid.astype(float))
        .assign(y_centroid = lambda _df: _df.y_centroid.astype(float))
        .assign(ra_peak = lambda _df: _df.ra_peak.astype(float))
        .assign(dec_peak = lambda _df: _df.dec_peak.astype(float))
        .assign(ra_centroid = lambda _df: _df.ra_centroid.astype(float))
        .assign(dec_centroid = lambda _df: _df.dec_centroid.astype(float))
        .assign(peak_value = lambda _df: _df.peak_value.astype(float))
    )
    
    df_results['peak_id'] = df_results.groupby(['HSCobjid', 'clump_id'])['x_peak'].rank(method='first').astype(int)
    df_results['peak_detection_id'] = df_results.index + 1

    # remove peaks that are to close to the boundary of the bounding box
    padding = 0.005
    
    df_fits_meta = pd.read_parquet(DATA_PATH + 'galaxies_HSCfull_fits_meta.gzip', engine='pyarrow')
    # finale columns for output
    cols = [
        'HSCobjid', 'band', 'clump_id', 'detection_id', 
        'labels', 'scores', 'is_peak_detection',
        'x_peak', 'y_peak', 
        'x_centroid', 'y_centroid',
        'ra_peak', 'dec_peak', 
        'ra_centroid', 'dec_centroid',
        'peak_value', 
        'peak_id', 'peak_detection_id', 
        'x_peak_normed', 'y_peak_normed',
        'x_centroid_normed', 'y_centroid_normed',
    ]
    
    df_peaks = (df_results
        .merge(df_pred[['detection_id', 'px_x1_normed', 'px_x2_normed', 'px_y1_normed', 'px_y2_normed']], how='inner', on='detection_id')
        .merge(df_fits_meta[df_fits_meta['band']=='G'][['HSCobjid', 'fluxMag0', 'px_fits_x', 'px_fits_y']], how='inner', on='HSCobjid')
        .assign(x_peak_normed = lambda _df: _df.x_peak/_df.px_fits_x)
        .assign(y_peak_normed = lambda _df: _df.y_peak/_df.px_fits_y)
        .assign(x_centroid_normed = lambda _df: _df.x_centroid/_df.px_fits_x)
        .assign(y_centroid_normed = lambda _df: _df.y_centroid/_df.px_fits_y)
        .assign(is_in_padding = False)
        .assign(is_in_padding = lambda _df: _df.is_in_padding.where(
            (_df.x_centroid_normed > (_df.px_x1_normed+padding)) & (_df.x_centroid_normed < (_df.px_x2_normed-padding)) & (_df.y_centroid_normed > (_df.px_y1_normed+padding)) & (_df.y_centroid_normed < (_df.px_y2_normed-padding))
        , True))
        .query('is_in_padding == False')
        .assign(x_peak_normed = lambda _df: _df.x_peak_normed.astype(float))
        .assign(y_peak_normed = lambda _df: _df.y_peak_normed.astype(float))
        .assign(x_centroid_normed = lambda _df: _df.x_centroid_normed.astype(float))
        .assign(y_centroid_normed = lambda _df: _df.y_centroid_normed.astype(float))
    )
    
    # removing duplicates
    duplicates = df_peaks.groupby(['HSCobjid', 'band', 'x_centroid_normed', 'y_centroid_normed']).filter(lambda x: len(x) > 1)  #  HAVING COUNT(*) > 1
    duplicates['duplicate_id'] = duplicates.groupby(['HSCobjid', 'band'])['scores'].rank(ascending=False, method='max').astype(int)
    peaks_to_remove = duplicates[duplicates['duplicate_id']!=1]['peak_detection_id']
    df_peaks = df_peaks[~df_peaks['peak_detection_id'].isin(peaks_to_remove)][cols]
    # [END POST-PROCESSING]

    # [START FINISHING-UP]
    print('Writing results...')
    output_file = 'peaks_Zoobot-U+GRIZY_ensemble_CLAUDS+HSC_bands_sigma_{}_cleaned.gzip'.format(sigma)
    df_peaks.to_parquet(PREDICTIONS_FILE_PATH + output_file, compression='gzip')
    # [END FINISHING-UP]
# [END Main]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
# [END all]