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
sys.path.append('../shared/')
import GalaxyMeasurements
from itertools import repeat
from operator import itemgetter
import multiprocessing

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import convolve_fft

import photutils
from photutils.utils import calc_total_error
from photutils.psf.matching import create_matching_kernel, SplitCosineBellWindow
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceCatalog, SegmentationImage
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
def get_reference_band(u, g, r, i, z, y):
    d = {
        'U': u,
        'G': g,
        'R': r,
        'I': i,
        'Z': z,
        'Y': y,
    }
    return max(d, key=d.get)

def load_data():
    """
        Function for loading the necessary dataFrame.
        Change accordingly, example below is for reading CLAUDS+HSC crossmatches

        Args: none
        Returns: Pandas DataFrame
    """

    # define your data set, here some dataframes saved as parquet-files
    df_meta = pd.read_parquet(DATA_PATH + 'galaxy_image_meta_information_file.gzip', engine='pyarrow')
    df_phot = (pd
        .read_parquet(PREDICTIONS_FILE_PATH + 'predictions_file_name.gzip', engine='pyarrow')
        .query('band == "U"')
        .merge(df_meta, how='inner', on='HSCobjid')
    )
    return df_phot


def run_photometry(df_phot, bands, fwhm_multiplier_annulus_min, fwhm_multiplier_annulus_max, apertures_count):
    results = []
    fits_files = {}
    psf_files = {}
    
    cols_to_keep = [
        'HSCobjid', 'band', 'clump_id', 'detection_id', 'labels', 'scores', 
        'is_peak_detection', 'x_peak', 'y_peak', 'x_centroid', 'y_centroid', 'ra_peak', 'dec_peak',
        'ra_centroid', 'dec_centroid', 'peak_value', 'peak_id',
        'peak_detection_id', 'x_peak_normed', 'y_peak_normed',
        'x_centroid_normed', 'y_centroid_normed', 'aperture_radius_step'
    ]
    cols_to_keep.extend(['aperture_radius_'+str(i) for i in range(apertures_count)])
    
    window = SplitCosineBellWindow(alpha=0.35, beta=0.3)
    
    for objid in df_phot['HSCobjid'].unique():
        for band in bands:
            if band=='U':
                _fits_file = FITS_FILE_PATH + str(objid)[-3:] + '/' + str(objid) + '_CLAUDS-U.fits'
                _psf_file =  PSF_FILE_PATH  + str(objid)[-3:] + '/' + str(objid) + '_PSF_CLAUDS-U.fits'
            else:
                _fits_file = FITS_FILE_PATH + str(objid)[-3:] + '/' + str(objid) + '_HSC-{}_var.fits'.format(band)
                _psf_file =  PSF_FILE_PATH  + str(objid)[-3:] + '/' + str(objid) + '_PSF_HSC-{}.fits'.format(band)
            fits_files[band] = fits.open(_fits_file)
            psf_files[band] = fits.open(_psf_file)

        # get max size of the PSF images, they can vary!
        psf_sizes = [psf_files[band][0].data.shape for band in bands]
        psf_x_max = max(psf_sizes, key=itemgetter(0))[0]
        psf_y_max = max(psf_sizes, key=itemgetter(1))[1]
        
        _df = df_phot[(df_phot['HSCobjid']==objid)].copy()
    
        # get the filterband for which the PSF is the largest to use as reference_band for convolving the FITS
        reference_band = _df['reference_band'].iloc[0]
        # and pad the image to the largest size
        x, y = psf_files[reference_band][0].data.shape
        x_pad = (psf_x_max - x) // 2
        y_pad = (psf_y_max - y) // 2
        reference_psf = np.pad(psf_files[reference_band][0].data, ((x_pad, x_pad), (y_pad, y_pad)), 'edge')
        
        if len(_df) > 0:
            # px-coords for the circular aperture
            positions = [tuple(r) for r in _df[['x_centroid', 'y_centroid']].to_numpy().tolist()]
            # using worst seeing for aperture radius
            aperture_radius = _df['reference_psf_fwhm'].iloc[0] / HSC_ARCSEC_PER_PIXEL / 2
            # define increasing radii for measuring intensity curve, used later for aperture correction
            aperture_radii, aperture_radii_step = np.linspace(
                aperture_radius, 
                aperture_radius*fwhm_multiplier_annulus_min, 
                num=apertures_count, 
                endpoint=True, 
                retstep=True, 
                dtype=float
            )
            # add aperture radii to resulting dataFrame
            _df['aperture_radius_step'] = aperture_radii_step
            for i in range(apertures_count):
                _df['aperture_radius_'+str(i)] = aperture_radii[i]
            
            # add columns to keep for resulting dataFrame
            _df_phot_list = [_df[cols_to_keep].reset_index()]
            
            for i, (band, f_file) in enumerate(fits_files.items()):
                # define image files and stuff
                science_image = f_file[1].data
                x, y = science_image.shape
                imwcs = WCS(f_file[1].header)
                var_image = f_file[3].data
    
                # masking clumps
                masked_clumps = GalaxyMeasurements.clump_mask(
                    px_x=_df['x_centroid'],
                    px_y=_df['y_centroid'],
                    radius=aperture_radius,
                    img_size=science_image.shape,
                    # debug=True
                )
                
                # Data RMS uncertainty is combination of background RMS and source Poisson uncertainties
                # HSC-weight = inverse variance map
                # CLAUDS-weight = weight map
                if band == 'U':
                    background_rms = 1 / np.sqrt(var_image)
                    effective_gain = 1.62
                else:
                    background_rms = np.sqrt(var_image)
                    effective_gain = 3.0
                error = calc_total_error(data=science_image, bkg_error=background_rms, effective_gain=effective_gain)
    
                # convolve fits with new PSF
                # if reference_band != band:
                # pad the image to the largest size
                x, y = psf_files[band][0].data.shape
                x_pad = (psf_x_max - x) // 2
                y_pad = (psf_y_max - y) // 2
                psf = np.pad(psf_files[band][0].data, ((x_pad, x_pad), (y_pad, y_pad)), 'edge')
                # higher resolution first, then reference
                kernel = create_matching_kernel(psf, reference_psf, window=window)
                science_image = convolve_fft(science_image, kernel, boundary='wrap')
    
                # setting masked pixel to nan for background substraction
                masked_fits_image = science_image.copy()
                masked_fits_image[masked_clumps] = np.nan
                
                # doing aperture photometry
                _aperture = [photutils.aperture.CircularAperture(positions, r=r) for r in aperture_radii]
                _annulus_aperture = photutils.aperture.CircularAnnulus(positions, r_in=aperture_radius*fwhm_multiplier_annulus_min, r_out=aperture_radius*fwhm_multiplier_annulus_max)
                
                sigclip = SigmaClip(sigma=3.0, maxiters=10)
                _phot_table = photutils.aperture.aperture_photometry(science_image, _aperture, error=error)
                _bkgstats = photutils.aperture.ApertureStats(science_image, _annulus_aperture, error=error, sigma_clip=sigclip)
            
                # and with clumps masked:
                _bkgstats_mask = photutils.aperture.ApertureStats(masked_fits_image, _annulus_aperture, error=error, sigma_clip=sigclip)
            
                # creating photo-stats table
                _phot_table.remove_column('id')
                _phot_table.remove_column('xcenter')
                _phot_table.remove_column('ycenter')
                for i in range(apertures_count):
                    _phot_table.rename_column('aperture_sum_'+str(i) , 'aperture_sum_'+str(i)+'_'+band.lower())
                    _phot_table.rename_column('aperture_sum_err_'+str(i) , 'aperture_sum_err_'+str(i)+'_'+band.lower())
                    _phot_table['total_bkg_'+str(i)+'_'+band.lower()] = _bkgstats.median * _aperture[i].area
                    _phot_table['aperture_sum_bkgsub_'+str(i)+'_'+band.lower()] = _phot_table['aperture_sum_'+str(i)+'_'+band.lower()] - _phot_table['total_bkg_'+str(i)+'_'+band.lower()]
                    _phot_table['total_bkg_masked_'+str(i)+'_'+band.lower()] = _bkgstats_mask.median * _aperture[i].area
                    _phot_table['aperture_sum_bkgsub_masked_'+str(i)+'_'+band.lower()] = _phot_table['aperture_sum_'+str(i)+'_'+band.lower()] - _phot_table['total_bkg_masked_'+str(i)+'_'+band.lower()]
                _df_phot_list.append(_phot_table.to_pandas())
            
            results.append(pd.concat(_df_phot_list, axis=1, ignore_index=False))
        
        # close FITS
        for i, (band, f_file) in enumerate(fits_files.items()):
            f_file.close()
        for i, (band, f_file) in enumerate(psf_files.items()):
            f_file.close()
        
    df_result_chunk = pd.concat(results)
    return df_result_chunk


def calc_AB_mags(df, bands, zeropoint_dict, aperture_count):
    """
      Function to add calculated ABmags and ABmag errors to the resulting dataframe
      given the corresponding zeropoints in a dict.
      Args:
        df - dataframe to modify
        bands - string/list with filterbands measured, e.g. 'UGRIZY'
        zeropoint_dict - dictionary, key: band, value: zeropoint, e.g. {'U': 27.0}
        aperture_count - int, number of apertures measured
      Returns:
        none, changes df directly
    """

    for band in bands:
        zeropoint = zeropoint_dict[band]
        for i in range(aperture_count):
            df[['clump_mag_'+str(i)+'_'+band.lower(), 'clump_mag_err_'+str(i)+'_'+band.lower()]] = df.apply(
                lambda x: GalaxyMeasurements.fluxes2mags(
                    x['aperture_sum_bkgsub_masked_'+str(i)+'_'+band.lower()], 
                    x['aperture_sum_err_'+str(i)+'_'+band.lower()], 
                    zeropoint
                ), axis=1, result_type='expand'
            )
# [END classes and functions]


# [START Main]
def main():
    # some settings, probably replace them later with command line args
    fwhm_multiplier_annulus_min = 2.0
    fwhm_multiplier_annulus_max = 3.0
    apertures_count = 5
    
    # load data
    print('Loading data...')
    df_pred = load_data()
    cols = [
        'HSCobjid', 'band', 'clump_id', 'detection_id', 'labels', 'scores', 'is_peak_detection',
        'x_peak', 'y_peak', 'x_centroid', 'y_centroid', 'ra_peak', 'dec_peak',
        'ra_centroid', 'dec_centroid', 'peak_value', 'peak_id',
        'peak_detection_id', 'x_peak_normed', 'y_peak_normed',
        'x_centroid_normed', 'y_centroid_normed',
        'petroR90_r', 'useeing', 'gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing'
    ]
    assert(all(col in df_pred.columns for col in cols))

    # split data into chunks based on HSCobjid
    print('Splitting data...')
    chunks = np.array_split(df_pred['HSCobjid'].unique(), NUM_WORKERS)
    df_chunks = [df_pred[df_pred['HSCobjid'].isin(chunk)] for chunk in chunks]

    # [START photometry]
    start = time.time()
    print('Starting aperture photometry...')
    bands = 'UGRIZY' # for CLAUDS+HSC
    # bands = 'GRIZY' # for HSC only

    # create list for function call via multiprocessing
    mp_chunks = zip(df_chunks, repeat(bands), repeat(fwhm_multiplier_annulus_min), repeat(fwhm_multiplier_annulus_max), repeat(apertures_count))

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        data = pool.starmap(run_photometry, mp_chunks)
    df_results = pd.concat(data)

    elapsed = (time.time() - start)  
    print('Finished aperture photometry in {} seconds.'.format(elapsed))
    # [END photometry]

    # [START POST-PROCESSING]
    print('Starting ABmag conversion...')
    zp = {
        'U': 30.0,
        'G': 27.0,
        'R': 27.0,
        'I': 27.0,
        'Z': 27.0,
        'Y': 27.0,
    }
    calc_AB_mags(df_results, bands, zp, apertures_count)
    # [END POST-PROCESSING]

    # [START FINISHING-UP]
    print('Writing results...')
    output_file = 'phot_Zoobot-U+GRIZY_ensemble_HSC_sigma_1.0.gzip'
    df_results.to_parquet(PHOTOMETRY_FILE_PATH + output_file, compression='gzip')
    # [END FINISHING-UP]
# [END Main]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
# [END all]