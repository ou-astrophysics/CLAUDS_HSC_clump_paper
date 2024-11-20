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

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.convolution import convolve_fft
from astropy.modeling.models import Moffat2D

import photutils
from photutils.utils import calc_total_error
from photutils.psf.matching import create_matching_kernel, SplitCosineBellWindow
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

# small aperture, but will be corrected by correction factor derived from measuring
# star-like objects from the CLAUDS and HSC catalogue
FWHM_MULTIPLIER = 0.5
FWHM_MULTIPLIER_ANNULUS_MIN = 1.0
FWHM_MULTIPLIER_ANNULUS_MAX = 1.5

# Zeropoints for cnversion of ADU into nJy
ZEROPOINT_DICT = {
    'U': 30.0 * u.ABmag,
    'G': 27.0 * u.ABmag,
    'R': 27.0 * u.ABmag,
    'I': 27.0 * u.ABmag,
    'Z': 27.0 * u.ABmag,
    'Y': 27.0 * u.ABmag,
}

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


def get_aperture_correction(aperture_radius: float, psf_fwhm: float) -> float:
    """
      Calculates the aperture correction factor for a given aperture and PSF-FWHM.
      A standard 2D-Moffat function is evaluated over a grid of 10x10px and the 
      enclosed flux within the given aperture is summed to give a ratio of total
      flux and flux within the aperture.

      Args:
        aperture_radius : radius of the aperture for which the correction should be calculated
        psf_fwhm : FWHM of the Moffat-function, e.g. typical seeing

      Returns:
        Aperture correction factor
    """

    flux = 1.0
    beta = 2.5
    alpha = psf_fwhm / (2*np.sqrt(2**(1/beta)-1))
    amplitude = flux * (beta-1) / (np.pi*alpha**2)
    x_0 = y_0 = 0.0

    dx = dy = 0.01
    x = np.arange(-10, 10, dx)
    y = np.arange(-10, 10, dy)

    yy, xx = np.meshgrid(y, x, sparse=True)
    moffat_see = Moffat2D().evaluate(xx, yy, amplitude=amplitude, x_0=x_0, y_0=y_0, gamma=alpha, alpha=beta)

    cond_aper = np.sqrt(xx**2 + yy**2) <= aperture_radius
    flux_aper = np.sum(moffat_see[cond_aper]) * dx * dy

    aper_adjust = flux/flux_aper

    return aper_adjust


def run_photometry(df_phot, bands):
    results = []
    fits_files = {}
    psf_files = {}

    # define the columns to keep for the final dataframe
    # depends on input data from load_data()
    cols_to_keep = [
        'HSCobjid', 'band', 'clump_id', 'detection_id', 'labels', 'scores', 
        'is_peak_detection', 'x_peak', 'y_peak', 'x_centroid', 'y_centroid', 'ra_peak', 'dec_peak',
        'ra_centroid', 'dec_centroid', 'peak_value', 'peak_id',
        'peak_detection_id', 'x_peak_normed', 'y_peak_normed',
        'x_centroid_normed', 'y_centroid_normed',
        # these columns will be added later in the script
        'aperture_radius_px', 'annulus_min_px', 'annulus_max_px', 'aper_corr'
    ]
    
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
        
        _df = df_phot[(df_phot['HSCobjid']==objid)].copy() # keeping the original data

        if len(_df) > 0:
            # get the filterband for which the PSF is the largest to use as reference_band for convolving the FITS
            reference_band = _df['reference_band'].iloc[0]
            # and pad the image to the largest size
            x, y = psf_files[reference_band][0].data.shape
            x_pad = (psf_x_max - x) // 2
            y_pad = (psf_y_max - y) // 2
            reference_psf = np.pad(psf_files[reference_band][0].data, ((x_pad, x_pad), (y_pad, y_pad)), 'edge')

            # using worst seeing for aperture radius
            aperture_px = _df['reference_psf_fwhm'].iloc[0] / HSC_ARCSEC_PER_PIXEL * FWHM_MULTIPLIER
            annulus_min_px = _df['reference_psf_fwhm'].iloc[0] / HSC_ARCSEC_PER_PIXEL * FWHM_MULTIPLIER_ANNULUS_MIN
            annulus_max_px = _df['reference_psf_fwhm'].iloc[0] / HSC_ARCSEC_PER_PIXEL * FWHM_MULTIPLIER_ANNULUS_MAX

            # add aperture and annuli radii to resulting dataFrame
            _df['aperture_radius_px'] = aperture_px
            _df['annulus_min_px'] = annulus_min_px
            _df['annulus_max_px'] = annulus_max_px

            # get the aperture correction
            aper_corr = get_aperture_correction(aperture_radius=aperture_px, psf_fwhm=_df['reference_psf_fwhm'].iloc[0] / HSC_ARCSEC_PER_PIXEL)
            _df['aper_corr'] = aper_corr

            # limit to columns to keep for resulting dataFrame
            _df_phot_list = [_df[cols_to_keep].reset_index()]

            # px-coords for the circular aperture
            positions = [tuple(r) for r in _df[['x_centroid', 'y_centroid']].to_numpy().tolist()]

            for i, (band, f_file) in enumerate(fits_files.items()):
                # define image files and stuff
                science_image = f_file[1].data
                x, y = science_image.shape
                imwcs = WCS(f_file[1].header)
                var_image = f_file[3].data
                
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
                
                # doing aperture photometry
                # masking clumps, all peaks plus adjacent peaks from DAOStarFinder
                masked_clumps = GalaxyMeasurements.clump_mask(
                    px_x=_df['x_centroid'],
                    px_y=_df['y_centroid'],
                    radius=aperture_px,
                    img_size=science_image.shape,
                    # debug=True
                )
                # setting masked pixel to nan for background substraction
                masked_fits_image = science_image.copy()
                masked_fits_image[masked_clumps] = np.nan

                _aperture = photutils.aperture.CircularAperture(positions, r=aperture_px)
                _annulus_aperture = photutils.aperture.CircularAnnulus(positions, r_in=annulus_min_px, r_out=annulus_max_px)
                
                sigclip = SigmaClip(sigma=3.0, maxiters=10)
                _phot_table = photutils.aperture.aperture_photometry(science_image, _aperture, error=error)
                _bkgstats = photutils.aperture.ApertureStats(masked_fits_image, _annulus_aperture, error=error, sigma_clip=sigclip)
            
                # creating photo-stats table
                _phot_table.remove_column('id')
                _phot_table.remove_column('xcenter')
                _phot_table.remove_column('ycenter')
                _phot_table.rename_column('aperture_sum', 'aperture_sum_'+band.lower())
                _phot_table.rename_column('aperture_sum_err', 'aperture_sum_err_'+band.lower())
                _phot_table['total_bkg_'+band.lower()] = _bkgstats.median * _aperture.area
                _phot_table['aperture_sum_bkgsub_'+band.lower()] = _phot_table['aperture_sum_'+band.lower()] - _phot_table['total_bkg_'+band.lower()]
                # applying aperture correction
                _phot_table['clump_flux_ADU_corr_'+band.lower()] = _phot_table['aperture_sum_bkgsub_'+band.lower()] * aper_corr
                _phot_table['clump_flux_ADU_err_corr_'+band.lower()] = _phot_table['aperture_sum_err_'+band.lower()] * aper_corr
                # background stats from annulus
                _phot_table['bkg_median_'+band.lower()] = _bkgstats.median
                _phot_table['bkg_mad_std_'+band.lower()] = _bkgstats.mad_std
                _phot_table['bkg_mean_'+band.lower()] = _bkgstats.mean
                _phot_table['bkg_std_'+band.lower()] = _bkgstats.std
                # converting ADUs into nJy
                _phot_table['clump_flux_nJy_'+band.lower()] = _phot_table['clump_flux_ADU_corr_'+band.lower()] * ZEROPOINT_DICT[band].to(u.nJy)
                _phot_table['clump_flux_nJy_err_'+band.lower()] = _phot_table['clump_flux_ADU_err_corr_'+band.lower()] * ZEROPOINT_DICT[band].to(u.nJy)
                # converting flux into ABmags
                _phot_table['clump_flux_ABmag_'+band.lower()] = _phot_table['clump_flux_nJy_'+band.lower()].to(u.ABmag)
                _phot_table['clump_flux_ABmag_err_'+band.lower()] = (2.5 * np.log10(1 + _phot_table['clump_flux_nJy_err_'+band.lower()]/_phot_table['clump_flux_nJy_'+band.lower()])).value * u.ABmag

                _df_phot_list.append(_phot_table.to_pandas())

            results.append(pd.concat(_df_phot_list, axis=1, ignore_index=False))
        
        else:
            print('--> row skipped')
        # close FITS
        for i, (band, f_file) in enumerate(fits_files.items()):
            f_file.close()
        for i, (band, f_file) in enumerate(psf_files.items()):
            f_file.close()
        
    df_result_chunk = pd.concat(results)
    return df_result_chunk
# [END classes and functions]


# [START Main]
def main():
    # load data
    print('Loading data...')
    df_pred = load_data()
    print('We\'ve got {} peaks, {} clumps and {} galaxies'.format(len(df_pred), df_pred['detection_id'].nunique(), df_pred['HSCobjid'].nunique()))

    # split data into chunks based on HSCobjid
    print('Splitting data...')
    chunks = np.array_split(df_pred['HSCobjid'].unique(), NUM_WORKERS)
    df_chunks = [df_pred[df_pred['HSCobjid'].isin(chunk)] for chunk in chunks]

    # [START photometry]
    start = time.time()
    print('Starting aperture photometry...')
    bands = 'UGRIZY' # for CLAUDS+HSC
    # bands = 'GRIZY' # for HSC only
    print('...on bands: {}, using {} cores...'.format(bands, NUM_WORKERS))

    # create list for function call via multiprocessing
    mp_chunks = zip(df_chunks, repeat(bands))

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        data = pool.starmap(run_photometry, mp_chunks)
    df_results = pd.concat(data)

    elapsed = (time.time() - start)  
    print('Finished aperture photometry in {} seconds.'.format(elapsed))
    # [END photometry]

    # [START POST-PROCESSING]
    # [END POST-PROCESSING]

    # [START FINISHING-UP]
    print('Writing results...')
    print('... for {} peaks, {} clumps and {} galaxies...'.format(len(df_results), df_results['detection_id'].nunique(), df_results['HSCobjid'].nunique()))
    output_file = 'phot_out.gzip'
    print('to {}.'.format(PHOTOMETRY_FILE_PATH + output_file))
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