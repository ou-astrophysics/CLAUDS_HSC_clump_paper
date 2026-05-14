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
import argparse
sys.path.append('../shared/')
import GalaxyMeasurements

from itertools import repeat
from operator import itemgetter
import multiprocessing

import astropy.units as u
from astropy.io import fits
from astropy.stats import SigmaClip

import photutils
from photutils.psf import ImagePSF
from photutils.utils import calc_total_error
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


def load_data():
    """
        Function for loading the necessary dataFrame.
        Change accordingly, example below is for reading CLAUDS+HSC crossmatches

        Args: none
        Returns: Pandas DataFrame
    """

    cols = ['HSCobjid', 'useeing', 'gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing']
    df_galaxies = (pd
        .read_parquet(DATA_PATH + 'galaxies_HSCfull.gzip', engine='pyarrow')
        .query('GRIZY_exists==True & U_band_exists==True & PSF_U_band_exist==True & is_data_U==True')
        .assign(useeing = lambda _df: _df.useeing.where(_df.useeing>0.0, 0.919000))
        .assign(gseeing = lambda _df: _df.gseeing.where(_df.gseeing>0.0, 0.793401))
        .assign(rseeing = lambda _df: _df.rseeing.where(_df.rseeing>0.0, 0.760218))
        .assign(iseeing = lambda _df: _df.iseeing.where(_df.iseeing>0.0, 0.607064))
        .assign(zseeing = lambda _df: _df.zseeing.where(_df.zseeing>0.0, 0.687697))
        .assign(yseeing = lambda _df: _df.yseeing.where(_df.yseeing>0.0, 0.694443))
        [cols]
    )
    
    df_phot = (pd
        .read_parquet(PREDICTIONS_FILE_PATH + 'peaks_Zoobot-U+GRIZY_ensemble_CLAUDS+HSC_bands_sigma_1.0_cleaned.gzip', engine='pyarrow')
        .query('band == "U"')
        .merge(df_galaxies, how='inner', on='HSCobjid')
    )
    return df_phot


def get_aperture_correction(aperture_radius: float, psf_data: np.ndarray, oversampling_rate: int=1) -> float:
    """
      Calculates the aperture correction for a given PSF-model image.
      The image PSF is evaluated over a grid of 20x20px and the enclosed
      flux within the given aperture is summed to give a ratio of total
      flux and flux within the aperture.

      Args:
        aperture_radius : radius of the aperture for which the correction should be calculated
        psf_data : PSF-model image as numpy array
        oversampling_rate : oversampling of the PSF-model image

      Returns:
        Aperture correction factor
    """

    flux = 1.0
    x_0 = y_0 = 0.0

    dx = dy = 0.01
    x = np.arange(-10, 10, dx)
    y = np.arange(-10, 10, dy)

    yy, xx = np.meshgrid(y, x, sparse=True)
    psf_model = ImagePSF(psf_data, oversampling=oversampling_rate)
    psf_see = psf_model.evaluate(
        x=xx, y=yy, 
        flux=flux, 
        x_0=x_0, y_0=y_0
    )

    cond_aper = np.sqrt(xx**2 + yy**2) <= aperture_radius
    flux_aper = np.sum(psf_see[cond_aper]) * dx * dy

    aper_adjust = flux/flux_aper

    return aper_adjust 


def run_photometry(df_phot, bands, fwhm_multiplier, fwhm_multiplier_annulus_min, fwhm_multiplier_annulus_max, clump_mask_radius):
    results = []

    cols_to_keep = [
        'HSCobjid', 'band', 'clump_id', 'detection_id', 'labels', 'scores', 
        'is_peak_detection', 'x_peak', 'y_peak', 'x_centroid', 'y_centroid', 'ra_peak', 'dec_peak',
        'ra_centroid', 'dec_centroid', 'peak_value', 'peak_id',
        'peak_detection_id', 'x_peak_normed', 'y_peak_normed',
        'x_centroid_normed', 'y_centroid_normed',
    ]

    for objid in df_phot['HSCobjid'].unique():
        _df = df_phot[(df_phot['HSCobjid']==objid)].copy() # keeping the original data
        if len(_df) > 0:
            # px-coords for the circular aperture
            positions = [tuple(r) for r in _df[['sim_x_centroid', 'sim_y_centroid']].to_numpy().tolist()]

            # limit to columns to keep for resulting dataFrame
            _df_phot_list = [_df[cols_to_keep].reset_index()]

            for band in bands:
                if band=='U':
                    _fits_file = FITS_FILE_PATH + str(objid)[-3:] + '/' + str(objid) + '_CLAUDS-U.fits'
                    _psf_file =  PSF_FILE_PATH  + str(objid)[-3:] + '/' + str(objid) + '_PSF_CLAUDS-U.fits'
                else:
                    _fits_file = FITS_FILE_PATH + str(objid)[-3:] + '/' + str(objid) + '_HSC-{}_var.fits'.format(band)
                    _psf_file =  PSF_FILE_PATH  + str(objid)[-3:] + '/' + str(objid) + '_PSF_HSC-{}.fits'.format(band)
                f_file = fits.open(_fits_file)
                psf_file = fits.open(_psf_file)

                # using seeing for aperture radius and other settings
                seeing_px = _df[band.lower()+'seeing'].iloc[0] / HSC_ARCSEC_PER_PIXEL
                oversampling_rate = 2 if band != 'U' and seeing_px <= 3.0 else 1
                aperture_px = seeing_px * fwhm_multiplier
                annulus_min_px = seeing_px * fwhm_multiplier_annulus_min
                annulus_max_px = seeing_px * fwhm_multiplier_annulus_max

                # define image files and stuff
                science_image = f_file[1].data + f_file[4].data
                # science_image = f_file[1].data # measure the background only w/o simulated clumps for testing the background influence
                # science_image = f_file[4].data # use only the bare simulated clumps for testing how flux is recovered
                # imwcs = WCS(f_file[1].header)
                var_image = f_file[3].data
                psf_image = psf_file[0].data[3:-3,3:-3]
                
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
                
                # doing aperture photometry
                # masking clumps, all peaks plus adjacent peaks from DAOStarFinder
                masked_clumps = GalaxyMeasurements.clump_mask(
                    px_x=_df['sim_x_centroid'],
                    px_y=_df['sim_y_centroid'],
                    radius=seeing_px * clump_mask_radius,
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
                effective_median = _bkgstats.median.copy()
                # ensure that we're not adding something to the measured flux
                effective_median[effective_median<0.0] = 0.0
                
                # creating photo-stats table
                _phot_table.remove_column('id')
                _phot_table.remove_column('xcenter')
                _phot_table.remove_column('ycenter')
                _phot_table.rename_column('aperture_sum', 'aperture_sum_'+band.lower())
                _phot_table.rename_column('aperture_sum_err', 'aperture_sum_err_'+band.lower())
                _phot_table['total_bkg_'+band.lower()] = effective_median * _aperture.area
                _phot_table['total_bkg_err_'+band.lower()] = _bkgstats.mad_std
                _phot_table['aperture_area_'+band.lower()] = _aperture.area
                _phot_table['total_bkg_area_'+band.lower()] = _bkgstats.sum_aper_area
                _phot_table['aperture_total_err_'+band.lower()] = np.sqrt( (_phot_table['aperture_sum_err_'+band.lower()]**2).value + ( (np.pi/2) * (_aperture.area**2 / _bkgstats.sum_aper_area) * _bkgstats.mad_std**2 ).value )
                _phot_table['aperture_sum_bkgsub_'+band.lower()] = _phot_table['aperture_sum_'+band.lower()] - _phot_table['total_bkg_'+band.lower()]
                # applying aperture correction
                aper_corr = get_aperture_correction(aperture_radius=aperture_px, psf_data=psf_image, oversampling_rate=oversampling_rate)
                _phot_table['aper_corr_'+band.lower()] = aper_corr
                _phot_table['clump_flux_ADU_corr_'+band.lower()] = _phot_table['aperture_sum_bkgsub_'+band.lower()] * aper_corr
                _phot_table['clump_flux_ADU_err_corr_'+band.lower()] = _phot_table['aperture_total_err_'+band.lower()] * aper_corr # use total (error aperture and background) here
                _phot_table['clump_flux_ADU_'+band.lower()] = _phot_table['aperture_sum_'+band.lower()] * aper_corr
                _phot_table['clump_flux_ADU_err_'+band.lower()] = _phot_table['aperture_sum_err_'+band.lower()] * aper_corr # use only aperture error here
                # background stats from annulus
                _phot_table['bkg_median_'+band.lower()] = _bkgstats.median
                _phot_table['bkg_mad_std_'+band.lower()] = _bkgstats.mad_std
                _phot_table['bkg_mean_'+band.lower()] = _bkgstats.mean
                _phot_table['bkg_std_'+band.lower()] = _bkgstats.std
                # converting ADUs into nJy
                _phot_table['clump_flux_nJy_'+band.lower()] = _phot_table['clump_flux_ADU_corr_'+band.lower()] * ZEROPOINT_DICT[band].to(u.nJy)
                _phot_table['clump_flux_nJy_err_'+band.lower()] = _phot_table['clump_flux_ADU_err_corr_'+band.lower()] * ZEROPOINT_DICT[band].to(u.nJy)
                _phot_table['clump_flux_nJy_raw_'+band.lower()] = _phot_table['clump_flux_ADU_'+band.lower()] * ZEROPOINT_DICT[band].to(u.nJy)
                _phot_table['clump_flux_nJy_err_raw_'+band.lower()] = _phot_table['clump_flux_ADU_err_'+band.lower()] * ZEROPOINT_DICT[band].to(u.nJy)
                # converting flux into ABmags
                _phot_table['clump_flux_ABmag_'+band.lower()] = _phot_table['clump_flux_nJy_'+band.lower()].to(u.ABmag)
                _phot_table['clump_flux_ABmag_err_'+band.lower()] = (2.5 * np.log10(1 + _phot_table['clump_flux_nJy_err_'+band.lower()]/_phot_table['clump_flux_nJy_'+band.lower()])).value * u.ABmag
                _phot_table['clump_flux_ABmag_raw_'+band.lower()] = _phot_table['clump_flux_nJy_raw_'+band.lower()].to(u.ABmag)
                _phot_table['clump_flux_ABmag_err_raw_'+band.lower()] = (2.5 * np.log10(1 + _phot_table['clump_flux_nJy_err_raw_'+band.lower()]/_phot_table['clump_flux_nJy_raw_'+band.lower()])).value * u.ABmag
                # add aperture and annuli radii to resulting dataFrame
                _phot_table['aperture_radius_px_'+band.lower()] = aperture_px
                _phot_table['annulus_min_px_'+band.lower()] = annulus_min_px
                _phot_table['annulus_max_px_'+band.lower()] = annulus_max_px
                _df_phot_list.append(_phot_table.to_pandas())

                # close FITS
                f_file.close()
                psf_file.close()
            
            results.append(pd.concat(_df_phot_list, axis=1, ignore_index=False))
        
    df_result_chunk = pd.concat(results)
    return df_result_chunk
# [END classes and functions]


# [START Main]
def main():
    # Parse arguments from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--ap', dest='ap', help='aperture radius in multiples of seeing FWHM', type=float)
    parser.add_argument('--an_min', dest='an_min', help='min annulus radius in multiples of seeing FWHM', type=float)
    parser.add_argument('--an_max', dest='an_max', help='max annulus radius in multiples of seeing FWHM', type=float)
    parser.add_argument('--mask_rad', dest='mask_rad', help='radius of circular mask for adjacent clumps in multiples of seeing FWHM', type=float)

    args = parser.parse_args()

    fwhm_multiplier = args.ap
    fwhm_multiplier_annulus_min = args.an_min
    fwhm_multiplier_annulus_max = args.an_max
    clump_mask_radius = args.mask_rad

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
    mp_chunks = zip(df_chunks, repeat(bands), repeat(fwhm_multiplier), repeat(fwhm_multiplier_annulus_min), repeat(fwhm_multiplier_annulus_max), repeat(clump_mask_radius))

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        data = pool.starmap(run_photometry, mp_chunks)
    df_results = pd.concat(data)

    elapsed = (time.time() - start)  
    print('Finished aperture photometry in {} seconds.'.format(elapsed))
    # [END photometry]

    # [START POST-PROCESSING]
    df_gal_extinction = pd.read_parquet(DATA_PATH + 'HSC_gal_extinction.gzip', engine='pyarrow')

    df_results = (df_results
        .merge(df_gal_extinction, on='HSCobjid', how='inner')
        .rename(columns={
            'clump_flux_ABmag_u': 'clump_flux_ABmag_meas_u',
            'clump_flux_ABmag_g': 'clump_flux_ABmag_meas_g',
            'clump_flux_ABmag_r': 'clump_flux_ABmag_meas_r',
            'clump_flux_ABmag_i': 'clump_flux_ABmag_meas_i',
            'clump_flux_ABmag_z': 'clump_flux_ABmag_meas_z',
            'clump_flux_ABmag_y': 'clump_flux_ABmag_meas_y',
        })
        .assign(clump_flux_ABmag_u = lambda _df: _df.clump_flux_ABmag_meas_u - _df.a_u)
        .assign(clump_flux_ABmag_g = lambda _df: _df.clump_flux_ABmag_meas_g - _df.a_g)
        .assign(clump_flux_ABmag_r = lambda _df: _df.clump_flux_ABmag_meas_r - _df.a_r)
        .assign(clump_flux_ABmag_i = lambda _df: _df.clump_flux_ABmag_meas_i - _df.a_i)
        .assign(clump_flux_ABmag_z = lambda _df: _df.clump_flux_ABmag_meas_z - _df.a_z)
        .assign(clump_flux_ABmag_y = lambda _df: _df.clump_flux_ABmag_meas_y - _df.a_y)
    )    
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