#!/usr/bin/env python

# [START all]
# [START libraries]
import pandas as pd
import numpy as np
import os
import sys
import time
import argparse

from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary, describe
from prospect.models import SpecModel, priors
from prospect.models.sedmodel import SedModel
from prospect.sources import CSPSpecBasis
from prospect.fitting import lnprobfn, fit_model
from prospect.plotting import corner
from prospect.plotting.utils import best_sample, sample_posterior
from prospect.io import write_results as writer
from prospect.io import read_results as reader
from functools import partial

from itertools import repeat
import multiprocessing
# [END libraries]


# [START initial settings]
NUM_WORKERS = multiprocessing.cpu_count()
# [END initial settings]


# [START classes and functions]
def build_obs(obj_id, clump_id, peak_id, filter_set, flux_list, flux_err_list, redshift, redshift_err, is_specz, flux_units='mags'):
    """
      Creates a dictionary with parameters for a single observation.

      Args:
        obj_id : object-ID for the host galaxy, here HSCobjid
        clump_id : ID of the clump
        peak_id : ID of the detected peak of a clum
        filter_set : list of filter-names used by FSPS, prospector and sedpy
        flux_list : list of flux/brightness in the same order as filter_set
        flux_err_list : list of flux/brightness errors in the same order as filter_set
        redshift: redshift of the host galaxy
        redshift_err : redshift error of the host galaxy
        is_specz : BOOL, True if redshift of host galaxy is measured from spectroscopic data
        flux_units : either 'mags' (default) or 'maggies'

      Returns:
        obs : dictionary with observation object parameters
    """

    filters = load_filters(filter_set)
    
    if flux_units == 'mags':
        # convert the magnitudes to maggies
        maggies = np.array([10**(-0.4 * mag) for mag in flux_list])
        # convert the magnitude errors to flux uncertainties (including a noise floor)
        maggies_err = np.array([mag_err for mag_err in flux_err_list])
        maggies_err = np.hypot(maggies_err, 0.05)
        maggies_err = np.clip(maggies_err, 0.05, np.inf)
        maggies_unc = maggies_err * maggies / 1.086 # 1.086 = 1/(0.4*np.log(10))
    else:
        maggies = np.array([maggy for maggy in flux_list])
        maggies_unc = np.array([flux_err_list])

    # array of effective wavelengths for each of the filters, useful for plotting
    phot_wave = np.array([f.wave_effective for f in filters])

    obs = dict(
        obj_id=obj_id,
        clump_id=clump_id,
        peak_id=peak_id,
        wavelength=None, 
        spectrum=None, 
        unc=None, 
        redshift=redshift,
        redshift_err=redshift_err,
        is_specz=is_specz,
        maggies=maggies, 
        maggies_unc=maggies_unc,
        phot_mask=np.isfinite(np.squeeze(maggies)),
        filters=filters,
        phot_wave=phot_wave
    )
    
    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs


def build_model(sfh='delayedTau', object_redshift=None, object_redshift_err=None, object_metallicity=None, object_metallicity_err=None, add_duste=False, add_neb=False, **kwargs):
    """
      Construct a model.
      This method defines a number of parameter specification dictionaries 
      and uses them to initialize a `models.sedmodel.SedModel` object.

      Args:
        sfh : (optional, default delayed-tau model) defines the applied SFH for the model, either 'delayedTau' or 'constant'
        object_redshift : (optional, default: None) if given, the model redshift is set to this value
        object_redshift_err : (optional, default: None) if an error range for redshift is given, the redshift will also be fit using the range as prior
        object_metallicity : (optional, default: None) - if given, the model logzsol is set to this value
        object_metallicity_err : (optional, default: None) if an error range for metallicity is given, the metallicity will also be fit using the range as prior
        add_dust : (optional, default: False) - Switch to add (fixed) parameters relevant for dust emission
        add_neb : (optional, default: False) - Switch to add (fixed) parameters relevant for nebular emission, and turn nebular emission on

      Returns:
        model: `models.sedmodel.SedModel` object
    """

    # --- Get a basic delay-tau SFH parameter set. ---
    model_params = TemplateLibrary['parametric_sfh']

    if sfh == 'constant':
        # params for constant SFH
        model_params['sfh'] = {'N': 1, 'isfree': False, 'init': 4}
        model_params['const'] = {'N': 1, 'isfree': False, 'init': 1.0} # Defines the constant component (fraction) of the SFH.
        model_params['fburst'] = {'N': 1, 'isfree': False, 'init': 0.0} # Defines the fraction of mass formed in an instantaneous burst of star formation.
        model_params['tau'] = {'N': 1, 'isfree': False, 'init': 1.0} # set fixed to 1.0
    else:
        # params for delayed-tau, leave other to default
        model_params['tau'] = {'N': 1, 'isfree': True, 'init': 1.0, 'prior': priors.LogUniform(mini=1e-1, maxi=10)}

    model_params['imf_type'] = {'N': 1, 'isfree': False, 'init': 1} # Chabrier (2003)
    model_params['dust_type'] = {'N': 1, 'isfree': False, 'init': 2}  # Calzetti et al. (2000)
    model_params['dust1'] = {'N': 1, 'isfree': False, 'init': 0.0}  # required for Calzetti

    model_params['dust2'] = {'N': 1, 'isfree': True, 'init': 0.6, 'prior': priors.TopHat(mini=0.0, maxi=6.0)}
    model_params['tage'] = {'N': 1, 'isfree': True, 'init': 0.3, 'prior': priors.TopHat(mini=0.001, maxi=10.0)}
    model_params['mass'] = {'N': 1, 'isfree': True, 'init': 1e6, 'prior': priors.LogUniform(mini=1e4, maxi=1e9)}

    # Change the model parameter specifications based on some keyword arguments
    if object_redshift is not None:
        model_params['zred'] = {'N': 1, 'isfree': False, 'init': object_redshift}

    if object_redshift is not None and object_redshift_err is not None:
        model_params['zred'] = {'N': 1, 'isfree': True, 'init': object_redshift, 'prior': priors.Normal(mean=object_redshift, sigma=object_redshift_err)}

    if object_metallicity is not None:
        model_params['logzsol'] = {'N': 1, 'isfree': False, 'init': object_metallicity}

    if object_metallicity is not None and object_metallicity_err is not None:
        model_params['logzsol'] = {'N': 1, 'isfree': True, 'init': object_metallicity, 'prior': priors.TopHat(mini=-4.0, maxi=2.0)}
        # model_params['logzsol'] = {'N': 1, 'isfree': True, 'init': object_metallicity, 'prior': priors.Normal(mean=object_metallicity, sigma=object_metallicity_err)}

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary['dust_emission'])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary['nebular'])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = SedModel(model_params)

    return model


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    """
      Args:
        zcontinuous : a value of 1 insures that we use interpolation between SSPs to 
                      have a continuous metallicity parameter (`logzsol`)
    """
    sps = CSPSpecBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    
    return sps


def run_sed_fit(df, output_dir, sfh='delayedTau', filter_list='ugrizy', fit_metallicity=False, add_dust_emission=True, add_nebular=True, flux_units='mags'):
    # build the sps-object
    sps = build_sps()

    for idx, data in df.iterrows():
        obj_id = data['HSCobjid']
        clump_id = data['clump_id']
        peak_id = data['peak_id']
    
        # construct filename for results after fitting
        file_name = output_dir + str(obj_id)[-3:] + '/' + 'sed_fit_{}_{}_{}.h5'.format(obj_id, clump_id, peak_id)

        # construct list of filters
        filter_dict = {
            'u': 'cfht_megacam_u_9302', # fields: E-COSMSOS, ELAIS-N1, DEEP2-3 >> filter 'u' - cfht_megacam_u_9302
            'g': 'hsc_g_v2018', 
            'r': 'hsc_r2_v2018', 
            'i': 'hsc_i2_v2018', 
            'z': 'hsc_z_v2018', 
            'y': 'hsc_y_v2018', 
            'j': 'vista_vircam_J', 
            'h': 'vista_vircam_H', 
            'k': 'vista_vircam_Ks',
        }
        filter_dict_clauds_special = {
            'u': 'cfht_megacam_us_9301', # fields: XMM-LSS >> filter 'uS' - cfht_megacam_us_9301
            'g': 'hsc_g_v2018', 
            'r': 'hsc_r2_v2018', 
            'i': 'hsc_i2_v2018', 
            'z': 'hsc_z_v2018', 
            'y': 'hsc_y_v2018', 
            'j': 'vista_vircam_J', 
            'h': 'vista_vircam_H', 
            'k': 'vista_vircam_Ks',
        }
        if 'CLAUDS_field' in df.columns and data['CLAUDS_field'] == 'XMM-LSS':
            filters = [filter_dict_clauds_special[f] for f in filter_list]
        else:
            filters = [filter_dict[f] for f in filter_list]

        # check if field exists and make other assignments
        flux_field = 'ABmag' if flux_units == 'mags' else 'maggies'
        is_specz = data['is_specz'] if 'is_specz' in df.columns else False
        redshift_err = data['SDSS_redshift_error'] if 'SDSS_redshift_error' in df.columns else None

        # build the observation dictionary
        obs = build_obs(
            obj_id=obj_id,
            clump_id=clump_id,
            peak_id=peak_id,
            filter_set=filters,
            flux_list=[data['clump_flux_'+flux_field+'_{}'.format(band)] for band in filter_list],
            flux_err_list=[data['clump_flux_'+flux_field+'_err_{}'.format(band)] for band in filter_list],
            redshift=data['SDSS_redshift'],
            redshift_err=redshift_err,
            is_specz=is_specz,
            flux_units=flux_units, 
        )

        # build the model
        # set logzsol per args flag
        if fit_metallicity:
            object_metallicity=0.0
            object_metallicity_err=0.25
        else:
            object_metallicity=0.0
            object_metallicity_err=None
        
        # set redshift fixed for all
        object_redshift=data['SDSS_redshift']
        object_redshift_err=None
        # set redshift fixed if host galaxy has specz
        # if data['is_specz']:
        #     object_redshift=data['SDSS_redshift']
        #     object_redshift_err=None
        # else:
        #     object_redshift=data['SDSS_redshift']
        #     object_redshift_err=data['SDSS_redshift_error']
        
        model = build_model(
            sfh=sfh,
            object_redshift=object_redshift, 
            object_redshift_err=object_redshift_err, 
            object_metallicity=object_metallicity,
            object_metallicity_err=object_metallicity_err, 
            add_duste=add_dust_emission, 
            add_neb=add_nebular
        )

        # Sampling the Posterior: Nested sampling
        run_params = {}
        run_params['sps_libraries'] = sps.ssp.libraries
        run_params['dynesty'] = True
        run_params['optimize'] = False
        run_params['emcee'] = False
        run_params['nlive_init'] = 400
        run_params['nested_method'] = 'rwalk'
        run_params['nested_target_n_effective'] = 3000
        run_params['nested_dlogz_init'] = 0.05
        # run_params['nested_maxcall'] = int(1e7)
        run_params['verbose'] = False
        
        try:
            output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
            
            hfile = file_name
            writer.write_hdf5(
                hfile, 
                run_params,
                model, 
                obs,
                sampler=output['sampling'][0], 
                optimize_result_list=output['optimization'][0],
                sps=sps,
                tsample=output['sampling'][1],
                toptimize=output['optimization'][1]
            )
        except(RuntimeError):
            pass
        
    created_process = multiprocessing.Process()
    process_id = created_process._identity
    print('finished for CPU: {}'.format(process_id))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# [END classes and functions]


# [START Main]
def main():
    # Parse arguments from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data_file', help='Dataframe to process', type=str)
    parser.add_argument('--output_dir', dest='output_dir', nargs='?', default='~/sed_fits_GRIZY/', help='SED-fits destination directory', type=str)
    parser.add_argument('--sfh', dest='sfh', nargs='?', default='delayedTau', help='SFH, either delayedTau or constant', type=str)
    parser.add_argument('--filter_list', dest='filter_list', nargs='?', default='ugrizy', help='string with list of filters', type=str)
    parser.add_argument('--flux_units', dest='flux_units', nargs='?', default='mags', help='flux/brightness type, either mags or maggies', type=str)
    parser.add_argument('--fit_metal', dest='fit_metallicity', type=str2bool, nargs='?', const=False, default=False, help='Use clump-specific prior for metallicity')

    args = parser.parse_args()
    output_dir = args.output_dir
    sfh = args.sfh
    assert sfh in ['delayedTau', 'constant'], 'Not a valid SFH. Choose either delayedTau or constant.'
    filter_list = args.filter_list.lower()
    flux_units = args.flux_units
    assert flux_units in ['mags', 'maggies'], 'Not a valid flux/brightness unit. Choose either mags or maggies.'
    fit_metallicity = args.fit_metallicity
    add_dust_emission = True
    add_nebular = True

    # load data
    print('Loading data...')
    df = pd.read_parquet(args.data_file, engine='pyarrow')
    
    # sample only if result file does not yet exist
    file_exists_list = []
    for idx, data in df.iterrows():
        obj_id = data['HSCobjid']
        clump_id = data['clump_id']
        peak_id = data['peak_id']
    
        # construct filename for results after fitting
        file_name = output_dir + str(obj_id)[-3:] + '/' + 'sed_fit_{}_{}_{}.h5'.format(obj_id, clump_id, peak_id)
        if os.path.exists(file_name):
            file_exists_list.append(data['peak_detection_id'])
    # remove exisiting objects
    df = df[~df['peak_detection_id'].isin(file_exists_list)]

    print('>> We\'ve got {} peaks, {} clumps and {} galaxies'.format(len(df), df['detection_id'].nunique(), df['HSCobjid'].nunique()))

    # split data into chunks based on peak_detection_id
    print('Splitting data...')
    chunks = np.array_split(df['peak_detection_id'].unique(), NUM_WORKERS)
    df_chunks = [df[df['peak_detection_id'].isin(chunk)] for chunk in chunks]

    # [START SED-FITTING]
    start = time.time()
    print('Starting SED-fitting...')
    print('Time: {}'.format(time.ctime()))
    print('>> using {} cores...'.format(NUM_WORKERS))
    print('>> SFH: {}'.format(sfh))
    print('>> Filters: {}'.format(filter_list.upper()))
    print('>> fitting metallicity: {}'.format(fit_metallicity))

    # create list for function call via multiprocessing
    mp_chunks = zip(
        df_chunks, 
        repeat(output_dir), 
        repeat(sfh), 
        repeat(filter_list), 
        repeat(fit_metallicity), 
        repeat(add_dust_emission), 
        repeat(add_nebular),
        repeat(flux_units), 
    )

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        pool.starmap(run_sed_fit, mp_chunks)
    # [END SED-FITTING]

    # [START POST-PROCESSING]
    # [END POST-PROCESSING]

    # [START FINISHING-UP]
    elapsed = (time.time() - start)
    print('Finished SED-fitting in {} seconds.'.format(elapsed))
    # [END FINISHING-UP]
# [END Main]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
# [END all]
