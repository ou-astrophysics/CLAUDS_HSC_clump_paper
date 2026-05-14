import numpy as np
from scipy.stats.distributions import chi2
from scipy import stats
from scipy.ndimage import gaussian_filter as norm_kde
from prospect.plotting.utils import best_sample
from prospect.plotting.sfh import parametric_sfr
from dynesty import utils as dyfunc

from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary, describe
from prospect.models import SpecModel, priors
from prospect.models.sedmodel import SedModel
from prospect.sources import CSPSpecBasis


def get_quantiles(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.
    Re-used from https://github.com/bd-j/prospector/blob/main/prospect/plotting/corner.py
    
    Args:
      x : (array) samples
      q : (array-like) the list of quantiles to compute from `[0., 1.]`.
      weights : (array) The associated weights from each sample
      
    Returns:
      Returns array with the weighted sample quantiles computed at `q`.
    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError('Quantiles must be between 0. and 1.')

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError('Dimension mismatch: len(weights) != len(x).')
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def hpd(trace, mass_frac) :
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])


def get_median(sample, alpha, weights):
    ql, qm, qh = get_quantiles(sample, [alpha, 0.5, 1.0-alpha], weights=weights)
    q_minus, q_plus = qm - ql, qh - qm

    return qm, q_minus, q_plus


def get_map(max_apost, sample, alpha, weights):
    inv_q = stats.percentileofscore(np.sort(sample), max_apost, 'weak') / 100
    inv_ql = max(inv_q - 0.5 + alpha, 0.0)
    inv_qh = min(inv_q + 0.5 - alpha, 1.0)
    ql, qh = get_quantiles(sample, [inv_ql, inv_qh], weights=weights)
    q_minus, q_plus = max_apost - ql, qh - max_apost

    return q_minus, q_plus


def get_mode(sample, alpha, weights, bins):
    n, b = np.histogram(sample, bins=bins, weights=weights)
    n = norm_kde(n*1.0, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])
    xx, bins, wght = b0, b, n
    n, b = np.histogram(xx, bins=bins, weights=wght)

    bin_mode = b[np.argwhere(n==np.max(n))]
    mode_idx = np.argmin(np.abs(sample - bin_mode))    
    qm = sample[mode_idx]

    inv_q = stats.percentileofscore(np.sort(sample), qm, 'weak') / 100
    inv_ql = max(inv_q - 0.5 + alpha, 0.0)
    inv_qh = min(inv_q + 0.5 - alpha, 1.0)

    ql, qh = get_quantiles(sample, [inv_ql, inv_qh], weights=weights)
    q_minus, q_plus = qm - ql, qh - qm

    return qm, q_minus, q_plus


def collect_results(result, bands, logify, alpha=0.05, smooth=0.02):
    """
    Calcuclates the median, mode and MAP point estimate from a posterior sample
    together with a credible interval around the point estimates. Stores these
    values with photometry, SFR and maximum log-likelihood in a dict.
    
    Args:
      result : result from Prospecter fitting output
      bands : Filterbands used for the fit
      logify : (array-like) labels for which the log-values should be added
      alpha : (float, optional) credible interval, should be between 0 and 1
      smooth : same smooth-parameter Prospector uses for smoothing histograms 
               and defining smoothed bins
      
    Returns:
      Returns dict with all calculated and extracted values.
    """

    alpha = alpha/2.0
    samples = result['chain'].T
    labels = result['theta_labels']
    weights = result['weights']
    # weights = np.exp(result['lnprobability'])
    # Maximum A Posteriori (MAP)
    max_apost = best_sample(result)

    imax = np.argmax(result['lnprobability'])

    # get n-dimensional mode
    # bins = 100
    # counts, b = np.histogramdd(samples.T, weights=weights, bins=bins)
    # theta_idx = np.unravel_index(np.argmax(counts), counts.shape)
    
    fit_result_dict = {}
    sfh_dict = {}

    fit_result_dict['HSCobjid'] = result['obs']['obj_id']
    fit_result_dict['clump_id'] = result['obs']['clump_id']
    fit_result_dict['peak_id'] = result['obs']['peak_id']
    fit_result_dict['redshift'] = result['obs']['redshift']
    fit_result_dict['redshift_err'] = result['obs']['redshift_err']
    
    for i, sample in enumerate(samples):
        sample = sample.flatten()
    
        # add param to SFH-dict for SFR calculation
        sfh_dict[labels[i]] = max_apost[i]
    
        # building final result dict
        # median and quartiles around median
        qm, q_minus, q_plus = get_median(sample, alpha, weights)

        # credible interval around MAP
        q_minus_map, q_plus_map = get_map(max_apost[i], sample, alpha, weights)

        # mode and credible interval around mode
        bins = int(round(10. / smooth))
        qm_mode, q_minus_mode, q_plus_mode = get_mode(sample, alpha, weights, bins)

        # n-dimensional mode and credible interval around mode
        # ndim_mode_bin = (b[i][theta_idx[i]] + b[i][theta_idx[i]+1])/2
        # ndim_mode = sample[np.argmin(np.abs(sample - ndim_mode_bin))]
        # q_minus_ndim_mode, q_plus_ndim_mode = get_map(ndim_mode, sample, alpha, weights)

        # add values to dict
        fit_result_dict[labels[i]+'_MAP'] = max_apost[i]
        fit_result_dict[labels[i]+'_cr_lower'] = q_minus_map
        fit_result_dict[labels[i]+'_cr_upper'] = q_plus_map
        fit_result_dict[labels[i]+'_median'] = qm
        fit_result_dict[labels[i]+'_se_lower'] = q_minus
        fit_result_dict[labels[i]+'_se_upper'] = q_plus
        fit_result_dict[labels[i]+'_mode'] = qm_mode
        fit_result_dict[labels[i]+'_md_lower'] = q_minus_mode
        fit_result_dict[labels[i]+'_md_upper'] = q_plus_mode
        # fit_result_dict[labels[i]+'_ndim_mode'] = ndim_mode
        # fit_result_dict[labels[i]+'_ndim_md_lower'] = q_minus_ndim_mode
        # fit_result_dict[labels[i]+'_ndim_md_upper'] = q_plus_ndim_mode 
    
        if labels[i] in logify:
            log_sample = sample.copy()
            log_sample = np.log10(log_sample)

            log_max_apost = np.log10(max_apost[i])
            # log_ndim_mode = np.log10(ndim_mode)

            qm, q_minus, q_plus = get_median(log_sample, alpha, weights) # median and quartiles around median
            q_minus_map, q_plus_map = get_map(log_max_apost, log_sample, alpha, weights) # credible interval around MAP
            qm_mode, q_minus_mode, q_plus_mode = get_mode(log_sample, alpha, weights, bins) # mode and credible interval around mode
            # q_minus_ndim_mode, q_plus_ndim_mode = get_map(log_ndim_mode, log_sample, alpha, weights) # n-dimensional mode and credible interval around mode
            
            # add values to dict
            fit_result_dict['l'+labels[i]+'_MAP'] = log_max_apost
            fit_result_dict['l'+labels[i]+'_cr_lower'] = q_minus_map
            fit_result_dict['l'+labels[i]+'_cr_upper'] = q_plus_map
            fit_result_dict['l'+labels[i]+'_median'] = qm
            fit_result_dict['l'+labels[i]+'_se_lower'] = q_minus
            fit_result_dict['l'+labels[i]+'_se_upper'] = q_plus
            fit_result_dict['l'+labels[i]+'_mode'] = qm_mode
            fit_result_dict['l'+labels[i]+'_md_lower'] = q_minus_mode
            fit_result_dict['l'+labels[i]+'_md_upper'] = q_plus_mode
            # fit_result_dict['l'+labels[i]+'_ndim_mode'] = ndim_mode
            # fit_result_dict['l'+labels[i]+'_ndim_md_lower'] = q_minus_ndim_mode
            # fit_result_dict['l'+labels[i]+'_ndim_md_upper'] = q_plus_ndim_mode           
    
    # add SFR and sSFR to results
    # For the nonparametric SFH the 'current' SFR is just the SFR in the most recent age bin. 
    # For the parametric SFH, parametric_sfr should work, but you need to give the requested lookback time 
    # for the SFR estimation as the 'times' keyword (sorry, I should have set this to default to 0). 
    # Also you can generally just pass it the entire pbest_dict dictionary, especially to get properly mass-normalized SFRs. 
    fit_result_dict['SFR_MAP'] = parametric_sfr(times=0, tavg=1e-3, **sfh_dict)[0]
    if 'mass' in logify:
        fit_result_dict['sSFR_MAP'] = np.log10(fit_result_dict['SFR_MAP']) - fit_result_dict['lmass_MAP']
    try:
        float(result['bestfit']['mfrac'])
        fit_result_dict['mfrac_MAP'] = result['bestfit']['mfrac']
    except (ValueError, TypeError):
        fit_result_dict['mfrac_MAP'] = np.nan
    
    # Alternative way of calculating the SFR
    # https://github.com/bd-j/prospector/issues/166#issuecomment-607966209
    # from scipy.special import gamma, gammainc
    # mass = fit_result_dict['mass_MAP']
    # tage = fit_result_dict['tage_MAP']
    # tau = fit_result_dict['tau_MAP']
    # sfr = mass * (tage/tau**2) * np.exp(-tage/tau) / (gamma(2) * gammainc(2, tage/tau)) * 1e-9
    
    for i, band in enumerate(bands):
        fit_result_dict['photometry_MAP_'+band] = result['bestfit']['photometry'][i]
        fit_result_dict['maggies_'+band] = result['obs']['maggies'][i]
        fit_result_dict['maggies_unc_'+band] = result['obs']['maggies_unc'][i]

    # fit_result_dict['chi'] = [(result['obs']['maggies'] - result['bestfit']['photometry']) / result['obs']['maggies_unc']]
    fit_result_dict['chi2'] = np.sum((result['obs']['maggies'] - result['bestfit']['photometry'])**2 / result['obs']['maggies_unc']**2)
    fit_result_dict['chi2_prob'] = chi2.sf(fit_result_dict['chi2'], result['obs']['ndof'])

    # log-likelihood
    fit_result_dict['lnlikelihood_MAP'] = result['lnlikelihood'][imax]

    # evidence
    fit_result_dict['logz'] = result['logz'][-1]
    fit_result_dict['logz_err'] = result['logzerr'][-1]

    # posterior sample size
    fit_result_dict['sample_size'] = result['niter'][0]

    # effective sample size (EES)
    fit_result_dict['ess'] = (np.sum(result['weights']))**2 / np.sum(result['weights']**2) #dyfunc.get_neff_from_logwt(weights)

    return fit_result_dict


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
    model_params['mass'] = {'N': 1, 'isfree': True, 'init': 1e6, 'prior': priors.LogUniform(mini=1e2, maxi=1e11)}

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
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    
    return sps


def build_obs(obj_id, clump_id, peak_id, filter_set, mag_list, mag_err_list, redshift, redshift_err):
    """
      Creates a dictionary with parameters for a single observation.

      Args:
        obj_id : object-ID for the host galaxy, here HSCobjid
        clump_id : ID of the clump
        peak_id : ID of the detected peak of a clum
        filter_set : list of filter-names used by FSPS, prospector and sedpy
        mag_list : list of magnitudes in the same order as filter_set
        mag_err_list : list of magnitude errors in the same order as filter_set
        redshift: redshift of the host galaxy
        redshift_err : redshift error of the host galaxy

      Returns:
        obs : dictionary with observation object parameters
        file_name : filename with obj_id, clump_id and peak_id for saving fitting results
    """

    filters = load_filters(filter_set)

    # convert the magnitudes to maggies, 
    maggies = np.array([10**(-0.4 * mag) for mag in mag_list])
    # convert the magnitude errors to flux uncertainties (including a noise floor)
    maggies_err = np.array([mag_err for mag_err in mag_err_list])
    maggies_err = np.hypot(maggies_err, 0.05)
    maggies_err = np.clip(maggies_err, 0.05, np.inf)

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
        maggies=maggies, 
        maggies_unc=0.1 * maggies, # maggies_err * maggies / 1.086, # 1.086 = 1/(0.4*np.log(10))
        phot_mask=np.isfinite(np.squeeze(maggies)),
        filters=filters,
        phot_wave=phot_wave
    )
    
    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    # construct filename for results after fitting
    file_name = 'sed_fit_{}_{}_{}.h5'.format(obj_id, clump_id, peak_id)

    return obs, file_name


def build_mock_obs(obj_id, redshift, filter_set, maggies):

    filters = load_filters(filter_set)

    # array of effective wavelengths for each of the filters, useful for plotting
    phot_wave = np.array([f.wave_effective for f in filters])

    obs = dict(
        obj_id=obj_id,
        clump_id=1,
        peak_id=1,
        wavelength=None, 
        spectrum=None, 
        unc=None, 
        maggies=maggies, 
        maggies_unc=maggies*0.1,
        redshift=redshift,
        redshift_err=0.0,
        phot_mask=np.isfinite(np.squeeze(maggies)),
        filters=filters,
        phot_wave=phot_wave
    )
    
    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs