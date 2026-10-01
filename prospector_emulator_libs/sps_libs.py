
import numpy as np
from astropy.cosmology import Planck18


# TEMP
cosmology = Planck18

def continuity_sfh_agebins_sfrs(zred, logsfr_ratios, logmass):
    """
    Calculate agebins, massbins, and SFRs from input parameters
    using input redshift, logsfr_ratios, logmass and tuniv from astropy.cosmology

    Parameters
    ----------
    zred : float
        input redshift
    logsfr_ratios : list or np.array
        continuity_sfh sfr parameters
    logmass : float
        input logmass

    Returns
    -------
    agebins : np.array of shape (nbins, 2)
        start and finish of each bin (unit in yrs)
    massbins : np.array of shape (nbins,)
        total formed mass in each bin
    sfrs : np.array of shape (nbins,)
        star forming rate in each bin
    """
    nbins = len(logsfr_ratios)+1
    tuniv = cosmology.age(zred).value
    tbinmax = 0.85 * tuniv * 1e9
    lim1, lim2 = 7.4772, 8.0
    agelims = ([0, lim1] +
               np.linspace(lim2, np.log10(tbinmax), nbins-2).tolist() +
               [np.log10(tuniv*1e9)])
    agebins = 10**(np.array([agelims[:-1], agelims[1:]]))
    agebins = agebins.T

    mass = 10**logmass
    sratios = 10**np.clip(logsfr_ratios, -10, 10)
    dt = agebins[:, 1] - agebins[:, 0]
    coeffs = np.array([ (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
                        for i in range(nbins)])
    m1 = mass / coeffs.sum()
    massbins = m1 * coeffs
    sfrs = massbins / dt
    return agebins, massbins, sfrs

def delayed_tau_trunc(tl, tage, tau, ttrunc, sf_slope, scale):
    """
    Returns Star forming rate of prospector parametric SFH model
    Units of tl, tage, tau has to match (yr or Gyr)
    
    Parameters
    ----------
    tl : ndarray
        lookback time to evaluate SFR. 
        tl=0 is galaxy at its redshift; if tl>tage then SFR returns 0
    tage : float
        Total age of the galaxy
    tau : float
        e-folding time of the SFH
    ttrunc : float
        Truncation time of the SFH, counting from the start of galaxy
    sf_slope : float
        Truncation slope after ttrunc
    scale : float
        Overall scaling of the total SFR
    
    Returns
    -------
    sfr : ndarray
        Total SFR w.r.t input lookback time grid
    """
    tl_trunc = tage - ttrunc
    sfr1 = scale * (tage-tl) * np.exp(-(tage-tl)/tau) * np.where((tl>=tl_trunc) & (tl<=tage), 1, 0)
    sfr2 = scale * np.clip(ttrunc*np.exp(-ttrunc/tau) + sf_slope*(tl_trunc-tl), 0, None) * np.where((tl>=0) & (tl<=tl_trunc), 1, 0)
    sfr = sfr1 + sfr2
    return sfr


def parametric_sfrs(zred, 
                    tage_tuniv, 
                    logmass, 
                    tau, 
                    fburst, 
                    fage_burst, 
                    fage_trunc, 
                    sf_slope,
                    tl=None):
    """
    High level parametric SFR calculation, returns SFR with given prospector inputs, and relevant time in Gyrs
    
    Parameters
    ----------
    zred : float
        Galaxy redshift
    tage_tuniv : float
        Percentage of galaxy age with respect to tuniv at its redshift
    logmass : float
        Log10(M/Msol)
    tau : float
        e-folding time of its SFR, in unit of [Gyr]
    fburst : float
        Fraciton of mass that are from late burst
    fage_burst : float
        Percentage of galaxy burst time with respect to its tage
    fage_trunc : float
        Percentage of galaxy SFR truncation time w.r.t its tage
    sf_slope : float
        SFR truncation slope after fage_trunc
    tl : ndarray
        lookback time to evaluate SFR in unit [yrs]
        if not provided, automatically generated a 200 pt log-spaced grid
        tl=0 is galaxy at its redshift; if tl>tage then SFR returns 0
    
    Returns
    -------
    tage : float
        Total age calculated from astropy.Planck18.age, in [Gyr]
    tburst : float
        Burst time from when galaxy was borned, in [Gyr]
    ttrunc : float
        SFR truncation time from when galaxy was borned, in [Gyr]
    scale : float
        Scaling coefficient to make sure SFR returns correct formed mass
    tl : ndarray
        Returns lookback time grid that are used to evaluated SFR, in [Gyr]
    sfr : ndarray
        Star formation rate at each tl grid point
    """
    tuniv = cosmology.age(zred).value * 1e9
    tage = tuniv * tage_tuniv
    tburst = tage * fage_burst
    tau = tau*1e9
    mass = 10**logmass
    mass_noburst = mass * (1-fburst)
    ttrunc = tage * fage_trunc
    tage_after_trunc = tage - ttrunc
    area_delayed_tau = tau**2 * (1-(1+(ttrunc/tau))*np.exp(-ttrunc/tau))

    if ttrunc*np.exp(-ttrunc/tau) + sf_slope*tage_after_trunc > 0:
        area_trunc = ttrunc*np.exp(-ttrunc/tau)*tage_after_trunc + sf_slope*0.5*tage_after_trunc**2
    else:
        area_trunc = -0.5 * ttrunc**2/sf_slope * np.exp(-2*ttrunc/tau)
    scale = mass_noburst / (area_delayed_tau+area_trunc)

    if tl is None:
        tl = 10**(np.linspace(7, np.log10(tage), 200))
    sfr = delayed_tau_trunc(tl, tage, tau, ttrunc, sf_slope, scale)
    return tage/1e9, tburst/1e9, ttrunc/1e9, scale, tl/1e9, sfr



def continuity_sfh_percentiles_steps(agelims=None,
                                     sfrs=None,
                                     n_transition=20,
                                     transition_start_idx=3,
                                     percentiles=[16, 50, 84]):
    """
    From a given flat sample of all parameters and zred_idx, logsfr_ratios_idx, logmass_idx,
    calculate percentile SFRs in each bin, and append the last datapoint so that its ready for plt.step()

    Parameters
    ----------
    agelims: np.array of shape (n_samples, n_sfr_bins+1)
    
    sfrs: np.array of shape (nsamples, n_sfr_bins)

    n_transition : int
        How finely to chop during age lims where change in redshift cause agebins to change
        and calculate continuous percentiles within those regions
    transition_start_idx : int
        When does fine chopping starts for agebins, including lookback time 0 (starting agelim)
        First two agebins should always be fixed, hense default=3
    percentiles : list
        Percentiles to evaluate piecewise SFH
    
    Returns
    -------
    all_age_lims : np.array of shape (nsteps, )
        All the age limits, including finely chopped agelims between major bins (unit in yrs)
    qs_agebins_all_sfrs : np.array of shape (nsfh, nsteps)
        Evaluated percentile SFRs between all_age_lims, with last datapoint repeated for plt.step()
    """
    if type(percentiles) is int or type(percentiles) is float:
        percentiles = [percentiles]
    n_percentiles = len(percentiles)
    # nbins = len(logsfr_ratios_idx)+1
    nbins = sfrs.shape[1]

    transition_idx = np.arange(transition_start_idx, nbins)
    # loop over each step in MCMC chain to calculate agebins and SFRs
    all_age_lims = agelims
    all_sfrs = sfrs

    age_lims_tbins = np.zeros((n_transition)*len(transition_idx))

    qs_agebins_sfr = np.zeros((n_percentiles, nbins))
    qs_age_tbins_sfr = np.zeros((n_percentiles, (n_transition-1)*len(transition_idx)))

    # first calculate SFR percentiles in major bins
    qs_agebins_sfr = np.percentile(all_sfrs, percentiles, axis=0)

    # for each transitions, cut them into n_transition and 
    # loop over each subbins
    for i, bound_idxi in enumerate(transition_idx):
        bstart = np.min(all_age_lims[:,bound_idxi])
        bend = np.max(all_age_lims[:,bound_idxi])
        fine_steps = np.linspace(bstart, bend, num=n_transition)
        # save the actual age fine limites in age_lims_tbins
        age_lims_tbins[i*n_transition: (i+1)*n_transition] = fine_steps

        for j in range(n_transition-1):
            b0 = fine_steps[j]
            b1 = fine_steps[j+1]
            f0 = np.clip(all_age_lims[:,bound_idxi]-b0, 0, (b1-b0))
            f1 = np.clip(b1-all_age_lims[:,bound_idxi], 0, (b1-b0))
            # the average sfr for EACH chain within this subbin
            ave_sfrs_inbin = (f0 * all_sfrs[:,bound_idxi-1] + f1 * all_sfrs[:,bound_idxi]) / (f0+f1)
            qs_age_tbins_sfr[:, j+i*(n_transition-1)] = np.percentile(ave_sfrs_inbin, percentiles)

    # stitch age lims together, first few age lims are all the same, as well as the last limit
    age_all_lims = np.hstack([all_age_lims[0][0:np.min(transition_idx)], age_lims_tbins, all_age_lims[0][-1]])
    # create new sfrs array but the first few elements are the same as the original sfrs
    qs_agebins_all_sfrs = qs_agebins_sfr[:, :np.min(transition_idx)]
    # stitch percentile agebins and qs_age_tbins together, everytime n_transition points are added, also
    # stitch the next major bin to it
    for i, bound_idxi in enumerate(transition_idx):
        qs_agebins_all_sfrs = np.hstack([qs_agebins_all_sfrs, qs_age_tbins_sfr[:, i*(n_transition-1): (i+1)*(n_transition-1)], qs_agebins_sfr[:, bound_idxi][:,None]])

    # add the last element to sfr array so that they have the same dimension as agelims
    qs_agebins_all_sfrs = np.hstack([qs_agebins_all_sfrs, qs_agebins_all_sfrs[:,-1][:,None]])

    return age_all_lims, qs_agebins_all_sfrs

def parametric_sfh_percentiles(flat_samples,
                               theta=None,
                               zred_idx=0,
                               logmass_idx=2,
                               tage_tuniv_idx=3,
                               tau_idx=4,
                               fage_trunc_idx=5,
                               sf_slope_idx=6,
                               fburst_idx=7,
                               fage_burst_idx=8,
                               pts=200,
                               percentiles=[16, 50, 84]):
    """
    Returns selected percentiles of parametric SFH and lookback time grid in [Gyr]

    Parameters
    ----------
    flat_samples : np.array of shape (nsamples, ndims)
        Flattened MCMC chain
    theta : prospector_mcmc.theta instance
        theta instance containing all theta metadata, will ignore _idx keywords
    zred_idx : int
        Where redshift is in the flat_sample MCMC parameters
    logmass_idx : int
        Where logmass is in the parameters
    tage_tuniv_idx : int
        Where tage_tuniv is in parameters
    tau_idx : int
        Where tau is in parameters
    fage_trunc_idx : int
        Where fage_trunc is in parameters
    sf_slope_idx : int
        Where sf_slope is in parameters
    fburst_idx : int
        Where fburst is in parameters
    fage_burst_idx : int
        Where fage_burst is in parameters
    pts : int
        Number of time grid points to evaluate SFRs
    percentiles : list or float or int
        Percentiles to evaluate SFRs

    Returns
    -------
    common_tl : ndarray of shape (npts, )
        lookback time grid for all SFRs in [Gyr]
    sfh_percentiles : ndarray of shape (npercentiles, npts)
        SFR at all percentiles, w.r.t. common_tl
    tl_burst_percentiles : ndarray of shape (npercentiles, )
        Percentiles of burst lookback times
    """
    if theta is not None:
        zred_idx = theta.zred_idx
        logmass_idx = theta.logmass_idx
        tage_tuniv_idx = theta.tage_tuniv_idx
        tau_idx = theta.tau_idx
        fage_trunc_idx = theta.fage_trunc_idx
        sf_slope_idx = theta.sf_slope_idx
        fburst_idx = theta.fburst_idx
        fage_burst_idx = theta.fage_burst_idx

    nsamples = flat_samples.shape[0]
    all_sfrs = np.empty((nsamples, pts))
    all_tl_bursts = np.empty((nsamples))
    zreds = flat_samples[:,zred_idx]
    tage_tunivs = flat_samples[:,tage_tuniv_idx]
    logmasses = flat_samples[:, logmass_idx]
    taus = flat_samples[:, tau_idx] # * 1e9
    fbursts = flat_samples[:, fburst_idx]
    fage_bursts = flat_samples[:, fage_burst_idx]
    fage_truncs = flat_samples[:, fage_trunc_idx]
    sf_slopes = flat_samples[:, sf_slope_idx]

    tunivs = cosmology.age(zreds).value * 1e9
    tages = tunivs * tage_tunivs
    max_tage = np.max(tages)
    common_tl = 10**(np.linspace(7, np.log10(max_tage), pts))
    for i in range(nsamples):
        tagei, tbursti, _, _, _, sfri = parametric_sfrs(zreds[i],
                                                        tage_tunivs[i],
                                                        logmasses[i],
                                                        taus[i],
                                                        fbursts[i],
                                                        fage_bursts[i],
                                                        fage_truncs[i],
                                                        sf_slopes[i],
                                                        tl=common_tl)
        tl_burst = tagei - tbursti
        all_sfrs[i] = sfri
        all_tl_bursts[i] = tl_burst
    # loop over each common_tl pts to grab percentiles
    if type(percentiles) is float or type(percentiles) is int:
        percentiles = [percentiles]
    # n_percentiles = len(percentiles)
    sfh_percentiles = np.nanpercentile(all_sfrs, percentiles, axis=0)
    tl_burst_percentiles = np.nanpercentile(all_tl_bursts, percentiles)
    return common_tl/1e9, sfh_percentiles, tl_burst_percentiles


