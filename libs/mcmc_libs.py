# import os
import numpy as np
# import pandas as pd
# import pyarrow.dataset as ds
import yaml
import emcee
import multiprocessing
from scipy.stats import norm, t, lognorm, loguniform, truncnorm
from functools import partial
# import custom_prospector_tools as cpt
# from types import SimpleNamespace
import matplotlib.pyplot as plt
# from astropy.cosmology import Planck18
from pathlib import Path
import corner
from IPython.display import display, Math
# from IPython import get_ipython
# import sedpy
# import argparse
# import gc
import time
from datetime import datetime
import h5py
# from numba import njit
import copy
import libs.emulator_libs as elibs
import libs.data_libs as dlibs
import libs.sps_libs as spslibs
from pprint import pprint

# ---------------------------------------------------------
# Built-in defaults
# ---------------------------------------------------------

# default config is for user building their own mcmc routine without a config file, 
# so there are no catalog, filter, emulator in the defaults (user has to specify it)
default_configs = {
    "Files": {
        'filters': None,
    },

    # "Emulator": None,

    "MCMC": {
        "nwalkers": 32,
        "jitter": 1.0e-4,
        "nsteps": 5000,
        "discard": 2000,
        "thin": 10,
        "zprior": True,
        "parallel": False,
        "n_processes": 4,
        "verbose": True
    },

    "Outputs": {
        "save_sampler": False,
        "sampler_filename": "emulator_mcmc_sampler.h5",
        "output_dir": "mcmc_outputs",
        "output_filename": "emulator_mcmc_results.h5",
        "save_plots": False,
        "plots_dir": "mcmc_outputs",
    },
    # prior_dicts should contain all possible parameter and their default range/init
    # only the ones exist in loaded emulator will be used, and bounds will be overrided if 
    # emulator parameters have smaller bounds
    "prior_dicts": {
        "zred": 
        {
            "init": 0.1,
            "bounds": [0.0, 3.0],
            "prior": {
                "dist": "uniform"
            }
        },
        "logmass": 
        {
            "init": 10.0,
            "bounds": [7.5, 13.5],
            "prior": {
                "dist": "uniform"
            }
        },
        "logzsol": 
        {
            "init": 0.0,
            "bounds": [-2.0, 0.2],
            "prior": {
                "dist": "uniform"
            }
        },
        "logsfr_ratios": 
        {
            "init": 0.0,
            "bounds": [-5.0, 5.0],
            "prior": {
                "dist": "student_t",
                "df": 2,
                "loc": 0.0,
                "scale": 0.3
            }
        },
        # dust attenuation
        "dust2": 
        {
            "init": 0.3,
            "bounds": [0.0, 4.0],
            "prior": {
                "dist": "truncnorm",
                "loc": 0.3,
                "scale": 1.0
            }
        },
        "dust_ratio": 
        {
            "init": 1.0,
            "bounds": [0.0, 2.0],
            "prior": {
                "dist": "truncnorm",
                "loc": 1.0,
                "scale": 1.0
            },
        },
        "dust_index": 
        {
            "init": -1.0,
            "bounds": [-1.2, 0.4],
            "prior": {
                "dist": "uniform",
            }
        },
        # dust emission
        "duste_qpah": 
        {
            "init": 2.0,
            "bounds": [0.0, 7.0],
            "prior": {
                "dist": "truncnorm",
                "loc": 2.0,
                "scale": 2.0
            },
        },
        "log10_duste_gamma": 
        {
            "init": -2.0,
            "bounds": [-4.0, 0.0],
            "prior": {
                "dist": "truncnorm",
                "loc": -2.0,
                "scale": 2.0
            },
        },
        "duste_umin": 
        {
            "init": 1.0,
            "bounds": [0.1, 25.0],
            "prior": {
                "dist": "truncnorm",
                "loc": 1.0,
                "scale": 20.0
            },
        },
        # nebular emission
        "gas_logz": 
        {
            "init": 0.0,
            "bounds": [-2.0, 0.5],
            "prior": {
                "dist": "uniform",
            },
        },
        "gas_logu": 
        {
            "init": -2.0,
            "bounds": [-4.0, -1.0],
            "prior": {
                "dist": "uniform",
            }
        },
        # AGN fraction
        "log10_fagn": 
        {
            "init": -4.0,
            "bounds": [-5.0, np.log10(3.0)],
            "prior": {
                "dist": "uniform",
            }
        },
        "log10_agn_tau": 
        {
            "init": np.log10(10.0),
            "bounds": [np.log10(5.0), np.log10(150.0)],
            "prior": {
                "dist": "uniform",
            }
        },
        }

    }


def apply_overrides(config, overrides):
    for key, value in overrides.items():

        if value is None:
            continue

        if "." not in key:
            config[key] = value
            continue

        parts = key.split(".")
        target = config

        for part in parts[:-1]:
            target = target[part]

        target[parts[-1]] = value


# ===========================================
# priors and logpdfs
# ===========================================


def create_logprior_func(config):

    lo, hi = config["bounds"]
    prior = config["prior"]
    dist = prior["dist"]

    if dist == "uniform":
        return partial(uniform_logpdf, low=lo, high=hi)

    elif dist == "loguniform":
        if lo <= 0:
            raise ValueError("loguniform requires positive lower bound")
        return partial(loguniform.logpdf, a=lo, b=hi)

    elif dist == "loguniform1p":
        if lo <= -1:
            raise ValueError("loguniform1p requires lower bound > -1")
        return partial(loguniform1p_logpdf, a=lo, b=hi)

    elif dist == "truncnorm":
        loc = prior["loc"]
        scale=prior["scale"]
        return partial(norm_logpdf_cutoffs, loc=loc, scale=scale, low=lo, high=hi)

    elif dist == "student_t":
        loc=prior.get("loc", 0.0)
        scale=prior.get("scale", 3.0)
        return partial(t_logpdf_cutoffs, low=lo, high=hi, df=prior["df"], loc=loc, scale=scale)        

    else:
        raise ValueError(f"Unknown prior distribution: {dist}")


def uniform_logpdf(val, low=0.0, high=1.0):
    """
    Return uniform logpdf at val, return 0.0 for low <= val <= high
    """
    if float(low) <= val <= float(high):
        return 0.0
    else:
        return -np.inf
    
def norm_logpdf_cutoffs(val, loc, scale, low, high):
    """
    Return gaussian logpdf at val, for gaussian with mean=loc and std=scale,
    with cutoffs at low and high (inclusive)

    Args:
        val (float): The value at which to evaluate the log-pdf
        loc (float): mean of the gaussian distribution
        scale (float): standard deviation of the gaussian
        low (float): cutoff lower bound
        high (float): cutoff upper bound
    """
    if loc is None or scale is None:
        return uniform_logpdf(val, low=low, high=high)
    else:
        return norm.logpdf(val, loc=loc, scale=scale) + uniform_logpdf(val, low=low, high=high)
    
def lognorm_logpdf_cutoffs(val, low, high, s=None, scale=None, norm_mean=None, norm_sig=None):
    """
    Return lognormal logpdf at val, for base 10 lognormal distribution 
    that is a gaussian with mean=loc and std=scale in log space.
    The funciton automatically calculate spread and scale of the lognormal function.
    Ctoffs at low and high (inclusive)

    Args:
        val (float): The value at which to evaluate the log-pdf
        low (float): cutoff lower bound
        high (float): cutoff upper bound
        norm_mean (float): Mean of the gaussian in log10(val) space
        norm_scale (float): Standard deviation of the gaussian in log10(val) space
        s (float): Alternatively, lognormal shape parameter, s=np.log(10) * norm_scale
        scale (float): Alternatively, lognormal scale parameter, scale=10**(norm_mean)
    """
    if norm_mean is not None and norm_sig is not None:
        s = norm_sig * np.log(10)
        scale = 10**(norm_mean)
    return lognorm.logpdf(val, s=s, scale=scale) + uniform_logpdf(val, low=low, high=high)

def t_logpdf_cutoffs(val, low=-5, high=5, df=2, loc=0, scale=0.3):
    """
    Scipy.stats.t distribution (Student's t) with cutoffs low/high

    Args:
        val (float): The value at which to evaluate the log-pdf
        low (float): cutoff lower bound
        high (float): cutoff upper bound
        df (int): Degrees of freedom for the Student's t-distribution
        loc (float): mean of the distribution
        scale (float): The scale parameter used to stretch or compress the distribution

    Returns:
        float: The log-probability density value evaluated at `val`.
    """
    return t.logpdf(val, df=df, loc=loc, scale=scale) + uniform_logpdf(val, low=low, high=high)


def loguniform1p_logpdf(val, low, high):
    if val < low or val > high:
        return -np.inf
    if low <= -1:
        raise ValueError(
            "loguniform1p requires low > -1"
        )
    log_range = np.log1p(high) - np.log1p(low)
    return -np.log1p(val) - np.log(log_range)



def get_theta_percentiles(samples, discard=1000, thin=10, percentiles=[16, 50, 84]):
    """
    Returns selected percentiles of thetas from input samples

    Parameters
    ----------
    samples : np.array
        MCMC chains, can be full sample or flattened
    discard : int
        if samples is the full chain, how many steps to discard
    thin : int
        if samples is the full chain, only sample every thin steps after discard
    percentiles : float or int or list
        Which percentile(s) to evaluate theta

    Returns
    -------
    theta_percentiles : np.array
        Evaluated percentile array of all theta
    """
    if isinstance(percentiles, int) or isinstance(percentiles, float):
        percentiles = [percentiles]
    ndim = samples.shape[-1]
    if samples.ndim == 3:
        flat_samples = samples[discard+thin-1::thin, :, :].reshape(-1, ndim)
    else:
        flat_samples = samples

    theta_percentiles = np.percentile(flat_samples, percentiles, axis=0)
    # diff_percentiles = np.diff(percentiles, axis=0)
    return theta_percentiles



# ===========================================
# Plotting functions
# ===========================================


def plot_chain(samples,
               figsize=(10, 8),
               ylabels=None,
               nrowcol=None,
               save=False,
               filename='mcmc_results_chain.png',
               output_dirname='',
               dpi=300,
               **kwargs):
    """
    From a input full MCMC sample, plot chains for each parameter

    Parameters
    ----------
    figsize : tuple
    ylabels : list of str
        Each parameter names
    nrowcol : tuple/list/ndarray
        Custom number of rows and cols
    save : bool
        Whether to save the plot or not
    filename : str
        Output plot file name, if save=True
    output_dirname : str
        Save plot in this directory
    dpi : int
        Saved plot dpi
    **kwargs : 
        Any matplotlib.pyplot.plot() supported arguments
    """
    ndim = samples.shape[2]
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault('linewidth', 0.7)
    kwargs.setdefault('alpha', 0.1)
    kwargs.setdefault('color', 'k')
    if ylabels is None:
        ylabels = [f'theta{i}' for i in range(ndim)]
    if nrowcol is None:
        nrows = int(np.ceil(np.sqrt(ndim))) # calculate the nrow as if the plot is square, take the smallest number that accommodate it
        ncols = int(np.ceil(ndim / nrows))  # use the above nrows, calculate the resulting ncols and round up
    else:
        nrows = nrowcol[0]
        ncols = nrowcol[1]
        if nrows * ncols < ndim:
            raise ValueError(f'Figure with ({nrows},{ncols}) subplots are not enough for {ndim} parameters')

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    for i in range(nrows*ncols):
        row = i % nrows
        col = i // nrows
        ax = axs[row, col]
        if i < ndim:
            ax.plot(samples[:,:,i], **kwargs)
            ax.grid(alpha=0.2)
            ax.set_ylabel(ylabels[i])
        else:
            ax.set_visible(False)
    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)


def plot_corner(flat_samples,
                ylabels=None,
                figsize=(14,14),
                ticklabelsize=5,
                save=False,
                filename='mcmc_results_corner.png',
                output_dirname='', 
                dpi=300,
                **kwargs):
    """
    From a MCMC flat sample, plot corner plots

    Parameters
    ----------
    flat_samples : ndarray
        Flattened MCMC samples from emcee
    ylabels : list of str
        Each parameter names
    figsize : tuple
        Adjust figure size
    ticklabelsize : int
        Adjust each subplots' tick number size
    save : bool
        Whether to save the plot
    filename : str
        If save=True, the filename to save plot to
    output_dirname : str
        Directory to save the plot to
    dpi : int
        Saved plot dpi
    **kwargs : 
        Any corner.corner() supported arguments
    """
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs.setdefault('label_kwargs', {"fontsize": 6.5})
    kwargs.setdefault('contour_kwargs', {"colors": 'k', "linewidths": 0.7, 'alpha': 0.7})
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('show_titles', True)
    kwargs.setdefault('title_kwargs', {"fontsize": 5})
    kwargs.setdefault('quantiles', [0.16, 0.5, 0.84])   # corner.py use quantile to describe percentile?
    # contour keyword, data_kwargs, contourf_kwargs etc

    ndim = flat_samples.shape[1]
    if ylabels is None:
        ylabels = [f'theta{i}' for i in range(ndim)]
    fig_corner = plt.figure(figsize=figsize)
    corner.corner( 
        flat_samples, 
        labels=ylabels, 
        fig=fig_corner, 
        **kwargs
    )
    for ax in fig_corner.get_axes():
        ax.tick_params(axis='both', labelsize=ticklabelsize)

    if save:
        plt.savefig(output_dir / filename, dpi=dpi)



def plot_sed_sfh(lamb_obs,
                 spec_obs,
                 err_obs,
                 lamb_model,
                 spec_model,
                 agelims_model=None,
                 sfrsteps_model=None,
                 qs_agelims=None,
                 qs_sfrsteps=None,
                 tl_model=None,
                 sfr_model=None,
                 tl_burst_model=None,
                 qs_tl=None,
                 qs_sfrs=None,
                 qs_tl_burst=None,
                 external_phots=None,
                 figsize=(8,8),
                 sed_obs_kwargs=None,
                 sed_model_kwargs=None,
                 sed_external_phot_kwargs=None,
                 sfh_kwargs=None,
                 sfh_range_kwargs=None,
                 qs_tl_burst_kwargs=None,
                 save=False,
                 wl_min=None,
                 wl_max=None,
                 xscale='log',
                 yscale='log',
                 filename='mcmc_results_sed_sfh.png',
                 output_dirname='', 
                 dpi=300,
                 title_kwargs=None
                 ):
    """
    Plot data points, model and external photometry
    Optionally plot Star Formation History. All ages/lookback times have to have consistent unit (Gyr or yr)

    Parameters
    ----------
    lamb_obs : ndarray
        Observed spectra wavelength points
    spec_obs : ndarray
        Observed spectra
    err_obs : ndarray
        Observed spectra uncertainties
    lamb_model : ndarray
        Prospecter output spectra wavelength grid
    spec_model : ndarray
        Prospector output model spectra
    agelims_model : ndarray of shape (nbins, 2) or (nbins+1, )
        For continuity_sfh
        Prospector output agebins or 1-d age limits including start and end
        (nbins+1) is for plt.step compatible format
    sfrsteps_model : ndarray of shape (nbins, ) or (nbins+1, )
        For continuity_sfh
        If agebins are provided above, use SFR for each bin
        If agelims are provided above, use SFR with last element repeated (for plt.step format)
    qs_agelims : ndarray of shape (nsfr_step, )
        For continuity_sfh
        Common age limits for all SFR percentiles
    qs_sfrsteps : ndarray of shape (2, nsfr_step) or (3, nsfr_step)
        For continuity_sfh
        Percentiles of SFH from continuity_sfh_percentiles_steps(), has to be plt.step format (last element for each SFR has to repeat)
        If shape[0] = 3, treat the array as [16, 50, 84]
    tl_model : ndarray
        For parametric_sfh
        lookback time for model SFR
    sfr_model : ndarray
        For parametric_sfh
        Parametric SFH output SFR at each lookback time
    tl_burst_model : float
        For parametric_sfh
        Burst lookback time
    qs_tl : ndarray of shape (nsfr, )
        For parametric_sfh
        Common lookback time grid for SFH percentiles
    qs_sfrs : ndarray of shape (2, nsfr) or (3, nsfr)
        For parametric_sfh
        SFR percentiles with respect to the qs_tl grid
    external_phots : dict
        Dictionary of external photometry from refcat, from catalog.get_external_phots(SPHERExRefID)
    figsize : tuple
        Adjust figure size
    save : bool
        Whether to save the plot
    filename : str
        If save=True, the filename to save plot to
    output_dirname : str
        Directory to save the plot to
    dpi : int
        Saved plot dpi
    xscale/yscale : str
        can be 'linear' or 'log'
    sed_obs_kwargs : dict
        Arguments for ax.errorbar()
    sed_model_kwargs : dict 
        Arguments for ax.plot()
    sed_external_phot_kwargs : dict 
        Arguments for ax.errorbar()
    sfh_kwargs : dict 
        Arguments for ax.step()
    sfh_range_kwargs : dict 
        Arguments for ax.fill_between()
    qs_tl_burst_kwargs : dict 
        Arguments for ax.axvspan()
    title_kwargs : dict 
        Arguments for ax.set_title()
    """
    # age_factor = 1.0

    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    sed_obs_kwargs = sed_obs_kwargs or {}
    sed_model_kwargs = sed_model_kwargs or {}
    sed_external_phot_kwargs = sed_external_phot_kwargs or {}
    sfh_kwargs = sfh_kwargs or {}
    sfh_range_kwargs = sfh_range_kwargs or {}
    qs_tl_burst_kwargs = qs_tl_burst_kwargs or {}
    external_phots = external_phots or {}

    # set up default plotting styles
    sed_obs_kwargs.setdefault('fmt', 'o')
    sed_obs_kwargs.setdefault('elinewidth', 1)
    sed_obs_kwargs.setdefault('markersize', 3)
    sed_obs_kwargs.setdefault('markerfacecolor', 'none')
    sed_obs_kwargs.setdefault('markeredgecolor', 'tab:blue')
    sed_obs_kwargs.setdefault('color', 'tab:blue')
    sed_obs_kwargs.setdefault('markeredgewidth', 1.5)
    sed_obs_kwargs.setdefault('capsize', 2)
    sed_obs_kwargs.setdefault('alpha', 0.7)

    sed_model_kwargs.setdefault('linewidth', 1)
    sed_model_kwargs.setdefault('color', 'tab:orange')
    sed_model_kwargs.setdefault('alpha', 0.9)

    sed_external_phot_kwargs.setdefault('fmt', 'o')
    sed_external_phot_kwargs.setdefault('elinewidth', 1)
    sed_external_phot_kwargs.setdefault('markersize', 5)
    sed_external_phot_kwargs.setdefault('markerfacecolor', 'none')
    # sed_external_phot_kwargs.setdefault('markerfacecolor', 'o')
    sed_external_phot_kwargs.setdefault('markeredgewidth', 1.5)
    sed_external_phot_kwargs.setdefault('capsize', 2)
    sed_external_phot_kwargs.setdefault('alpha', 0.5)

    sfh_kwargs.setdefault('color', 'tab:orange')
    sfh_kwargs.setdefault('linewidth', 1)
    sfh_kwargs.setdefault('alpha', 1)

    sfh_range_kwargs.setdefault('color', 'tab:blue')
    sfh_range_kwargs.setdefault('step', 'post')
    sfh_range_kwargs.setdefault('alpha', 0.2)

    qs_tl_burst_kwargs.setdefault('color', 'tab:green')
    qs_tl_burst_kwargs.setdefault('alpha', 0.1)

    nrow = 2
    ncol = 1
    if agelims_model is sfrsteps_model is qs_agelims is qs_sfrsteps is tl_model is sfr_model is qs_tl is qs_sfrs is None:
        # if SFHs are not provided just plot SED
        figsize = (8,4)
        nrow = 1
    
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    for i, axi in enumerate(fig.get_axes()):
        if i == 0:
            # plot SED
            nonzeros = err_obs != 50000
            axi.errorbar(lamb_obs[nonzeros],
                            spec_obs[nonzeros],
                            err_obs[nonzeros],
                            **sed_obs_kwargs,
                            label='Observed spectra')

            axi.plot(lamb_model,
                        spec_model,
                        **sed_model_kwargs,
                        label='Medium model')
            axi.grid()
            lbs_mins = [np.min(lamb_obs)]
            lbs_maxs = [np.max(lamb_obs)]
            # if np.max(lamb_model)>7:
            #     axi.set_xscale('log')
            if external_phots:
                # Two dummy plots to cycle the color to tab:green
                axi.plot([],[])
                axi.plot([],[])
                for key, each_phot_dict in external_phots.items():
                    axi.errorbar(each_phot_dict['wavelength'],
                                    each_phot_dict['flux'],
                                    each_phot_dict['flux_error'],
                                    **sed_external_phot_kwargs,
                                    label=key)
                    lbs_mins.append(np.min(each_phot_dict['wavelength']))
                    lbs_maxs.append(np.max(each_phot_dict['wavelength']))

            axi.legend()
            xmin = np.min(lbs_mins) * 0.8
            xmax = np.max(lbs_maxs) * 1.2
            if wl_min is not None:
                xmin = wl_min
            if wl_max is not None:
                xmax = wl_max
            if xmax > 7.5:
                axi.set_xscale('log')

            axi.set_xlim(xmin, xmax)
            if title_kwargs is not None:
                title_strings = []
                spherex_id = title_kwargs.get('spherex_id')
                zspec = title_kwargs.get('zspec')
                zphot = title_kwargs.get('zphot')
                zphot_u68 = title_kwargs.get('zphot_u68')
                zphot_l68 = title_kwargs.get('zphot_l68')
                zmcmc_16 = title_kwargs.get('zmcmc_16')
                zmcmc_med = title_kwargs.get('zmcmc_med')
                zmcmc_84 = title_kwargs.get('zmcmc_84')
                frac102 = title_kwargs.get('frac102')
                fontsize = title_kwargs.get('fontsize')
                if spherex_id is not None:
                    title_strings.append(f"SPHERExRefID={spherex_id}")
                if zspec is not None:
                    title_strings.append(fr"$z_{{spec}}$={zspec:.4f}")
                if zphot is not None:
                    if zphot_u68 is not None and zphot_l68 is not None:
                        title_strings.append(fr"$z_{{phot}}={zphot:.3f}^{{+{(zphot_u68-zphot):.3f}}}_{{-{(zphot-zphot_l68):.3f}}}$")
                    else:
                        title_strings.append(fr"$z_{{phot}}={zphot:.3f}$")
                if zmcmc_med is not None:
                    if zmcmc_84 is not None and zmcmc_16 is not None:
                        title_strings.append(fr"$z_{{mcmc}}={zmcmc_med:.3f}^{{+{(zmcmc_84-zmcmc_med):.3f}}}_{{-{(zmcmc_med-zmcmc_16):.3f}}}$")
                    else:
                        title_strings.append(fr"$z_{{mcmc}}={zmcmc_med:.3f}$")
                if frac102 is not None:
                    title_strings.append(f"frac102={frac102:.3f}")
                if fontsize is None:
                    fontsize = 10
                title = ', '.join(title_strings)
                axi.set_title(title, fontsize=fontsize)

            axi.set_xlabel(r'wavelength [$\mu m$]')
            axi.set_ylabel(r'Flux [$\mu m$]')
        if i == 1:
            # if i loop to 1, plot SFH
            if agelims_model is not None: # if continuity sfh is provided
                # check if agelims_model is agebins from prospector or already converted to step required agelims
                if agelims_model.ndim == 2:
                    agelims_model = np.hstack([agelims_model[:,0], agelims_model[-1,1]])
                    sfrsteps_model = np.hstack([sfrsteps_model, sfrsteps_model[-1]])
                # check first agelims are in Gyr
                # convert to Gyr in plottings
                if agelims_model[0] == 1:
                    agelims_model = agelims_model/1e9
                axi.step(agelims_model, 
                         sfrsteps_model, 
                         where='post', 
                         **sfh_kwargs, 
                         label='median parameter SFH')
                if qs_agelims is not None and qs_sfrsteps is not None:
                    if qs_agelims[0] == 1:
                        qs_agelims = qs_agelims/1e9
                    if qs_sfrsteps.shape[0] == 2:
                        idx_84 = 1
                    elif qs_sfrsteps.shape[0] == 3:
                        idx_84 = 2
                        # if 50 percentile SFH exist, plot it
                        sfh_median_kwargs = sfh_kwargs.copy()
                        sfh_median_kwargs['color'] = 'tab:green'
                        axi.step(qs_agelims,
                                 qs_sfrsteps[1],
                                 where='post',
                                 **sfh_median_kwargs,
                                 label='Median SFH from each agebin')
                    axi.fill_between(qs_agelims,
                                     qs_sfrsteps[0],
                                     qs_sfrsteps[idx_84],
                                     **sfh_range_kwargs,
                                     label='68 credible inverval from each agebin')
                axi.set_xlim(np.max(agelims_model), 0.01)
                    
            # if parametric_sfh keywords are given
            elif tl_model is not None:
                if np.median(tl_model) > 1000:  # if it's a big number assuming its in yr
                    tl_model = tl_model/1e9
                axi.plot(tl_model, sfr_model, **sfh_kwargs, label='median parameter SFH')
                if qs_tl is not None and qs_sfrs is not None:
                    if np.median(qs_tl) > 1000:
                        qs_tl = qs_tl/1e9
                        # qs_sfrs = qs_sfrs/1e9
                    if qs_sfrs.shape[0] == 2:
                        idx_84 = 1
                    elif qs_sfrs.shape[0] == 3:
                        idx_84 = 2
                        # if 50 percentile SFH exist, plot it
                        sfh_median_kwargs = sfh_kwargs.copy()
                        sfh_median_kwargs['color'] = 'tab:green'
                        axi.plot(qs_tl, qs_sfrs[1], **sfh_median_kwargs, label=r'Mediah SFH from each $t_{lookback}$')
                    axi.fill_between(qs_tl, qs_sfrs[0], qs_sfrs[2], **sfh_range_kwargs, label=r'68% credible inverval from each $t_{lookback}$')
                    axi.set_xlim(np.max(qs_tl), 0.01)
                ymin, ymax = axi.get_ylim()
                if tl_burst_model is not None:
                    if tl_burst_model > 1000:
                        tl_burst_model = tl_burst_model/1e9
                    axi.vlines(tl_burst_model, ymin=ymin, ymax=ymax, **sfh_kwargs, linestyle='--', label=r'median parameter $tburst_{lookback}$')
                if qs_tl_burst is not None:
                    if np.median(qs_tl_burst) > 1000:
                        qs_tl_burst = qs_tl_burst/1e9
                    if qs_tl_burst.shape[0] == 2:
                        idx_84 = 1
                    elif qs_tl_burst.shape[0] == 3:
                        idx_84 = 2
                        axi.vlines(qs_tl_burst[1], ymin=ymin, ymax=ymax, **sfh_median_kwargs, linestyle='--', label=r'Median $tburst_{lookback}$')
                    axi.axvspan(qs_tl_burst[0], qs_tl_burst[idx_84], **qs_tl_burst_kwargs, label=r'68% cerdible tburst')

            axi.grid(alpha=0.22)
            axi.set_xscale(xscale)
            axi.set_yscale(yscale)
            axi.set_title('Star Formation History')
            axi.set_xlabel('lookback time [Gyr]')
            axi.set_ylabel(r'SFR [$M_{Sol}/yr$]')
            axi.legend()
    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)


def display_fits(theta_percentiles, keys=None):
    """
    Display function for median parameters and uncertainty from MCMC

    Parameters
    ----------
    theta_percentiles : ndarray of shape (3, nparameters)
        [16, 50, 84] percentiles of each parameter
    keys : list of str
        All parameter names
    """
    ndim = theta_percentiles.shape[1]
    median_theta = theta_percentiles[1]
    sigma_theta = np.diff(theta_percentiles, axis=0)

    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':   # type: ignore
            ipython_shell = True
    except:
        ipython_shell = False

    if keys is None:
        keys = [f'theta{i}' for i in range(ndim)]

    for i in range(ndim):
        if ipython_shell:
            ylabeli = keys[i].replace('_', r'\_')
            txt = fr"{ylabeli}={median_theta[i]:.3f}^{{+{sigma_theta[1][i]:.3f}}}_{{-{sigma_theta[0][i]:.3f}}}"
            display(Math(txt))
        else:
            print(f"{keys[i]}=\t{median_theta[i]:.3f}\t+{sigma_theta[1][i]:.3f}\t-{sigma_theta[0][i]:.3f}")


# ===========================================
# Main emulator_mcmc class
# ===========================================

class emulator_mcmc:
    """

    """
    def __init__(
            self,
            config_filename=None,
            emulator=None,
            nwalkers=None,
            jitter=None,
            nsteps=None,
            discard=None,
            thin=None,
            zprior=None,
            filters=None,
            parallel=None,
            n_processes=None,
            verbose=None,
            output_dir=None,
            save_sampler=None,
            sampler_filename=None,
            output_filename=None,
            save_plots=None,
            plots_dir=None,
            ): 

        # initialize emulator and filters to None
        self.emulator = None
        self.filters = None

        # copy a version of default_configs
        self.config = copy.deepcopy(default_configs)
        # if config file is provided, override the defaults
        if config_filename is not None:
            with open(config_filename, "r") as file:
                yaml_config = yaml.safe_load(file)
            self.config.update(yaml_config)

        apply_overrides(
            self.config,
            {
                "MCMC.nwalkers": nwalkers,
                "MCMC.jitter": jitter,
                "MCMC.nsteps": nsteps,
                "MCMC.discard": discard,
                "MCMC.thin": thin,
                "MCMC.zprior": zprior,
                "MCMC.parallel": parallel,
                "MCMC.n_processes": n_processes,
                "MCMC.verbose": verbose,
                "Files.filters": filters,
                "Outputs.output_dir": output_dir,
                "Outputs.save_sampler": save_sampler,
                "Outputs.sampler_filename": sampler_filename,
                "Outputs.output_filename": output_filename,
                "Outputs.save_plots": save_plots,
                "Outputs.plots_dir": plots_dir
            }
        )

        # set up attributes
        self.nwalkers = self.config["MCMC"]["nwalkers"]
        self.jitter = self.config["MCMC"]["jitter"]
        self.nsteps = self.config["MCMC"]["nsteps"]
        self.discard = self.config["MCMC"]["discard"]
        self.thin = self.config["MCMC"]["thin"]
        self.zprior = self.config["MCMC"]["zprior"]
        self.parallel = self.config["MCMC"]["parallel"]
        self.n_processes = self.config["MCMC"]["n_processes"]
        self.verbose = self.config["MCMC"]["verbose"]
        self.output_dir = self.config["Outputs"]["output_dir"]
        self.save_sampler = self.config["Outputs"]["save_sampler"]
        self.sampler_filename = self.config["Outputs"]["sampler_filename"]
        self.output_filename = self.config["Outputs"]["output_filename"]
        self.plots_dir = self.config["Outputs"]["plots_dir"]
        self.save_plots = self.config["Outputs"]["save_plots"]

        self.filters = self._resolve_filters(self.config["Files"]["filters"])
        self.emulator = self._resolve_emulator(emulator)
        self.prior_dicts = None
        self.logprior_funcs = None
        # self._mcmc_setup_signature = None

        self.user_prior = copy.deepcopy(self.config["prior_dicts"])

        # create self.prior_dicts and self.logprior_funcs from emulator and config
        if self.emulator is not None:
            self.setup_mcmc()

    def _resolve_emulator(self, emulator=None):
        # Explicitly supplied
        if emulator is not None:
            if isinstance(emulator, elibs.LoadedSPSEmulator):
                return emulator

            # Otherwise assume it is a filename/path
            return elibs.LoadedSPSEmulator(emulator)

        # Try config
        emulator_path = self.config.get("emulator")

        if emulator_path is not None:
            return elibs.LoadedSPSEmulator(emulator_path)

        # Allow incomplete construction
        return None

    @staticmethod
    def _resolve_filters(filters):
        if filters is False:
            return None

        if isinstance(filters, (str, Path)):
            return dlibs.read_filters(filters)

        return filters

    def check_mcmc_ready(self):
        missing = []
        if self.emulator is None:
            missing.append("emulator")
        # if self.filters is None:
            # missing.append("filters")
        if missing:
            raise RuntimeError("MCMC is not ready. Missing required attributes: " + ", ".join(missing))
        return True

    @staticmethod
    def build_mcmc_prior_dicts(mcmc_prior_dicts, emulator_prior_dicts):
        """
        Reconcile MCMC priors with emulator training bounds.
        The emulator determines the maximum allowed parameter support.
        The MCMC prior distribution/settings are retained.

        Parameters
        ----------
        mcmc_prior_dicts : dict
            User-defined MCMC priors.
        emulator_prior_dicts : dict
            Priors/bounds stored with the trained emulator.
        Returns
        -------
        prior_dicts : dict
            Only parameters present in emulator_prior_dict, with bounds
            clipped to the emulator training range.
        """

        prior_dicts = {}
        fixed_dicts = {}

        for key, train_config in emulator_prior_dicts.items():

            if key not in mcmc_prior_dicts:
                raise KeyError(
                    f"Emulator has prior information for '{key}', "
                    f"but MCMC prior_dicts does not define it."
                )

            mcmc_config = copy.deepcopy(mcmc_prior_dicts[key])
            # -----------------------------------------------------
            # Fixed parameter
            # -----------------------------------------------------
            if mcmc_config.get("fixed") is not None:

                fixed_value = mcmc_config["fixed"]

                # TODO this assumes user input dimension is correct, should check it in the future
                if key == "logsfr_ratios":
                    for i, value in enumerate(fixed_value):
                        fixed_dicts[f"logsfr_ratios{i}"] = value
                else:
                    fixed_dicts[key] = fixed_value

                continue

            # -----------------------------------------------------
            # Free parameter
            # -----------------------------------------------------

            train_low, train_high = train_config["bounds"]
            mcmc_low, mcmc_high = mcmc_config["bounds"]

            # MCMC prior cannot extend outside emulator validity.
            effective_low = max(mcmc_low, train_low)
            effective_high = min(mcmc_high, train_high)

            if effective_low >= effective_high:
                raise ValueError(
                    f"No valid overlap for '{key}': "
                    f"MCMC bounds={mcmc_config['bounds']}, "
                    f"emulator bounds={train_config['bounds']}"
                )

            mcmc_config["bounds"] = [effective_low, effective_high]

            prior_dicts[key] = mcmc_config

        return prior_dicts, fixed_dicts

    @staticmethod
    def build_logprior_funcs(prior_dicts, train_param_keys):
        """
        Build one log-prior function for each emulator parameter.
        The returned list follows exactly the ordering of train_param_keys.

        logsfr_ratios0, logsfr_ratios1, ... all use the prior defined
        under 'logsfr_ratios'.
        """
        logprior_funcs = []

        for key in train_param_keys:
            # logsfr_ratios0, logsfr_ratios1, ...
            if key.startswith("logsfr_ratios"):
                prior_key = "logsfr_ratios"
            else:
                prior_key = key
            if prior_key not in prior_dicts:
                raise KeyError(f"No MCMC prior defined for '{key}'. "f"Expected prior entry '{prior_key}'.")
            config = prior_dicts[prior_key]
            logprior_funcs.append(create_logprior_func(config))

        return logprior_funcs

    def build_initial(self, prior_dicts, train_param_keys):
        initial = []

        for key in train_param_keys:
            prior_key = (
                "logsfr_ratios"
                if key.startswith("logsfr_ratios")
                else key
            )

            if prior_key not in prior_dicts:
                raise KeyError(
                    f"No prior definition for '{key}'. "
                    f"Expected '{prior_key}'."
                )

            initial.append(prior_dicts[prior_key]["init"])

        return np.asarray(initial, dtype=float)


    def setup_mcmc(self, prior=None):
        """Build/rebuild all MCMC state that depends on the emulator."""

        if self.emulator is None:
            raise RuntimeError(
                "Cannot set up MCMC without an emulator."
            )
        
        if prior is None:
            prior_input = self.config["prior_dicts"]
        else:
            prior_input = prior

        # ---------------------------------------------------------
        # Build free-prior and fixed-parameter dictionaries
        # ---------------------------------------------------------
        self.prior_dicts, self.fixed_dicts = self.build_mcmc_prior_dicts(
            prior_input, 
            self.emulator.prior_dicts
            )
        # ---------------------------------------------------------
        # Full emulator parameter ordering
        # ---------------------------------------------------------
        self.train_param_keys = self.emulator.train_param_keys
        self.fixed_param_keys = [
            key
            for key in self.train_param_keys
            if key in self.fixed_dicts
        ]
        # ---------------------------------------------------------
        # Fixed/free parameters
        # ---------------------------------------------------------
        self.free_param_keys = [
            key
            for key in self.train_param_keys
            if key not in self.fixed_dicts
        ]

        self.fixed_param_vals = np.array(
            [
                self.fixed_dicts[key]
                for key in self.fixed_param_keys
            ],
            dtype=float,
        )
        # ---------------------------------------------------------
        # Dimensions
        # ---------------------------------------------------------
        self.ndim_train = len(self.train_param_keys)
        self.ndim_mcmc = len(self.free_param_keys)

        # ---------------------------------------------------------
        # Indices in full emulator x
        # ---------------------------------------------------------            
        self.index_fixed_in_train = np.array(
            [
                i
                for i, key in enumerate(self.train_param_keys)
                if key in self.fixed_dicts
            ],
            dtype=int,
        )

        self.index_free_in_train = np.array(
            [
                i
                for i, key in enumerate(self.train_param_keys)
                if key not in self.fixed_dicts
            ],
            dtype=int,
        )

        # ---------------------------------------------------------
        # Useful parameter indices
        # ---------------------------------------------------------

        self.train_index = {
            key: i
            for i, key in enumerate(self.train_param_keys)
        }

        self.mcmc_index = {
            key: i
            for i, key in enumerate(self.free_param_keys)
        }

        self.zred_train_index = self.train_index.get("zred")
        self.logmass_train_index = self.train_index.get("logmass")
        self.zred_mcmc_index = self.mcmc_index.get("zred")
        self.logmass_mcmc_index = self.mcmc_index.get("logmass")

        self.logsfr_ratios_train_index = np.array(
            [
                self.train_index[key]
                for key in self.train_param_keys
                if key.startswith("logsfr_ratios")
            ],
            dtype=int,
        )

        self.logsfr_ratios_mcmc_index = np.array(
            [
                self.mcmc_index[key]
                for key in self.free_param_keys
                if key.startswith("logsfr_ratios")
            ],
            dtype=int,
        )

        # ---------------------------------------------------------
        # Prior functions for free MCMC parameters only
        # ---------------------------------------------------------
        self.logprior_funcs = self.build_logprior_funcs(
            self.prior_dicts,
            self.free_param_keys,
        )

        has_logsfr = len(self.logsfr_ratios_train_index) > 0
        has_tage = "tau" in self.train_param_keys

        if has_logsfr and has_tage:
            raise ValueError(
                "Cannot determine SFH type: emulator contains parameters "
                "for both continuity and parametric SFHs."
            )
        elif has_logsfr:
            self.sfh_type = "continuity_sfh"
        elif has_tage:
            self.sfh_type = "parametric_sfh"
        else:
            self.sfh_type = None

        self.initial = self.build_initial(
            self.prior_dicts,
            self.free_param_keys,
        )

    
    def redshift_prior_funcs(
        self,
        prior_dicts,
        logprior_funcs,
        # initial,
        redshift,
        redshift_err,
    ):
        """
        Return temporary MCMC prior functions and initial values with
        an object-specific truncated-normal redshift prior.
        """
        if redshift_err <= 0:
            raise ValueError("redshift_err must be > 0.")

        # zred_index was already established in setup_mcmc()
        if self.zred_mcmc_index is None:
            raise RuntimeError("Cannot apply redshift prior because zred is not a free MCMC parameter")

        # Copy so the persistent MCMC state is unchanged
        logprior_funcs_new = logprior_funcs.copy()
        # initial_new = initial.copy()

        # Keep the reconciled emulator bounds from prior_dict
        zred_config = copy.deepcopy(prior_dicts["zred"])

        # Object-specific redshift prior
        zred_config["prior"] = {
            "dist": "truncnorm",
            "loc": redshift,
            "scale": redshift_err,
        }

        # Use the generic prior factory
        logprior_funcs_new[self.zred_mcmc_index] = create_logprior_func(zred_config)

        # Object-specific initial value
        # initial_new[self.zred_index] = redshift

        return logprior_funcs_new#, initial_new


    def theta_to_x(self, theta):
        x = np.empty(self.ndim_train, dtype=float)
        x[self.index_fixed_in_train] = self.fixed_param_vals
        x[self.index_free_in_train] = theta
        return x

    def get_param_value(self, key, theta):
        """
        Return the value of a parameter from theta if free,
        otherwise from fixed_dict.
        """
        if key in self.fixed_dicts:
            return self.fixed_dicts[key]

        return theta[self.mcmc_index[key]]

    def get_logsfr_ratios(self, theta):
        """Reconstruct the full logsfr_ratios array from theta/fixed values."""

        logsfr_ratios = np.empty(
            len(self.logsfr_ratios_train_index),
            dtype=float,
        )

        for i, train_idx in enumerate(self.logsfr_ratios_train_index):
            key = self.train_param_keys[train_idx]

            if key in self.fixed_dicts:
                logsfr_ratios[i] = self.fixed_dicts[key]
            else:
                logsfr_ratios[i] = theta[self.mcmc_index[key]]

        return logsfr_ratios



    def log_prior(self, theta, logprior_funcs=None):
        if logprior_funcs is None:
            logprior_funcs = self.logprior_funcs

        logp = 0.0
        for value, logprior_func in zip(theta, logprior_funcs):
            lp = logprior_func(value)
            if not np.isfinite(lp):
                return -np.inf
            logp += lp

        return logp

    def log_probability(self, theta, flux, flux_error=None, flux_lbs=None, filters=None, logprior_funcs=None):
        lp = self.log_prior(theta, logprior_funcs=logprior_funcs)

        if not np.isfinite(lp):
            return -np.inf, 0.0
        x = self.theta_to_x(theta)
        prediction = self.emulator.predict_one(x)

        zred = self.get_param_value("zred", theta)
        lbs_model_shifted = prediction["lbs"] * (1.0+zred)
        flux_model = prediction["flux"]
        mfrac = prediction["mfrac_scaled"]
        if filters is None:
            if flux_lbs is None:
                raise ValueError("lbs and filters cannot both be None!")
            flux_model_conv = np.interp(flux_lbs, lbs_model_shifted, flux_model)
        else:
            flux_model_conv = dlibs.convolve_filter(wl=lbs_model_shifted, flux=flux_model, filters=filters)

        if flux_error is None:
            ll = -0.5 * np.sum((flux_model_conv-flux)**2)
        else:
            ll = -0.5 * np.sum((flux_model_conv-flux)**2 / flux_error**2 + np.log(2*np.pi*flux_error**2))
        log_prob = lp + ll

        return log_prob, mfrac

    def run_mcmc(
            self, 
            flux,
            flux_err=None,
            flux_lbs=None,
            filters=None,
            redshift=None, 
            redshift_err=None,
            initial=None,
            nwalkers=None,
            jitter=None,
            nsteps=None,
            zprior=None,
            discard=None,
            thin=None,
            prior=None,
            save_sampler=None,
            sampler_filename=None,
            verbose=None,
            parallel=None,
            n_processes=None,
            results=True,
            ):

        # if flux_err is None:
            # flux_err = np.ones_like(flux)
        # initial_user = initial is not None
        if filters is None:
            filters = self.filters
        else:
            filters = self._resolve_filters(filters)

        nwalkers = self.nwalkers if nwalkers is None else nwalkers
        jitter = self.jitter if jitter is None else jitter
        nsteps = self.nsteps if nsteps is None else nsteps
        zprior = self.zprior if zprior is None else zprior
        discard = self.discard if discard is None else discard
        thin = self.thin if thin is None else thin

        save_sampler = self.save_sampler if save_sampler is None else save_sampler
        sampler_filename = self.sampler_filename if sampler_filename is None else sampler_filename
        verbose = self.verbose if verbose is None else verbose
        parallel = self.parallel if parallel is None else parallel
        n_processes = self.n_processes if n_processes is None else n_processes

        self.check_mcmc_ready()

        # If setup has not happened, do it now.
        # if self.prior_dicts is None or self.logprior_funcs is None:
        self.setup_mcmc(prior=prior)
        initial = self.initial.copy() if initial is None else np.asarray(initial).copy()

        # TEMP
        self.flux = flux
        self.flux_err = flux_err

        # self.ndim = len(self.emulator.train_param_keys)
        logprior_funcs = self.logprior_funcs

        if zprior and "zred" in self.mcmc_index:

            if redshift is None and redshift_err is None:
                logprior_funcs = self.logprior_funcs
            elif redshift is not None and redshift_err is not None:
                logprior_funcs = self.redshift_prior_funcs(
                    self.prior_dicts,
                    self.logprior_funcs,
                    redshift,
                    redshift_err,
                )
                # if not initial_user:
                initial[self.zred_mcmc_index] = redshift
            else:
                raise ValueError("redshift and redshift_err must be provided together.")

        # set up initial position with jitters
        initial_pos = initial + jitter * np.random.randn(nwalkers, self.ndim_mcmc)

        if save_sampler:
            backend = emcee.backends.HDFBackend(sampler_filename)
            backend.reset(nwalkers, self.ndim_mcmc)
        else:
            backend = None        

        if parallel:
            with multiprocessing.Pool(processes=n_processes) as pool:
                self.sampler = emcee.EnsembleSampler(
                    nwalkers=nwalkers,
                    ndim=self.ndim_mcmc,
                    log_prob_fn=self.log_probability,
                    args=(flux, flux_err, flux_lbs, filters, logprior_funcs),
                    backend=backend,
                    pool=pool
                )
                self.sampler.run_mcmc(initial_pos, nsteps, progress=verbose)
        else:
                self.sampler = emcee.EnsembleSampler(
                    nwalkers=nwalkers,
                    ndim=self.ndim_mcmc,
                    log_prob_fn=self.log_probability,
                    args=(flux, flux_err, flux_lbs, filters, logprior_funcs),
                    backend=backend,
                )
                self.sampler.run_mcmc(initial_pos, nsteps, progress=verbose)

        self.full_samples = self.sampler.get_chain()
        if results:
            self.results = self.get_results(discard=discard, thin=thin)

    def get_results(
            self, 
            discard=None,
            thin=None):
        discard = self.discard if discard is None else discard
        thin = self.thin if thin is None else thin

        nsteps = self.full_samples.shape[0]
        if discard >= nsteps:
            raise ValueError('discard cannot be equal or larger than nsteps!')

        flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        flat_mfracs = self.sampler.get_blobs(discard=discard, thin=thin, flat=True)
        theta_percentiles = get_theta_percentiles(flat_samples, percentiles=[16, 50, 84])
        mfracs_percentiles = np.percentile(flat_mfracs, [16, 50, 84])
        autocorr = self.sampler.get_autocorr_time(discard=discard, thin=thin, tol=0)

        self.flat_samples = flat_samples

        x_med = self.theta_to_x(theta_percentiles[1])
        prediction_med = self.emulator.predict_one(x_med)

        if "zred" in self.free_param_keys:
            zred_med = theta_percentiles[1][self.mcmc_index["zred"]]
            zred_16 = theta_percentiles[0][self.mcmc_index["zred"]]
            zred_84 = theta_percentiles[2][self.mcmc_index["zred"]]
        else:
            zred_med = self.fixed_dicts["zred"]
            zred_16 = self.fixed_dicts["zred"]
            zred_84 = self.fixed_dicts["zred"]

        if "logmass" in self.free_param_keys:
            logmass_med = theta_percentiles[1][self.mcmc_index["logmass"]]
        else:
            logmass_med = self.fixed_dicts["logmass"]
        logsfr_ratios_med = self.get_logsfr_ratios(theta_percentiles[1])

        lbs_med = prediction_med["lbs"] * (1+zred_med)  # prediction lbs is rest frame
        flux_med = prediction_med["flux"]
        flux_med_conv = dlibs.convolve_filter(lbs_med, flux_med, filters=self.filters)
        mfrac_med = mfracs_percentiles[1]

        # save some med results for easy access
        self.lbs_med = lbs_med
        self.flux_med = flux_med
        self.zred_med = zred_med
        self.zred_16 = zred_16
        self.zred_84 = zred_84
        self.logmass_med = logmass_med
        self.logsfr_ratios_med = logsfr_ratios_med

        if self.sfh_type == 'continuity_sfh':
            agebins_med, massbins_med, med_sfrs_med = spslibs.continuity_sfh_agebins_sfrs(
                zred_med,
                logsfr_ratios_med,
                logmass_med,
            )

            results = {
                'flat_samples': flat_samples,
                'flat_mfracs': flat_mfracs,
                'theta_percentiles': theta_percentiles,
                'lbs_rest': prediction_med["lbs"],
                'lbs_med': lbs_med,
                'flux_med': flux_med,
                'flux_med_conv': flux_med_conv,
                'mfrac_med': mfrac_med,
                'mfracs_percentiles': mfracs_percentiles,
                'agebins_med': agebins_med,
                'massbins_med': massbins_med,
                'sfrs_med': med_sfrs_med,
                'free_param_keys': self.free_param_keys,
                'fixed_param_keys': self.fixed_param_keys,
                'fixed_param_vals': self.fixed_param_vals,
                'autocorr': autocorr
            }

        # TODO parametric_sfh

        return results

    def get_continuity_sfh_all_agelims_sfrs(self, flat_samples=None):
        if flat_samples is None:
            flat_samples = self.flat_samples
        n_samples = len(flat_samples)
        nbins = self.emulator.default_params["nbins"]
        all_age_lims = np.zeros((n_samples, nbins+1))
        all_sfrs = np.zeros((n_samples, nbins))

        for i in range(n_samples):
            theta_i = flat_samples[i]
            zred_i = self.get_param_value("zred", theta_i)
            logmass_i = self.get_param_value("logmass", theta_i)
            logsfr_ratios_i = self.get_logsfr_ratios(theta_i)
            agebins_i, _, sfrs_i = spslibs.continuity_sfh_agebins_sfrs(zred_i, logsfr_ratios_i, logmass_i)
            age_lims_i = np.hstack([agebins_i[:,0], agebins_i[-1,1]])
            all_age_lims[i] = age_lims_i
            all_sfrs[i] = sfrs_i
        return all_age_lims, all_sfrs

    def current_settings(self):
        print("nwalkers:", self.nwalkers)
        print("nsteps:", self.nsteps)
        print("jitter:", self.jitter)
        print("discard:", self.discard)
        print("thin:", self.thin)
        print("zprior:", self.zprior)
        print("save_sampler:", self.save_sampler)
        print("save_plots:", self.save_plots)
        # print("parallel:", self.parallel)
        print("\ncurrent prior:")
        pprint(self.prior_dicts, sort_dicts=False)


    # TODO saving file function
    def save_results(self, 
                     results=None, 
                     output_filename=None,
                     output_dir=None,):
        if results is None:
            results = self.results
        if output_filename is None:
            output_filename = self.output_filename
        if output_dir is None:
            output_dir = self.output_dir

        dlibs.save_h5_results(results, 
                        output_filename=output_filename, 
                        output_dir=output_dir,
                        metadata={
                            'free_param_keys': self.free_param_keys,
                            'fixed_param_keys': self.fixed_param_keys,
                            'fixed_param_vals': self.fixed_param_vals,
                            # 'start_time': start_datetime,
                            # 'end_time': end_datetime,
                            # 'SPHERExRefID': spherex_id,
                            "flux_obs": self.flux,
                            "flux_err_obs": self.flux_err,
                            'nwalkers': self.nwalkers,
                            'jitter': self.jitter,
                            'nsteps': self.nsteps,
                            'discard': self.discard,
                            'thin': self.thin,
                            'zprior': self.zprior,
                            'parallel': self.parallel
                            }
                        )

    