import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
import emcee
import multiprocessing
from scipy.stats import norm, t, lognorm, loguniform, truncnorm
from functools import partial
# import custom_prospector_tools as cpt
from types import SimpleNamespace
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from pathlib import Path
import corner
from IPython.display import display, Math
# from IPython import get_ipython
import sedpy
import argparse
import gc
import time
from datetime import datetime
import h5py
from numba import njit

# ===========================================
# priors and logpdfs
# ===========================================


def logprior_funcs(config):

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
    if type(percentiles) is int or type(percentiles) is float:
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


def display_fits(theta_percentiles, ylabels=None):
    """
    Display function for median parameters and uncertainty from MCMC

    Parameters
    ----------
    theta_percentiles : ndarray of shape (3, nparameters)
        [16, 50, 84] percentiles of each parameter
    ylabels : list of str
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

    if ylabels is None:
        ylabels = [f'theta{i}' for i in range(ndim)]

    for i in range(ndim):
        if ipython_shell:
            ylabeli = ylabels[i].replace('_', r'\_')
            txt = fr"{ylabeli}={median_theta[i]:.3f}^{{+{sigma_theta[1][i]:.3f}}}_{{-{sigma_theta[0][i]:.3f}}}"
            display(Math(txt))
        else:
            print(f"{ylabels[i]}=\t{median_theta[i]:.3f}\t+{sigma_theta[1][i]:.3f}\t-{sigma_theta[0][i]:.3f}")


# ===========================================
# Main emulator_mcmc class
# ===========================================

class emulator_mcmc:
    """

    """

    # ---------------------------------------------------------
    # Built-in defaults
    # ---------------------------------------------------------

    # default config is for user building their own mcmc routine without a config file, 
    # so there are no catalog, filter, emulator in the defaults (user has to specify it)
    default_configs = {
        "MCMC_settings": {
            "nwalkers": 32,
            "jitter": 1.0e-4,
            "nsteps": 20000,
            "discard": 5000,
            "thin": 20,
            "zprior": True,
            "parallel": False,
            "verbose": True
            },

        "Outputs": {
            "save_sampler": False,
            "sampler_filename": "use_id",
            "output_dir": "mcmc_outputs",
            "output_filename": "use_id",
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
           "dust_index": 
            {
                "init": -1.0,
                "bounds": [-1.2, 0.4],
                "prior": {
                    "dist": "uniform",
                }
            },
           "duste_qpah": 
            {
                "init": 2.0,
                "bounds": [0.0, 7.0],
                    "dist": "truncnorm",
                    "loc": 2.0,
                    "scale": 2.0
                }
            },
            # TODO all other parameters' default

        }

    


