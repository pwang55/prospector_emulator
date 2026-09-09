import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import t, truncnorm
import copy

# TODO add log10_duste_gamma, log10_fagn defaults, log10_agn_tau
# default myparam values
emu_params_continuity_sfh = {'zred': 0.1,
            'logmass': 10,
            'logzsol': 0.0,
            'nbins': 5,
            'logsfr_ratios': [0.0, 0.0, 0.0, 0.0],
            'dust2': 1.0,
            'dust_ratio': 1.0,
            'dust_index': -1.2,
            # 'duste_gamma': 0.01,
            'log10_duste_gamma': -2,
            'duste_umin': 1.0,
            'duste_qpah': 2.0,
            'add_neb_emission': False,
            'add_neb_continuum': False,
            'gas_logz': 0.0,
            'gas_logu': -2.0,
            'add_agn': False,
            # 'fagn': 0.0001,
            'log10_fagn': -4,
            # 'agn_tau': 5.0,
            'log10_agn_tau': np.log10(5.0)
            }

# parametric_sfh version of myparam
emu_params_parametric_sfh = {'zred': 0.1,
            'logmass': 10,
            'logzsol': 0.0,
            'tage_tuniv': 0.9,
            'tau': 2,
            'fage_trunc': 0.95,
            'sf_slope': -0.5,
            'fburst': 0.1,
            'fage_burst': 0.9,
            'dust2': 1.0,
            'dust_ratio': 1.0,
            'dust_index': -1.2,
            # 'duste_gamma': 0.01,
            'log10_duste_gamma': -2,
            'duste_umin': 1.0,
            'duste_qpah': 2.0,
            'add_neb_emission': False,
            'add_neb_continuum': False,
            'gas_logz': 0.0,
            'gas_logu': -2.0,
            'add_agn': False,
            # 'fagn': 0.0001,
            'log10_fagn': -4,
            # 'agn_tau': 5.0,
            'log10_agn_tau': np.log10(5.0)
            }

log10_params_to_convert = ["log10_duste_gamma", "log10_fagn", "log10_agn_tau"]

# after defining train_param_keys used in emulator training, this function
# returns the default_params to keep track of default information
def get_default_params(sfh_type, train_param_keys):    

    if sfh_type == "continuity_sfh":
        myparams = emu_params_continuity_sfh
    elif sfh_type == "parametric_sfh":
        myparams = emu_params_parametric_sfh

    prefix = "logsfr_ratios"

    sfr_indices = [
        int(key[len(prefix):])
        for key in train_param_keys
        if key.startswith(prefix)
    ]

    # TODO if nbins doesn't exist this will break for now
    nbins = max(sfr_indices) + 2 if sfr_indices else myparams["nbins"]

    default_params = {
        key: value
        for key, value in myparams.items()
        if key not in train_param_keys
        and key != "logsfr_ratios"
    }

    if sfh_type == "continuity_sfh":
        default_params["nbins"] = nbins

    # # check if there are log10_params that should be changed to prospector regular version
    # for key, value in default_params.items():
    #     if key in log10_params_to_convert:
    #         new_key = key.replace("log10_", "")
    #         default_params[new_key] = 10**value
    #         default_params.pop(key)

    return default_params

# update myparams used in prospector based on train_param_keys 
# and optionally a new default_params
def update_myparams(myparams, train_param_keys, default_params=None):
    prefix = "logsfr_ratios"

    # Count user-defined logsfr_ratios parameters
    nratios = sum(key.startswith(prefix) for key in train_param_keys)

    # Infer nbins
    nbins = nratios + 1 if nratios > 0 else myparams["nbins"]

    # Make a copy so the original is unchanged
    myparams_new = myparams.copy()

    # Update values from default_params, if provided
    if default_params is not None:
        myparams_new.update(default_params)

    # Always enforce nbins inferred from train_param_keys
    myparams_new["nbins"] = nbins
    myparams_new["logsfr_ratios"] = [0.0] * (nbins - 1)

    return myparams_new

def generate_random_values(prior_dicts, train_param_keys, nsamples, rng=None):
    """
    Generate random parameter values as a 2D array.

    Parameters
    ----------
    prior_dicts : dict
        Prior definitions. For example, "logsfr_ratios" defines
        the prior for logsfr_ratios0, logsfr_ratios1, etc.

    train_param_keys : list[str]
        Defines the exact column ordering of the output array.

    nsamples : int
        Number of samples to generate.

    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    rand_vals : ndarray, shape (nsamples, n_params)
        Random samples. Column i corresponds to train_param_keys[i].
    """
    if rng is None:
        rng = np.random.default_rng()
    n_params = len(train_param_keys)
    rand_vals = np.empty((nsamples, n_params), dtype=float)

    for i, key in enumerate(train_param_keys):
        # logsfr_ratios0, logsfr_ratios1, ...
        # all use the "logsfr_ratios" prior definition
        if key.startswith("logsfr_ratios"):
            prior_key = "logsfr_ratios"
        else:
            prior_key = key

        if prior_key not in prior_dicts:
            raise KeyError(
                f"No prior defined for parameter '{prior_key}' "
                f"(needed by '{key}')"
            )

        rand_vals[:, i] = sample_prior(prior_dicts[prior_key], nsamples, rng)

    return rand_vals



def sample_prior(config, nsample, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    lo, hi = config["bounds"]
    prior = config["prior"]
    dist = prior["dist"]

    if dist == "uniform":
        return rng.uniform(lo, hi, nsample)

    elif dist == "loguniform":
        if lo <= 0:
            raise ValueError("loguniform requires positive lower bound")
        return np.exp(rng.uniform(np.log(lo), np.log(hi), nsample))

    elif dist == "loguniform1p":
        if lo <= -1:
            raise ValueError("loguniform1p requires lower bound > -1")
        zu = rng.uniform(np.log1p(lo), np.log1p(hi), size=nsample)
        return np.expm1(zu)

    elif dist == "truncnorm":
        loc = prior.get("loc", 0.0)
        scale = prior.get("scale", 1.0)
        a = (lo - loc) / scale
        b = (hi - loc) / scale

        return truncnorm.rvs(
            a,
            b,
            loc=loc,
            scale=scale,
            size=nsample,
            random_state=rng,
        )

    elif dist == "student_t":
        rv = t(df=prior["df"],
               loc=prior.get("loc", 0.0),
               scale=prior.get("scale", 3.0))
        return rv.ppf(rng.uniform(rv.cdf(lo), rv.cdf(hi), nsample))

    # elif dist == "lognormal":
    #     if lo <= 0:
    #         raise ValueError("lognormal requires positive lower bound")
    #     log_lo = np.log(lo)
    #     log_hi = np.log(hi)
    #     a = (lo)
    else:
        raise ValueError(f"Unknown prior distribution: {dist}")


def convert_log10_params_to_prospector_params(params):
    # params = copy.deepcopy(params)
    for key in log10_params_to_convert:
        if key in params:
            new_key = key.replace("log10_", "")
            params[new_key] = 10.0 ** params.pop(key)
    return params

def rand_vals_to_all_params(rand_vals, train_param_keys, default_params):
    """
    Convert a 2D array of parameter values into a list of
    Prospector-style parameter dictionaries.

    Parameters
    ----------
    rand_vals : ndarray, shape (nsamples, n_params)
        Parameter values in the same ordering as train_param_keys.

    train_param_keys : list[str]
        Parameter name corresponding to each column of rand_vals.
        logsfr_ratios0, logsfr_ratios1, ... are combined into
        the single `logsfr_ratios` array.

    default_params : dict
        Default/fixed parameters not being sampled.

    Returns
    -------
    all_params : list[dict]
        One myparams-style dictionary per sample.
        Use all_params[i] when calling Prospector.
    """
    rand_vals = np.asarray(rand_vals)

    if rand_vals.ndim != 2:
        raise ValueError(
            f"rand_vals must be 2D, got shape {rand_vals.shape}"
        )

    nsamples, n_params = rand_vals.shape

    if n_params != len(train_param_keys):
        raise ValueError(
            f"rand_vals has {n_params} columns, but "
            f"train_param_keys has {len(train_param_keys)} entries."
        )

    # Find how many logsfr_ratios are being sampled
    sfr_indices = [
        int(key[len("logsfr_ratios"):])
        for key in train_param_keys
        if key.startswith("logsfr_ratios")
    ]

    nratios = max(sfr_indices) + 1 if sfr_indices else 0

    all_params = []

    for i in range(nsamples):

        # Deep copy so every sample is completely independent
        params = copy.deepcopy(default_params)

        # Make the logsfr_ratios array
        if nratios > 0:
            params["logsfr_ratios"] = [0.0] * nratios

        # Fill sampled parameters
        for j, key in enumerate(train_param_keys):
            value = rand_vals[i, j]

            if key.startswith("logsfr_ratios"):
                index = int(key[len("logsfr_ratios"):])
                params["logsfr_ratios"][index] = value

            else:
                params[key] = value

        # convert some log10_params to prospector style params
        params = convert_log10_params_to_prospector_params(params)
        all_params.append(params)

    return all_params

