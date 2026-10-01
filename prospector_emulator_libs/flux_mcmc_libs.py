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
# import corner
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
import prospector_emulator_libs.flux_emulator_libs as felibs
import prospector_emulator_libs.data_libs as dlibs
import prospector_emulator_libs.sps_libs as spslibs
import prospector_emulator_libs.mcmc_libs as mfuncs
from pprint import pprint
import torch


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
            parallel=None,
            vectorize=None,
            n_processes=None,
            verbose=None,
            output_dir=None,
            save_sampler=None,
            sampler_filename=None,
            output_filename=None,
            save_plots=None,
            plots_dir=None,
            ): 

        # initialize emulator to None
        self.emulator = None

        # copy a version of default_configs
        self.config = copy.deepcopy(mfuncs.default_configs)
        self.config["Outputs"]["sampler_filename"] = "flux_emulator_mcmc_sampler.h5"
        self.config["Outputs"]["output_filename"] = "flux_emulator_mcmc_results.h5"

        # if config file is provided, override the defaults
        if config_filename is not None:
            with open(config_filename, "r") as file:
                yaml_config = yaml.safe_load(file)
            self.config.update(yaml_config)

        mfuncs.apply_config_overrides(
            self.config,
            {
                "MCMC.nwalkers": nwalkers,
                "MCMC.jitter": jitter,
                "MCMC.nsteps": nsteps,
                "MCMC.discard": discard,
                "MCMC.thin": thin,
                "MCMC.zprior": zprior,
                "MCMC.parallel": parallel,
                "MCMC.vectorize": vectorize,
                "MCMC.n_processes": n_processes,
                "MCMC.verbose": verbose,
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
        self.vectorize = self.config["MCMC"]["vectorize"]
        self.n_processes = self.config["MCMC"]["n_processes"]
        self.verbose = self.config["MCMC"]["verbose"]
        self.output_dir = self.config["Outputs"]["output_dir"]
        self.save_sampler = self.config["Outputs"]["save_sampler"]
        self.sampler_filename = self.config["Outputs"]["sampler_filename"]
        self.output_filename = self.config["Outputs"]["output_filename"]
        self.plots_dir = self.config["Outputs"]["plots_dir"]
        self.save_plots = self.config["Outputs"]["save_plots"]

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
            if isinstance(emulator, felibs.LoadedFluxSPSEmulator):
                return emulator

            # Otherwise assume it is a filename/path
            return felibs.LoadedFluxSPSEmulator(emulator)

        # Try config
        emulator_path = self.config.get("emulator")

        if emulator_path is not None:
            return felibs.LoadedFluxSPSEmulator(emulator_path)

        # Allow incomplete construction
        return None

    def check_mcmc_ready(self):
        missing = []
        if self.emulator is None:
            missing.append("emulator")
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
            logprior_funcs.append(mfuncs.create_logprior_func(config))

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

        self.lamb_obs = self.emulator.lamb_obs
        self.train_param_keys = self.emulator.train_param_keys

        # ---------------------------------------------------------
        # Build free-prior and fixed-parameter dictionaries
        # ---------------------------------------------------------
        self.prior_dicts, self.fixed_dicts = self.build_mcmc_prior_dicts(
            prior_input, 
            self.emulator.prior_dicts
            )
        for key, value in self.emulator.default_params.items():
            if key not in self.train_param_keys:
                self.fixed_dicts.setdefault(key, value)
        # ---------------------------------------------------------
        # Full emulator parameter ordering
        # ---------------------------------------------------------
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

    def _setup_vectorized_device(self):
        """
        Put the emulator on the automatically selected device for
        vectorized MCMC evaluation.
        """
        self.emulator.to("auto")
        self.device = self.emulator.device

    
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
        logprior_funcs_new[self.zred_mcmc_index] = mfuncs.create_logprior_func(zred_config)

        # Object-specific initial value
        # initial_new[self.zred_index] = redshift

        return logprior_funcs_new#, initial_new


    def theta_to_x(self, theta):
        x = np.empty(self.ndim_train, dtype=float)
        x[self.index_fixed_in_train] = self.fixed_param_vals
        x[self.index_free_in_train] = theta
        return x

    def theta_batch_to_x(self, theta_batch):
        """
        Convert a batch of free MCMC parameters into the full emulator
        parameter vectors.

        Parameters
        ----------
        theta_batch : ndarray
            Shape (n_eval, ndim_mcmc).

        Returns
        -------
        x : ndarray
            Shape (n_eval, ndim_train).
        """

        theta_batch = np.asarray(theta_batch, dtype=float)
        if theta_batch.ndim != 2:
            raise ValueError("theta_batch must have shape (n_eval, ndim_mcmc).")
        n_eval = theta_batch.shape[0]
        x = np.empty((n_eval, self.ndim_train), dtype=float)
        if len(self.index_fixed_in_train) > 0:
            x[:, self.index_fixed_in_train] = self.fixed_param_vals
        x[:, self.index_free_in_train] = theta_batch
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


    def log_probability(self, theta, flux, flux_error, logprior_funcs=None):
        lp = self.log_prior(theta, logprior_funcs=logprior_funcs)

        if not np.isfinite(lp):
            return -np.inf, 0.0
        x = self.theta_to_x(theta)
        prediction = self.emulator.predict_one(x)
        flux_model = prediction["flux"]
        mfrac = prediction["mfrac"]
        ll = -0.5 * np.sum((flux_model-flux)**2 / flux_error**2 + np.log(2*np.pi*flux_error**2))
        log_prob = lp + ll

        return log_prob, mfrac


    def log_probability_vectorized(self, theta_batch, flux_t, flux_inv_var_t, flux_log_norm_t, logprior_funcs=None):
        """
        Vectorized log posterior for emcee.

        emcee supplies a batch of MCMC parameter vectors. Priors are
        evaluated on CPU, while valid positions are passed through the
        emulator in one batched call.

        The emulator is responsible for:
            - input scaling
            - neural-network inference
            - inverse flux scaling
            - inverse mfrac scaling

        Therefore prediction["flux"] is always physical linear flux.
        """

        theta_batch = np.asarray(theta_batch, dtype=float)

        if theta_batch.ndim != 2:
            raise ValueError("theta_batch must have shape (n_eval, ndim_mcmc).")

        n_eval = theta_batch.shape[0]
        if (theta_batch.shape[1] != self.ndim_mcmc):
            raise ValueError("theta_batch has incorrect parameter dimension.")

        # if logprior_funcs is None:logprior_funcs = self.logprior_funcs

        # =============================================================
        # 1. Priors
        # =============================================================
        #
        # Keep these on CPU. Your existing prior functions are scalar
        # scipy/Python callables and are cheap compared with model
        # inference.

        log_prior = np.empty(n_eval, dtype=float)

        for i in range(n_eval):
            log_prior[i] = self.log_prior(theta_batch[i], logprior_funcs=logprior_funcs)
        valid = np.isfinite(log_prior)

        # -------------------------------------------------------------
        # Allocate complete results.
        # -------------------------------------------------------------

        log_prob = np.full(n_eval, -np.inf, dtype=float,)
        mfrac = np.zeros(n_eval, dtype=float,)

        # If every proposal violates its prior, avoid the emulator
        # entirely.
        if not np.any(valid):
            return [(float(log_prob[i]), float(mfrac[i])) for i in range(n_eval)]

        # =============================================================
        # 2. Build full emulator input
        # =============================================================

        theta_valid = (theta_batch[valid])
        x_valid = (self.theta_batch_to_x(theta_valid))

        # =============================================================
        # 3. ONE batched emulator evaluation
        # =============================================================
        #
        # predict_torch() accepts RAW emulator parameters and returns
        # PHYSICAL quantities.
        #
        # It also leaves the result on self.emulator.device.

        prediction = self.emulator.predict_torch(x_valid)
        flux_model = prediction["flux"]

        # flux_model:
        #     (n_valid, n_flux)
        #
        # and is PHYSICAL LINEAR flux.

        if "mfrac" in prediction:
            mfrac_model = prediction["mfrac"]
        else:
            mfrac_model = torch.zeros(len(theta_valid), dtype=flux_model.dtype, device=self.device)

        # =============================================================
        # 4. Gaussian likelihood on device
        # =============================================================

        # residual = (flux_model - self._flux_obs_t.unsqueeze(0))
        # log_likelihood = -0.5 * torch.sum(residual.square() * self._flux_inv_var_t.unsqueeze(0) + self._flux_log_norm_t.unsqueeze(0), dim=1)
        residual = (flux_model - flux_t.unsqueeze(0))
        log_likelihood = -0.5 * torch.sum(residual.square() * flux_inv_var_t.unsqueeze(0) + flux_log_norm_t.unsqueeze(0), dim=1)
        # log_likelihood = -0.5 * torch.sum(residual.square() * flux_inv_var_t.unsqueeze(0), dim=1)

        # =============================================================
        # 5. Bring back only scalar results
        # =============================================================

        log_likelihood = (log_likelihood.cpu().numpy())
        mfrac_model = (mfrac_model.cpu().numpy())

        # =============================================================
        # 6. Posterior = prior + likelihood
        # =============================================================

        log_prob[valid] = (log_prior[valid] + log_likelihood)
        mfrac[valid] = mfrac_model

        # =============================================================
        # 7. emcee results
        # =============================================================

        return [(float(log_prob[i]), float(mfrac[i])) for i in range(n_eval)]

    def run_mcmc(
            self, 
            flux,
            flux_err=None,
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
            output_dir=None,
            save_sampler=None,
            sampler_filename=None,
            verbose=None,
            parallel=None,
            vectorize=None,
            n_processes=None,
            results=True,
            ):

        # if flux_err is None:
            # flux_err = np.ones_like(flux)
        # initial_user = initial is not None

        nwalkers = self.nwalkers if nwalkers is None else nwalkers
        jitter = self.jitter if jitter is None else jitter
        nsteps = self.nsteps if nsteps is None else nsteps
        zprior = self.zprior if zprior is None else zprior
        discard = self.discard if discard is None else discard
        thin = self.thin if thin is None else thin

        output_dir = self.output_dir if output_dir is None else output_dir
        save_sampler = self.save_sampler if save_sampler is None else save_sampler
        sampler_filename = self.sampler_filename if sampler_filename is None else sampler_filename
        verbose = self.verbose if verbose is None else verbose
        parallel = self.parallel if parallel is None else parallel
        vectorize = self.vectorize if vectorize is None else vectorize
        n_processes = self.n_processes if n_processes is None else n_processes

        self.check_mcmc_ready()

        # If setup has not happened, do it now.
        # if self.prior_dicts is None or self.logprior_funcs is None:
        self.setup_mcmc(prior=prior)
        if vectorize:
            self._setup_vectorized_device()
            if len(flux) != len(self.lamb_obs):
                raise ValueError("Observed flux length does not match the emulator output grid.")

            # save flux and flux err to device
            model_dtype = next(self.emulator.model.parameters()).dtype
            flux_t = torch.as_tensor(flux, dtype=model_dtype, device=self.device)
            flux_err_t = torch.as_tensor(flux_err, dtype=model_dtype, device=self.device)
            flux_inv_var_t = 1.0 / flux_err_t.square()
            flux_log_norm_t = torch.log(2.0 * torch.pi * flux_err_t.square())

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
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
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
                    args=(flux, flux_err, logprior_funcs),
                    backend=backend,
                    pool=pool
                )
                self.sampler.run_mcmc(initial_pos, nsteps, progress=verbose)
        elif vectorize:
            self.sampler = emcee.EnsembleSampler(
                    nwalkers=nwalkers,
                    ndim=self.ndim_mcmc,
                    log_prob_fn=self.log_probability_vectorized,
                    # Observed flux/error already live on device.
                    # Only pass the potentially run-specific priors.
                    args=(flux_t, flux_inv_var_t, flux_log_norm_t, logprior_funcs),
                    backend=backend,
                    vectorize=True,
                    # One mfrac scalar stored for every sample.
                    blobs_dtype=float,
                )
            self.sampler.run_mcmc(initial_pos, nsteps, progress=verbose)

        else:
            self.sampler = emcee.EnsembleSampler(
                nwalkers=nwalkers,
                ndim=self.ndim_mcmc,
                log_prob_fn=self.log_probability,
                args=(flux, flux_err, logprior_funcs),
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
        theta_percentiles = mfuncs.get_theta_percentiles(flat_samples, percentiles=[16, 50, 84])
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

        flux_med = prediction_med["flux"]
        mfrac_med = mfracs_percentiles[1]

        # save some med results for easy access
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
                'flux_fiducial_med': flux_med,
                'mfrac_med': mfrac_med,
                'mfracs_percentiles': mfracs_percentiles,
                'agebins_med': agebins_med,
                'massbins_med': massbins_med,
                'sfrs_med': med_sfrs_med,
                'lamb_obs': self.lamb_obs,
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
        print("vectorize", self.vectorize)
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
                            'parallel': self.parallel,
                            'vectorize': self.vectorize
                            }
                        )

    