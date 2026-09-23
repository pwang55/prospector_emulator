import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset #, DataLoader
import json
# from pathlib import Path
# import yaml
# from numba import njit
# from scipy.sparse.linalg import eigsh
import h5py

# input and output data scaler class
class FluxEmulatorScaler:
    """
    Preprocessing for an flux SPS emulator.

    x:

    flux:

    mfrac:

    """

    def __init__(self, eps=1e-12):
        self.eps = float(eps)
        self.x_mean = None
        self.x_std = None
        self.flux_mean = None
        self.flux_std = None
        self.mfrac_mean = None
        self.mfrac_std = None

    def fit(self,
            x,
            flux,
            mfrac,
            flux_scaling='standard',
            mfrac_scaling='standard'):
        x = np.asarray(x, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        mfrac = np.asarray(mfrac, dtype=np.float64)

        valid_flux_scalings = {
            'none',
            'standard',
            'global_standard',
            'log10_none',
            'log10_standard',
            'log10_global_standard'
        }
        valid_mfrac_scalings = {
            'none',
            'standard',
        }
        if flux_scaling not in valid_flux_scalings:
            raise ValueError(f'flux_scaling must be one of {valid_flux_scalings}')
        if mfrac_scaling not in valid_mfrac_scalings:
            raise ValueError(f'mfrac_scaling must be one of {valid_mfrac_scalings}')

        self.flux_scaling = flux_scaling
        self.mfrac_scaling = mfrac_scaling

        if x.ndim != 2:
            raise ValueError("x must have shape (ndata, nfeatures).")
        if flux.ndim != 2:
            raise ValueError("flux must have shape (ndata, nfilts).")
        if mfrac.ndim != 1:
            raise ValueError("mfrac must have shape (ndata,).")

        ndata = x.shape[0]

        if flux.shape[0] != ndata:
            raise ValueError("x and flux must contain the same number of samples.")
        if mfrac.shape[0] != ndata:
            raise ValueError("x and mfrac must contain the same number of samples.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN or infinity.")
        if not np.all(np.isfinite(flux)):
            raise ValueError("flux contains NaN or infinity.")
        if not np.all(np.isfinite(mfrac)):
            raise ValueError("mfrac contains NaN or infinity.")

        # Input scaling, one mean/std per input feature
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)

        constant_features = self.x_std < self.eps
        if np.any(constant_features):
            indices = np.flatnonzero(constant_features)
            raise ValueError("The following input features have nearly zero "f"standard deviation: {indices.tolist()}")

        # flux scaling
        # self.flux_mean = np.mean(flux, axis=0)
        # self.flux_std = np.std(flux, axis=0)

        if self.flux_scaling == "none" or self.flux_scaling == "log10_none":
            self.flux_mean = None
            self.flux_std = None

        elif self.flux_scaling == "standard":
            # Per-wavelength statistics in LINEAR flux space.
            self.flux_mean = np.mean(flux, axis=0)
            self.flux_std = np.std(flux, axis=0)

            bad_wavelengths = self.flux_std < self.eps

            if np.any(bad_wavelengths):
                indices = np.flatnonzero(bad_wavelengths)
                raise ValueError(f"The following wavelengths have nearly zero flux standard deviation: {indices.tolist()}")

        elif self.flux_scaling == "global_standard":
            # One shared mean/std for all wavelengths and samples.
            self.flux_mean = float(np.mean(flux))
            self.flux_std = float(np.std(flux))

            if self.flux_std < self.eps:
                raise ValueError("The global flux standard deviation is too small.")

        elif self.flux_scaling == "log10_standard":
            # log10_standard is defined entirely in log10-flux space.
            #
            # IMPORTANT:
            # flux_mean and flux_std below therefore refer to
            # statistics of log10(flux), not statistics of flux.

            if np.any(flux <= 0.0):
                raise ValueError("flux must be strictly positive when flux_scaling='log10_standard'.")
            log10_flux = np.log10(flux)

            self.flux_mean = np.mean(log10_flux, axis=0)
            self.flux_std = np.std(log10_flux, axis=0)

            bad_wavelengths = self.flux_std < self.eps

            if np.any(bad_wavelengths):
                indices = np.flatnonzero(bad_wavelengths)
                raise ValueError(f"The following wavelengths have nearly zero log10-flux standard deviation: {indices.tolist()}")

        elif self.flux_scaling == "log10_global_standard":
            # log10_global_standard is defined entirely in log10-flux space.
            #
            # IMPORTANT:
            # flux_mean and flux_std below therefore refer to
            # statistics of log10(flux), not statistics of flux.

            if np.any(flux <= 0.0):
                raise ValueError("flux must be strictly positive when flux_scaling='log10_global_standard'.")

            log10_flux = np.log10(flux)

            self.flux_mean = float(np.mean(log10_flux))
            self.flux_std = float(np.std(log10_flux))

            if self.flux_std < self.eps:
                raise ValueError("The global log10flux standard deviation is too small.")

        # Separate mfrac scaling
        self.mfrac_mean = float(np.mean(mfrac))
        self.mfrac_std = float(np.std(mfrac))

        if self.mfrac_std < self.eps and self.mfrac_scaling == 'standard':
            raise ValueError("The standard deviation of mfrac is too small.")

        return self

    def transform_x(self, x):
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return (x - self.x_mean) / self.x_std

    def transform_flux(self, flux):
        self._check_fitted()
        flux = np.asarray(flux, dtype=np.float64)

        if self.flux_scaling == "none":
            return flux

        elif self.flux_scaling == "standard":
            return (
                (flux - self.flux_mean)
                / self.flux_std
            )

        elif self.flux_scaling == "global_standard":
            return (
                (flux - self.flux_mean)
                / self.flux_std
            )

        elif self.flux_scaling == "log10_none":
            if np.any(flux <= 0.0):
                raise ValueError("flux must be strictly positive when flux_scaling='log10_none'.")
            log10_flux = np.log10(flux)
            return log10_flux

        elif self.flux_scaling == "log10_standard":
            if np.any(flux <= 0.0):
                raise ValueError("flux must be strictly positive when flux_scaling='log10_standard'.")

            log10_flux = np.log10(flux)
            return ((log10_flux - self.flux_mean) / self.flux_std)

        elif self.flux_scaling == "log10_global_standard":
            if np.any(flux <= 0.0):
                raise ValueError("flux must be strictly positive when flux_scaling='log10_global_standard'.")

            log10_flux = np.log10(flux)
            return ((log10_flux - self.flux_mean) / self.flux_std)
    
    def transform_mfrac(self, mfrac):
        self._check_fitted()
        mfrac = np.asarray(mfrac, dtype=np.float64)

        if self.mfrac_scaling == 'none':
            return mfrac

        elif self.mfrac_scaling == 'standard':
            return (mfrac - self.mfrac_mean) / self.mfrac_std

    def transform(self, x, flux, mfrac):
        return (
            self.transform_x(x),
            self.transform_flux(flux),
            self.transform_mfrac(mfrac),
        )

    def fit_transform(self, 
                      x, 
                      flux, 
                      mfrac,
                      flux_scaling='standard',
                      mfrac_scaling='standard'):
        self.fit(x, flux, mfrac, flux_scaling=flux_scaling, mfrac_scaling=mfrac_scaling)
        return self.transform(x, flux, mfrac)
    
    def inverse_transform_x(self, x_scaled):
        self._check_fitted()
        x_scaled = np.asarray(x_scaled, dtype=np.float64)

        return (x_scaled * self.x_std + self.x_mean)


    def inverse_transform_flux(self, flux_scaled):
        self._check_fitted()

        flux_scaled = np.asarray(flux_scaled, dtype=np.float64)

        if self.flux_scaling == "none":
            return flux_scaled

        elif self.flux_scaling == "standard":
            return (flux_scaled * self.flux_std + self.flux_mean)

        elif self.flux_scaling == "global_standard":
            return (flux_scaled * self.flux_std + self.flux_mean)

        elif self.flux_scaling == "log10_none":
            log10_flux = flux_scaled
            return 10.0 ** log10_flux
        elif self.flux_scaling == "log10_standard":
            log10_flux = (flux_scaled * self.flux_std + self.flux_mean)
            return 10.0 ** log10_flux

        elif self.flux_scaling == "log10_global_standard":
            log10_flux = (flux_scaled * self.flux_std + self.flux_mean)
            return 10.0 ** log10_flux

    def inverse_transform_mfrac(self, mfrac_scaled):
        self._check_fitted()
        mfrac_scaled = np.asarray(mfrac_scaled, dtype=np.float64)

        if self.mfrac_scaling == 'none':
            return mfrac_scaled

        elif self.mfrac_scaling == 'standard':
            return mfrac_scaled * self.mfrac_std + self.mfrac_mean

    def _check_fitted(self):
        if self.x_mean is None:
            raise RuntimeError("The scaler has not been fitted.")

    def state_dict(self):
        """
        Return all configuration and fitted statistics needed
        to reconstruct this scaler.
        """
        self._check_fitted()

        return {
            "eps": self.eps,

            "flux_scaling": self.flux_scaling,
            "mfrac_scaling": self.mfrac_scaling,

            "x_mean": self.x_mean,
            "x_std": self.x_std,

            "flux_mean": self.flux_mean,
            "flux_std": self.flux_std,

            "mfrac_mean": self.mfrac_mean,
            "mfrac_std": self.mfrac_std,
        }

    @classmethod
    def from_state_dict(cls, state):
        """
        Reconstruct a fitted FluxEmulatorScaler from a saved state dictionary.
        """
        scaler = cls(eps=state["eps"])

        scaler.flux_scaling = state["flux_scaling"]
        scaler.mfrac_scaling = state["mfrac_scaling"]

        scaler.x_mean = state["x_mean"]
        scaler.x_std = state["x_std"]

        scaler.flux_mean = state["flux_mean"]
        scaler.flux_std = state["flux_std"]

        scaler.mfrac_mean = state["mfrac_mean"]
        scaler.mfrac_std = state["mfrac_std"]

        return scaler


# Dataset class
class FluxSPSDataset(Dataset):
    def __init__(
        self,
        x_scaled,
        flux_scaled,
        mfrac_scaled,
    ):
        self.x = torch.as_tensor(x_scaled, dtype=torch.float32)
        self.flux = torch.as_tensor(flux_scaled, dtype=torch.float32)
        self.mfrac = torch.as_tensor(mfrac_scaled, dtype=torch.float32)

        ndata = self.x.shape[0]

        if self.x.ndim != 2:
            raise ValueError("x must have shape (ndata, nfeatures).")

        if self.flux.ndim != 2:
            raise ValueError("flux must have shape (ndata, nfilts).")

        if self.mfrac.ndim != 1:
            raise ValueError("mfrac must have shape (ndata,).")

        if self.flux.shape[0] != ndata:
            raise ValueError("x and c have different ndata")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        # return self.x[index], self.flux[index], self.mfrac[index]
        return {
            'x': self.x[index], 
            'flux': self.flux[index], 
            'mfrac': self.mfrac[index]
        }



# Class to create a full emulator architecture
class FluxSPSEmulator(nn.Module):
    def __init__(
        self,
        n_features,
        n_fluxes,
        shared_dims=(512, 512, 512, 512),
        flux_head_dims=(),
        mfrac_head_dims=(),
        activation="gelu",
        dropout=0.0,
        predict_mfrac=True
    ):
        super().__init__()

        self.n_features = int(n_features)
        self.n_fluxes = int(n_fluxes)
        self.predict_mfrac = predict_mfrac

        self.config = {
            "n_features": self.n_features,
            "n_fluxes": self.n_fluxes,
            "shared_dims": tuple(shared_dims),
            "flux_head_dims": tuple(flux_head_dims),
            "mfrac_head_dims": tuple(mfrac_head_dims),
            "activation": str(activation),
            "dropout": float(dropout),
            "predict_mfrac": bool(self.predict_mfrac)
        }

        self.shared, shared_output_dim = make_hidden_mlp(
            input_dim=self.n_features,
            hidden_dims=shared_dims,
            activation=activation,
            dropout=dropout,
        )

        self.flux_head_hidden, flux_hidden_dim = (
            make_hidden_mlp(
                input_dim=shared_output_dim,
                hidden_dims=flux_head_dims,
                activation=activation,
                dropout=dropout,
            )
        )

        self.flux_output = nn.Linear(flux_hidden_dim, self.n_fluxes)

        if self.predict_mfrac:
            self.mfrac_head_hidden, mfrac_hidden_dim = (
                make_hidden_mlp(
                    input_dim=shared_output_dim,
                    hidden_dims=mfrac_head_dims,
                    activation=activation,
                    dropout=dropout,
                )
            )
            self.mfrac_output = nn.Linear(mfrac_hidden_dim, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        flux_features = self.flux_head_hidden(shared_features)
        flux_pred = self.flux_output(flux_features)
        output = {'flux': flux_pred}

        if self.predict_mfrac:
            mfrac_features = self.mfrac_head_hidden(shared_features)
            mfrac_pred = self.mfrac_output(mfrac_features).squeeze(-1)
            output['mfrac'] = mfrac_pred

        return output

    
class FluxEmulatorLoss(nn.Module):
    """
    Loss for a direct-flux SPS emulator.

    Parameters
    ----------
    flux_loss : {"mse", "mae"}
        Loss function used for flux.

    flux_loss_space : {"scaled", "log10", "linear"}
        Space in which the flux loss is calculated.

        "scaled"
            Calculate the loss directly on the output of FluxEmulatorScaler,
            regardless of how they were scaled.

        "log10"
            Convert the scaled representation into unscaled
            log10(flux), then calculate the loss there.

        "linear"
            Convert the scaled representation into physical
            linear flux, then calculate the loss there.

    flux_scaling : {
        "none",
        "standard",
        "global_standard",
        "log10_none",
        "log10_standard",
        "log10_global_standard"
    }
        Scaling/transformation used by FluxEmulatorScaler.

    flux_scale_mean : scalar, array-like, or None
        Mean stored by FluxEmulatorScaler.

    flux_scale_std : scalar, array-like, or None
        Standard deviation stored by FluxEmulatorScaler.

    mfrac_loss : {"mse", "mae"}
        Loss function used for mfrac.

    mfrac_lambda : float
        Relative weight of the mfrac loss.
        total = flux_loss + mfrac_lambda * mfrac_loss

    wavelength_weights : array-like or None
        Optional wavelength-dependent loss weights.

    max_log10_flux : float or None
        Optional upper clipping threshold before converting
        log10 flux into linear flux.

        Normally None.
    """

    def __init__(
        self,
        flux_loss="mse",
        flux_loss_space="scaled",
        flux_scaling="standard",
        flux_scale_mean=None,
        flux_scale_std=None,
        mfrac_loss="mse",
        mfrac_lambda=0.01,
        wavelength_weights=None,
        max_log10_flux=None,
    ):
        super().__init__()

        self.flux_loss_name = flux_loss.lower()
        self.flux_loss_space = flux_loss_space.lower()
        self.flux_scaling = flux_scaling.lower()

        self.mfrac_loss_name = mfrac_loss.lower()
        self.mfrac_lambda = float(mfrac_lambda)

        self.max_log10_flux = max_log10_flux

        # =============================================================
        # Validate options
        # =============================================================

        valid_flux_losses = {
            "mse",
            "mae",
        }

        valid_flux_loss_spaces = {
            "scaled",
            "log10",
            "linear",
        }

        valid_flux_scalings = {
            "none",
            "standard",
            "global_standard",
            "log10_none",
            "log10_standard",
            "log10_global_standard",
        }

        valid_mfrac_losses = {
            "mse",
            "mae",
        }

        if self.flux_loss_name not in valid_flux_losses:
            raise ValueError(f"Unknown flux_loss {self.flux_loss_name!r}. Available options: {valid_flux_losses}")

        if self.flux_loss_space not in valid_flux_loss_spaces:
            raise ValueError(f"Unknown flux_loss_space {self.flux_loss_space!r}. Available options: {valid_flux_loss_spaces}")

        if self.flux_scaling not in valid_flux_scalings:
            raise ValueError(f"Unknown flux_scaling {self.flux_scaling!r}. Available options: {valid_flux_scalings}")

        if self.mfrac_loss_name not in valid_mfrac_losses:
            raise ValueError(f"Unknown mfrac_loss {self.mfrac_loss_name!r}. Available options: {valid_mfrac_losses}")

        if self.mfrac_lambda < 0.0:
            raise ValueError("mfrac_lambda must be nonnegative.")

        # =============================================================
        # Determine whether scaler statistics are required
        # =============================================================

        scaling_uses_stats = self.flux_scaling in {
            "standard",
            "global_standard",
            "log10_standard",
            "log10_global_standard",
        }

        # If loss is calculated directly in scaled space,
        # the scaler statistics are not needed.
        #
        # If loss is requested in log10 or linear space,
        # scaled representations must first be undone.
        if (
            self.flux_loss_space != "scaled"
            and scaling_uses_stats
        ):
            if flux_scale_mean is None:
                raise ValueError(
                    "flux_scale_mean is required when "
                    f"flux_loss_space={self.flux_loss_space!r} "
                    "and the chosen flux_scaling uses "
                    "mean/std scaling."
                )

            if flux_scale_std is None:
                raise ValueError(
                    "flux_scale_std is required when "
                    f"flux_loss_space={self.flux_loss_space!r} "
                    "and the chosen flux_scaling uses "
                    "mean/std scaling."
                )

        # =============================================================
        # Store scaler statistics as buffers
        # =============================================================

        if flux_scale_mean is None:
            self.flux_scale_mean = None

        else:
            self.register_buffer(
                "flux_scale_mean",
                torch.as_tensor(flux_scale_mean, dtype=torch.float32)
                )

        if flux_scale_std is None:
            self.flux_scale_std = None

        else:
            self.register_buffer(
                "flux_scale_std",
                torch.as_tensor(flux_scale_std, dtype=torch.float32)
            )

        # =============================================================
        # Wavelength weights
        # =============================================================

        if wavelength_weights is None:
            self.wavelength_weights = None

        else:
            wavelength_weights = torch.as_tensor(wavelength_weights, dtype=torch.float32)
            if wavelength_weights.ndim != 1:
                raise ValueError("wavelength_weights must have shape (nwavelength,).")
            self.register_buffer("wavelength_weights",wavelength_weights)

    # =================================================================
    # Basic scaling inversion
    # =================================================================

    def inverse_scale_flux(self, flux_scaled):
        """
        Undo mean/std scaling.

        This does NOT change linear flux into log10 flux or vice versa.

        Returns
        -------
        torch.Tensor

        Interpretation depends on flux_scaling:

        none
            -> linear flux

        standard
            -> linear flux

        global_standard
            -> linear flux

        log10_none
            -> log10 flux

        log10_standard
            -> log10 flux

        log10_global_standard
            -> log10 flux
        """

        if self.flux_scaling in {
            "none",
            "log10_none",
        }:
            return flux_scaled

        return (flux_scaled * self.flux_scale_std + self.flux_scale_mean)

    # =================================================================
    # Convert to log10 flux
    # =================================================================

    def to_log10_flux(self, flux_scaled):
        """
        Convert scaled model representation into unscaled log10 flux.
        """

        flux_unscaled = self.inverse_scale_flux(flux_scaled)

        # -------------------------------------------------------------
        # Already log10 flux
        # -------------------------------------------------------------

        if self.flux_scaling in {
            "log10_none",
            "log10_standard",
            "log10_global_standard",
        }:
            return flux_unscaled

        # -------------------------------------------------------------
        # Currently linear flux
        # -------------------------------------------------------------

        if self.flux_scaling in {
            "none",
            "standard",
            "global_standard",
        }:
            if torch.any(flux_unscaled <= 0.0):
                raise ValueError("Cannot calculate log10-space loss because the reconstructed linear flux contains non-positive values.")
            return torch.log10(flux_unscaled)

        raise RuntimeError("Unreachable flux-scaling branch.")

    # =================================================================
    # Convert to physical linear flux
    # =================================================================

    def to_linear_flux(self, flux_scaled):
        """
        Convert scaled model representation into physical linear flux.
        """

        flux_unscaled = self.inverse_scale_flux(flux_scaled)

        # -------------------------------------------------------------
        # Already linear flux
        # -------------------------------------------------------------

        if self.flux_scaling in {
            "none",
            "standard",
            "global_standard",
        }:
            return flux_unscaled

        # -------------------------------------------------------------
        # Currently log10 flux
        # -------------------------------------------------------------

        if self.flux_scaling in {
            "log10_none",
            "log10_standard",
            "log10_global_standard",
        }:
            log10_flux = flux_unscaled

            if self.max_log10_flux is not None:
                log10_flux = torch.clamp(log10_flux,max=float(self.max_log10_flux))
            return torch.pow(10.0,log10_flux,)

        raise RuntimeError("Unreachable flux-scaling branch.")

    # =================================================================
    # Convert representation according to requested loss space
    # =================================================================

    def convert_flux_for_loss(
        self,
        flux_scaled,
    ):
        """
        Convert flux into the representation requested by
        flux_loss_space.
        """

        if self.flux_loss_space == "scaled":
            return flux_scaled

        elif self.flux_loss_space == "log10":
            return self.to_log10_flux(flux_scaled)

        elif self.flux_loss_space == "linear":
            return self.to_linear_flux(flux_scaled)

        raise RuntimeError("Unreachable flux-loss-space branch.")

    # =================================================================
    # Flux residual reduction
    # =================================================================

    def reduce_flux_residual(
        self,
        residual,
    ):
        """
        Apply MSE or MAE, optionally with wavelength weights.
        """

        if self.flux_loss_name == "mse":
            element_loss = residual.square()

        elif self.flux_loss_name == "mae":
            element_loss = torch.abs(residual)

        else:
            raise RuntimeError("Unreachable flux-loss branch.")

        # -------------------------------------------------------------
        # No wavelength weighting
        # -------------------------------------------------------------

        if self.wavelength_weights is None:
            return torch.mean(element_loss)

        # -------------------------------------------------------------
        # Wavelength weighting
        # -------------------------------------------------------------

        weights = self.wavelength_weights

        # Normalize so mean weight = 1.
        weights = (weights / torch.mean(weights))
        return torch.mean(element_loss * weights.unsqueeze(0))

    # =================================================================
    # Flux loss
    # =================================================================

    def calculate_flux_loss(
        self,
        flux_pred_scaled,
        flux_true_scaled,
    ):
        """
        Calculate flux loss.

        Both prediction and target are assumed to have already
        passed through FluxEmulatorScaler.
        """

        flux_pred_for_loss = self.convert_flux_for_loss(flux_pred_scaled)
        flux_true_for_loss = self.convert_flux_for_loss(flux_true_scaled)
        residual = (flux_pred_for_loss - flux_true_for_loss)
        return self.reduce_flux_residual(residual)

    # =================================================================
    # mfrac loss
    # =================================================================

    def calculate_mfrac_loss(
        self,
        mfrac_pred_scaled,
        mfrac_true_scaled,
    ):
        residual = (mfrac_pred_scaled - mfrac_true_scaled)

        if self.mfrac_loss_name == "mse":
            return torch.mean(residual.square())

        elif self.mfrac_loss_name == "mae":
            return torch.mean(torch.abs(residual))

        raise RuntimeError("Unreachable mfrac-loss branch.")

    # =================================================================
    # Full loss
    # =================================================================

    def forward(
        self,
        prediction,
        flux_true_scaled,
        mfrac_true_scaled=None,
    ):
        """
        Parameters
        ----------
        prediction : dict
            Model output. Must contain:
                prediction["flux"]

            and optionally:
                prediction["mfrac"]

        flux_true_scaled : torch.Tensor
            True flux after passing through FluxEmulatorScaler.

        mfrac_true_scaled : torch.Tensor or None
            True mfrac after passing through FluxEmulatorScaler.
        """

        loss_flux = self.calculate_flux_loss(prediction["flux"], flux_true_scaled)

        # -------------------------------------------------------------
        # No mfrac prediction
        # -------------------------------------------------------------

        if "mfrac" not in prediction:
            loss_mfrac = (loss_flux.new_zeros(()))

            return {
                "total": loss_flux,
                "flux": loss_flux,
                "mfrac": loss_mfrac,
            }

        # -------------------------------------------------------------
        # mfrac prediction
        # -------------------------------------------------------------

        if mfrac_true_scaled is None:
            raise ValueError("mfrac_true_scaled is required when the model predicts mfrac.")

        loss_mfrac = self.calculate_mfrac_loss(prediction["mfrac"], mfrac_true_scaled)
        loss_total = loss_flux + self.mfrac_lambda * loss_mfrac

        return {
            "total": loss_total,
            "flux": loss_flux,
            "mfrac": loss_mfrac,
        }



class EarlyStopping:
    def __init__(
        self,
        patience=50,
        abs_tol=0.0,
        rel_tol=1e-3,
    ):
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if abs_tol < 0.0 or rel_tol < 0.0:
            raise ValueError("abs_tol and rel_tol must be nonnegative.")

        self.patience = int(patience)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.best_loss = float("inf")
        self.best_state = None
        self.best_epoch = None
        self.epochs_without_improvement = 0

    def update(
        self,
        monitored_loss,
        model,
        epoch,
    ):
        monitored_loss = float(monitored_loss)

        if self.best_loss == float("inf"):
            improved = True
        else:
            required_improvement = max(self.abs_tol, self.rel_tol * abs(self.best_loss))
            improved = (monitored_loss < self.best_loss - required_improvement)

        if improved:
            self.best_loss = monitored_loss

            # Store best weights on CPU.
            self.best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor
                in model.state_dict().items()
            }

            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0

        else:
            self.epochs_without_improvement += 1

        return (self.epochs_without_improvement >= self.patience)

    def restore_best_weights(
        self,
        model,
        device,
    ):
        if self.best_state is None:
            raise RuntimeError("No best model state was recorded.")

        model.load_state_dict(self.best_state)
        model.to(device)



# class to load pre-trained emulator from .pt file and can be used for prediction
class LoadedFluxSPSEmulator:
    def __init__(
        self,
        checkpoint_path,
        device="cpu",
    ):
        # self.device = torch.device(device)
        self.device = get_device(device)

        # Load checkpoint once
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        # Reconstruct model
        self.model = FluxSPSEmulator(**self.checkpoint["model_config"])
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Reconstruct fitted scaler
        self.scaler = FluxEmulatorScaler.from_state_dict(self.checkpoint["scaler_state"])

        self.train_param_keys = self.checkpoint["train_param_keys"]
        self.default_params = self.checkpoint["default_params"]
        self.prior_dicts = self.checkpoint["prior_dicts"]

    # single object inference method
    @torch.inference_mode()
    def predict_one(self, x):
        x_scaled = self.scaler.transform_x(x)
        model_dtype = next(self.model.parameters()).dtype

        x_tensor = torch.as_tensor(
            x_scaled,
            dtype=model_dtype,
            device=self.device,
        ).reshape(1, -1)

        prediction = self.model(x_tensor)
        flux_scaled = (prediction["flux"][0].cpu().numpy())
        flux = self.scaler.inverse_transform_flux(flux_scaled)

        result = {
            "flux_scaled": flux_scaled,
            "flux": flux,
        }

        if "mfrac" in prediction:
            mfrac_scaled = (prediction["mfrac"][0].cpu().item())
            result["mfrac_scaled"] = mfrac_scaled
            result["mfrac"] = (self.scaler.inverse_transform_mfrac(mfrac_scaled))

        return result

    # multiple object inference method
    @torch.inference_mode()
    def predict(self, x, batch_size=2000):
        x = np.asarray(x)

        # Redirect one-dimensional input to predict_one().
        if x.ndim == 1:
            return self.predict_one(x=x)

        if x.ndim != 2:
            raise ValueError("x must have shape (n_features,) or (n_objects, n_features).")

        x_scaled = self.scaler.transform_x(x)
        model_dtype = next(self.model.parameters()).dtype

        x_tensor = torch.as_tensor(x_scaled, dtype=model_dtype)
        flux_batches = []
        mfrac_batches = []

        for start in range(0, x_tensor.shape[0], batch_size):
            x_batch = x_tensor[start:start + batch_size].to(self.device)
            prediction = self.model(x_batch)
            flux_batches.append(prediction["flux"].cpu())

            if "mfrac" in prediction:
                mfrac_batches.append(prediction["mfrac"].cpu())

        flux_scaled = torch.cat(flux_batches, dim=0,).numpy()
        flux = self.scaler.inverse_transform_flux(flux_scaled)

        result = {
            "flux_scaled": flux_scaled,
            "flux": flux,
        }

        if mfrac_batches:
            mfrac_scaled = torch.cat(mfrac_batches, dim=0).numpy()
            result["mfrac_scaled"] = mfrac_scaled
            result["mfrac"] = (self.scaler.inverse_transform_mfrac(mfrac_scaled))

        return result

    def to(self, device):
        # self.device = torch.device(device)
        self.device = get_device(device)
        self.model.to(self.device)
        return self




# ===================================

def get_device(requested="auto"):
    requested = requested.lower()

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_data(filename):

    if filename.split('.')[-1] == "npz":
        dat = np.load(filename)
        x = dat["x"]
        lbs = dat["lbs"]
        coef = dat["coef"]
        mfrac = dat["mfrac"]
        train_param_keys = dat['train_param_keys'].tolist()
        default_params = json.loads(dat["default_params"].item())
        prior_dicts=json.loads(dat["prior_dicts"].item())
        try:
            flux = dat["flux"]
        except:
            flux = None

        out_dict = {
            "x": x,
            "lbs": lbs,
            "coef": coef,
            "mfrac": mfrac,
            "flux": flux,
            "train_param_keys": train_param_keys,
            "default_params": default_params,
            "prior_dicts": prior_dicts
        }
        return out_dict
        # return x_test, coef_test, mfrac_test, flux
    elif filename.split('.')[-1] == "h5":
        with h5py.File(filename, "r") as dat:
            lamb_obs = dat["lamb_obs"][()]
            x = dat["x"][()]
            lbs = dat["lbs"][()]
            coef = dat["coef"][()]
            flux_fiducial = dat["flux_fiducial"][()]
            mfrac = dat["mfrac"][()]
            scale = dat["scale"][()]
            train_param_keys = dat.attrs["train_param_keys"].tolist()
            default_params = json.loads(dat.attrs["default_params"])
            prior_dicts = json.loads(dat.attrs["prior_dicts"])
            try:
                flux = 10**(dat["log10flux"][()])
            except:
                flux = None
        out_dict = {
            "x": x,
            "lbs": lbs,
            "lamb_obs": lamb_obs,
            "coef": coef,
            "mfrac": mfrac,
            "flux": flux,
            "flux_fiducial": flux_fiducial,
            "scale": scale,
            "train_param_keys": train_param_keys,
            "default_params": default_params,
            "prior_dicts": prior_dicts
        }
        return out_dict
    
def get_activation(name):
    name = name.lower()
    activation_classes = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activation_classes:
        raise ValueError(
            f"Unknown activation {name!r}. "
            "Available options are "
            f"{list(activation_classes)}."
        )
    return activation_classes[name]

# function to make single hidden layer
def make_hidden_mlp(
    input_dim,
    hidden_dims,
    activation="gelu",
    dropout=0.0,
):
    """
    Create hidden layers without an output layer.

    Parameters
    ----------
    input_dim : int
        Input representation size.
    hidden_dims : sequence of int
        Width of each hidden layer. An empty tuple creates
        an identity operation.
    activation : str
        Activation-function name.
    dropout : float
        Dropout probability.

    Returns
    -------
    module : nn.Module
        Hidden network.
    final_dim : int
        Dimension of the final representation.
    """
    Activation = get_activation(activation)
    hidden_dims = tuple(hidden_dims)
    layers = []
    previous_dim = int(input_dim)

    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)

        if hidden_dim <= 0:
            raise ValueError("All hidden-layer widths must be positive.")

        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(Activation())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        previous_dim = hidden_dim

    if not layers:
        return nn.Identity(), previous_dim

    return nn.Sequential(*layers), previous_dim


def make_optimizer(
    name,
    parameters,
    learning_rate=1e-3,
    weight_decay=0.0,
    **kwargs,
):
    name = name.lower()

    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    raise ValueError("Optimizer must be 'adam', 'adamw', 'sgd', or 'rmsprop'.")


# one full epoch run function that makes data go through MLP and returns loss and gradient if training
def run_flux_epoch(
    model,
    data_loader,
    criterion,
    device,
    optimizer=None,
):
    """
    Run one training or validation epoch.

    If optimizer is None:
        Run validation.
    If optimizer is provided:
        Run training.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    accumulated = {
        "total": 0.0,
        "flux": 0.0,
        "mfrac": 0.0,
    }

    n_samples = 0

    grad_context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with grad_context:
        for batch in data_loader:
            x_batch = batch["x"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            flux_true = batch["flux"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            mfrac_true = batch["mfrac"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(x_batch)

            losses = criterion(
                prediction,
                flux_true,
                mfrac_true,
            )

            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    "The loss became nonfinite. "
                    "This is especially possible for linear-flux "
                    "losses because 10**log_flux can overflow."
                )

            if is_training:
                losses["total"].backward()
                optimizer.step()

            batch_size = x_batch.shape[0]
            n_samples += batch_size

            for name in accumulated:
                accumulated[name] += (
                    float(losses[name].detach().cpu())
                    * batch_size
                )

    if n_samples == 0:
        raise RuntimeError("The DataLoader returned no samples.")

    return {
        name: value / n_samples
        for name, value in accumulated.items()
    }


# this function takes train & valid data, model, optimizer, loss class and use run_one_epoch to actually fit the emulator
# returns training history
def fit_flux_emulator(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    max_epochs=500,
    patience=20,
    abs_tol=0.0,
    rel_tol=1e-5,
    monitor="total",
    verbose=True,
):
    if monitor not in {
        "total",
        "flux",
        "mfrac",
    }:
        raise ValueError("monitor must be 'total', 'flux', or 'mfrac'.")

    # Ensure both model and criterion are on the right device.
    model.to(device)
    criterion.to(device)

    early_stopping = EarlyStopping(
        patience=patience,
        abs_tol=abs_tol,
        rel_tol=rel_tol
    )

    history = {
        "train_total": [],
        "train_flux": [],
        "train_mfrac": [],
        "valid_total": [],
        "valid_flux": [],
        "valid_mfrac": [],
    }

    if verbose:
        print("Best validation loss: not available")
        print("Latest epoch: not started")

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_flux_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        valid_metrics = run_flux_epoch(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        for name in {
            "total",
            "flux",
            "mfrac",
        }:
            history[f"train_{name}"].append(train_metrics[name])
            history[f"valid_{name}"].append(valid_metrics[name])


        if verbose:
            # Move to the previous line and clear both displayed lines.
            print("\033[1A\033[2K", end="")
            print("\033[1A\033[2K", end="")

            print(
                f"Best epoch:\t{early_stopping.best_epoch} | "
                f"valid={early_stopping.best_loss:.5e}"
                # f"{early_stopping.epochs_without_improvement} epochs without improvement"
                # f"(epoch {early_stopping.best_epoch})"
            )

            print(
                f"Latest epoch:\t{epoch} | "
                f"valid={valid_metrics[monitor]:.5e} | "
                f"flux={valid_metrics['flux']:.5e} | "
                f"mfrac={valid_metrics['mfrac']:.5e} | "
                f"train={train_metrics[monitor]:.5e}"
            )

        # if verbose:
        #     print(
        #         f"Epoch {epoch:4d} | "
        #         f"train loss={train_metrics[monitor]:.6e} | "
        #         f"valid loss={valid_metrics[monitor]:.6e} | "
        #         # f"valid flux={valid_metrics['flux']:.6e} | "
        #         # f"valid mfrac={valid_metrics['mfrac']:.6e}"
        #     )

        should_stop = early_stopping.update(
            monitored_loss=valid_metrics[monitor],
            model=model,
            epoch=epoch,
        )

        if should_stop:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {early_stopping.best_epoch}. "
                    f"Best validation {monitor}: "
                    f"{early_stopping.best_loss:.6e}"
                )
            break

    early_stopping.restore_best_weights(
        model=model,
        device=device,
    )
    print("\033[?7h\033[0m")
    return history


# prediction function for full set of test data that has gradient tracking turned off
@torch.inference_mode()
def predict_flux_emulator(
    model,
    x_unscaled,
    scaler,
    device,
    batch_size=2000,
):
    model.eval()
    model.to(device)

    # Apply the input scaling fitted on training data.
    x_scaled = scaler.transform_x(x_unscaled)
    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)

    flux_predictions = []
    mfrac_predictions = []

    for start in range(0, x_tensor.shape[0], batch_size):

        x_batch = x_tensor[start:start + batch_size].to(device=device, dtype=torch.float32)
        prediction = model(x_batch)
        flux_predictions.append(prediction["flux"].cpu())

        if 'mfrac' in prediction:
            mfrac_predictions.append(prediction["mfrac"].cpu())

    flux_scaled = torch.cat(flux_predictions, dim=0).numpy()
    flux_physical = scaler.inverse_transform_flux(flux_scaled)

    result = {
        'flux_scaled': flux_scaled,
        'flux': flux_physical
    }

    if len(mfrac_predictions) > 0:
        mfrac_scaled = torch.cat(mfrac_predictions, dim=0).numpy()
        mfrac_physical = scaler.inverse_transform_mfrac(mfrac_scaled)
        result['mfrac_scaled'] = mfrac_scaled
        result['mfrac'] = mfrac_physical

    return result


# prediction function for single data entry
@torch.inference_mode()
def predict_flux_one(
    model,
    x,
    device,
    scaler,
    scaler_x_mean=None,
    scaler_x_std=None
):

    if scaler_x_mean is not None and scaler_x_std is not None:
        x_mean = np.asarray(scaler_x_mean, dtype=np.float32)
        x_std = np.asarray(scaler_x_std, dtype=np.float32)
    else:
        x_mean = scaler.x_mean.astype(np.float32)
        x_std = scaler.x_std.astype(np.float32)

    x_scaled = (np.asarray(x, dtype=np.float32) - x_mean) / x_std
    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=device).reshape(1, -1)
    prediction = model(x_tensor)
    flux_scaled = prediction["flux"][0].cpu().numpy()
    flux = scaler.inverse_transform_flux(flux_scaled)
    result = {"flux": flux}

    if "mfrac" in prediction:
        mfrac_scaled = (prediction["mfrac"][0].cpu().item())
        result["mfrac"] = (scaler.inverse_transform_mfrac(mfrac_scaled))

    return result



