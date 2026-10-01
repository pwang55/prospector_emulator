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
import prospector_emulator_libs.emulator_funcs as efuncs

# input and output data scaler class
class EmulatorScaler:
    """
    Preprocessing for an SPS emulator.

    Input X:
        Standardize every feature independently.

    PCA coefficients:
        Compute one mean and one standard deviation from coef[:, 0],
        then apply that same scalar transformation to every coefficient.

    mfrac:
        Standardize independently.

    Fit this object on the training data only.
    """

    def __init__(self, eps=1e-12):
        self.eps = float(eps)
        self.x_mean = None
        self.x_std = None
        self.coef_mean = None
        self.coef_std = None
        self.mfrac_mean = None
        self.mfrac_std = None

    def fit(self, 
            x, 
            coef, 
            mfrac, 
            coef_scaling='none',
            mfrac_scaling='standard'):
        x = np.asarray(x, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        mfrac = np.asarray(mfrac, dtype=np.float64)

        valid_coef_scalings = {
            'none',
            'standard',
            'coef0',
        }
        valid_mfrac_scalings = {
            'none',
            'standard',
            'match_coef0'
        }
        if coef_scaling not in valid_coef_scalings:
            raise ValueError(f'coef_scaling must be one of {valid_coef_scalings}')
        if mfrac_scaling not in valid_mfrac_scalings:
            raise ValueError(f'mfrac_scaling must be one of {valid_mfrac_scalings}')
        if mfrac_scaling == 'match_coef0' and coef_scaling != 'none':
            raise ValueError("mfrac_scaling='match_coef0' can only be used when coef_scaling='none'")

        self.coef_scaling = coef_scaling
        self.mfrac_scaling = mfrac_scaling

        if x.ndim != 2:
            raise ValueError("x must have shape (ndata, nfeatures).")
        if coef.ndim != 2:
            raise ValueError("coef must have shape (ndata, ncoefs).")
        if mfrac.ndim != 1:
            raise ValueError("mfrac must have shape (ndata,).")

        ndata = x.shape[0]

        if coef.shape[0] != ndata:
            raise ValueError("x and coef must contain the same number of samples.")
        if mfrac.shape[0] != ndata:
            raise ValueError("x and mfrac must contain the same number of samples.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN or infinity.")
        if not np.all(np.isfinite(coef)):
            raise ValueError("coef contains NaN or infinity.")
        if not np.all(np.isfinite(mfrac)):
            raise ValueError("mfrac contains NaN or infinity.")

        # Input scaling, one mean/std per input feature
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)

        constant_features = self.x_std < self.eps

        if np.any(constant_features):
            indices = np.flatnonzero(constant_features)
            raise ValueError("The following input features have nearly zero "f"standard deviation: {indices.tolist()}")

        self.coef_mean = np.mean(coef, axis=0)
        self.coef_std = np.std(coef, axis=0)

        # One shared coefficient mean/std, derived from coef[:, 0]
        self.coef0_mean = self.coef_mean[0]
        self.coef0_std = self.coef_std[0]

        if self.coef0_std < self.eps:
            raise ValueError("The standard deviation of coef[:, 0] is too small.")

        # Separate mfrac scaling
        self.mfrac_mean = float(np.mean(mfrac))
        self.mfrac_std = float(np.std(mfrac))

        if self.mfrac_std < self.eps:
            raise ValueError("The standard deviation of mfrac is too small.")

        return self

    def transform_x(self, x):
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return (x - self.x_mean) / self.x_std

    def transform_coef(self, coef):
        self._check_fitted()
        coef = np.asarray(coef, dtype=np.float64)

        if self.coef_scaling == 'none':
            return coef
        
        elif self.coef_scaling == 'standard':
            return (coef - self.coef_mean) / self.coef_std
        
        elif self.coef_scaling == 'coef0':
            return (coef - self.coef0_mean) / self.coef0_std
        
    def transform_mfrac(self, mfrac):
        self._check_fitted()
        mfrac = np.asarray(mfrac, dtype=np.float64)

        if self.mfrac_scaling == 'none':
            return mfrac

        elif self.mfrac_scaling == 'standard':
            return (mfrac - self.mfrac_mean) / self.mfrac_std

        elif self.mfrac_scaling == 'match_coef0':
            return (mfrac - self.mfrac_mean) / self.mfrac_std * self.coef0_std + self.coef0_mean

    def transform(self, x, coef, mfrac):
        return (
            self.transform_x(x),
            self.transform_coef(coef),
            self.transform_mfrac(mfrac),
        )

    def fit_transform(self, 
                      x, 
                      coef, 
                      mfrac,
                      coef_scaling='none',
                      mfrac_scaling='standard'):
        self.fit(x, coef, mfrac, coef_scaling=coef_scaling, mfrac_scaling=mfrac_scaling)
        return self.transform(x, coef, mfrac)

    def inverse_transform_x(self, x_scaled):
        self._check_fitted()
        x_scaled = np.asarray(x_scaled, dtype=np.float64)

        return (x_scaled * self.x_std + self.x_mean)

    def inverse_transform_coef(self, coef_scaled):
        self._check_fitted()
        coef_scaled = np.asarray(coef_scaled, dtype=np.float64)

        if self.coef_scaling == 'none':
            return coef_scaled
        
        elif self.coef_scaling == 'standard':
            return coef_scaled * self.coef_std + self.coef_mean
        
        elif self.coef_scaling == 'coef0':
            return coef_scaled * self.coef_std + self.coef0_mean

    def inverse_transform_mfrac(self, mfrac_scaled):
        self._check_fitted()
        mfrac_scaled = np.asarray(mfrac_scaled, dtype=np.float64)

        if self.mfrac_scaling == 'none':
            return mfrac_scaled

        elif self.mfrac_scaling == 'standard':
            return mfrac_scaled * self.mfrac_std + self.mfrac_mean

        elif self.mfrac_scaling == 'match_coef0':
            return (mfrac_scaled - self.coef0_mean) / self.coef0_std * self.mfrac_std + self.mfrac_mean

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

            "coef_scaling": self.coef_scaling,
            "mfrac_scaling": self.mfrac_scaling,

            "x_mean": self.x_mean,
            "x_std": self.x_std,

            "coef_mean": self.coef_mean,
            "coef_std": self.coef_std,
            "coef0_mean": self.coef0_mean,
            "coef0_std": self.coef0_std,

            "mfrac_mean": self.mfrac_mean,
            "mfrac_std": self.mfrac_std,
        }

    @classmethod
    def from_state_dict(cls, state):
        """
        Reconstruct a fitted EmulatorScaler from a saved state dictionary.
        """
        scaler = cls(eps=state["eps"])

        scaler.coef_scaling = state["coef_scaling"]
        scaler.mfrac_scaling = state["mfrac_scaling"]

        scaler.x_mean = state["x_mean"]
        scaler.x_std = state["x_std"]

        scaler.coef_mean = state["coef_mean"]
        scaler.coef_std = state["coef_std"]
        scaler.coef0_mean = state["coef0_mean"]
        scaler.coef0_std = state["coef0_std"]

        scaler.mfrac_mean = state["mfrac_mean"]
        scaler.mfrac_std = state["mfrac_std"]

        return scaler



# Dataset class
class SPSDataset(Dataset):
    def __init__(
        self,
        x_scaled,
        coef_scaled,
        mfrac_scaled,
    ):
        self.x = torch.as_tensor(x_scaled, dtype=torch.float32)
        self.coef = torch.as_tensor(coef_scaled, dtype=torch.float32)
        self.mfrac = torch.as_tensor(mfrac_scaled, dtype=torch.float32)

        ndata = self.x.shape[0]

        if self.x.ndim != 2:
            raise ValueError("x must have shape (ndata, nfeatures).")

        if self.coef.ndim != 2:
            raise ValueError("coef must have shape (ndata, ncoefs).")

        if self.mfrac.ndim != 1:
            raise ValueError("mfrac must have shape (ndata,).")

        if self.coef.shape[0] != ndata:
            raise ValueError("x and c have different ndata")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        # return self.x[index], self.coef[index], self.mfrac[index]
        return {
            'x': self.x[index], 
            'coef': self.coef[index], 
            'mfrac': self.mfrac[index]
        }

# Class to create a full emulator architecture
class SPSEmulator(nn.Module):
    def __init__(
        self,
        n_features,
        n_coefs,
        shared_dims=(512, 512, 512, 512),
        coef_head_dims=(),
        mfrac_head_dims=(),
        activation="gelu",
        dropout=0.0,
        predict_mfrac=True
    ):
        super().__init__()

        self.n_features = int(n_features)
        self.n_coefs = int(n_coefs)
        self.predict_mfrac = predict_mfrac

        self.config = {
            "n_features": self.n_features,
            "n_coefs": self.n_coefs,
            "shared_dims": tuple(shared_dims),
            "coef_head_dims": tuple(coef_head_dims),
            "mfrac_head_dims": tuple(mfrac_head_dims),
            "activation": str(activation),
            "dropout": float(dropout),
            "predict_mfrac": bool(self.predict_mfrac)
        }

        self.shared, shared_output_dim = efuncs.make_hidden_mlp(
            input_dim=self.n_features,
            hidden_dims=shared_dims,
            activation=activation,
            dropout=dropout,
        )

        self.coef_head_hidden, coef_hidden_dim = (
            efuncs.make_hidden_mlp(
                input_dim=shared_output_dim,
                hidden_dims=coef_head_dims,
                activation=activation,
                dropout=dropout,
            )
        )

        self.coef_output = nn.Linear(coef_hidden_dim, self.n_coefs)

        if self.predict_mfrac:
            self.mfrac_head_hidden, mfrac_hidden_dim = (
                efuncs.make_hidden_mlp(
                    input_dim=shared_output_dim,
                    hidden_dims=mfrac_head_dims,
                    activation=activation,
                    dropout=dropout,
                )
            )
            self.mfrac_output = nn.Linear(mfrac_hidden_dim, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        coef_features = self.coef_head_hidden(shared_features)
        coef_pred = self.coef_output(coef_features)
        output = {'coef': coef_pred}

        if self.predict_mfrac:
            mfrac_features = self.mfrac_head_hidden(shared_features)
            mfrac_pred = self.mfrac_output(mfrac_features).squeeze(-1)
            output['mfrac'] = mfrac_pred

        return output

# Loss function class
class EmulatorLoss(nn.Module):
    """
    Loss for PCA coefficients and mfrac.

    Available coefficient losses
    -----------------------------
    "mse"
    "rmse"
    "mae"
    "logflux_mse"
    "logflux_mae"
    "flux_mse"
    "flux_mae"
    "relative_flux_mse"
    "relative_flux_mae"

    Available mfrac losses
    ----------------------
    "mse"
    "rmse"
    "mae"

    Total loss
    ----------
    total = coefficient_loss
          + mfrac_lambda * mfrac_loss
    """

    def __init__(
        self,
        coef_loss="mse",
        mfrac_loss="mse",
        mfrac_lambda=1.0,
        pca_modes=None,
        pca_mean=None,
        coef_scale_mean=0.0,
        coef_scale_std=1.0,
        relative_flux_eps=1e-8,
        rmse_eps=1e-12,
        max_log10_flux=None,
        wavelength_weights=None,
    ):
        super().__init__()

        self.coef_loss_name = coef_loss.lower()
        self.mfrac_loss_name = mfrac_loss.lower()
        self.mfrac_lambda = float(mfrac_lambda)
        self.relative_flux_eps = float(relative_flux_eps)
        self.max_log10_flux = max_log10_flux
        self.rmse_eps = rmse_eps

        valid_coef_losses = {
            "mse",
            "rmse",
            "mae",
            "l1",
            "logflux_mse",
            "logflux_mae",
            "flux_mse",
            "flux_mae",
            "relative_flux_mse",
            "relative_flux_mae",
        }

        valid_mfrac_losses = {
            "mse",
            "rmse",
            "mae",
            "l1",
        }

        if self.coef_loss_name not in valid_coef_losses:
            raise ValueError("Unknown coefficient loss f{self.coef_loss_name!r}. Available options: {valid_coef_losses}")

        if self.mfrac_loss_name not in valid_mfrac_losses:
            raise ValueError("Unknown mfrac loss {self.mfrac_loss_name!r}. Available options: {valid_mfrac_losses}")

        if self.mfrac_lambda < 0.0:
            raise ValueError("mfrac_lambda must be nonnegative.")

        if self.relative_flux_eps <= 0.0:
            raise ValueError("relative_flux_eps must be positive.")

        # These are nontrainable tensors that should move
        # automatically when criterion.to(device) is called.
        self.register_buffer(
            "coef_scale_mean",
            torch.as_tensor(
                coef_scale_mean,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "coef_scale_std",
            torch.as_tensor(
                coef_scale_std,
                dtype=torch.float32,
            ),
        )

        if pca_modes is None:
            self.pca_modes = None
        else:
            self.register_buffer(
                "pca_modes",
                torch.as_tensor(
                    pca_modes,
                    dtype=torch.float32,
                ),
            )

        if pca_mean is None:
            self.pca_mean = None
        else:
            self.register_buffer(
                "pca_mean",
                torch.as_tensor(
                    pca_mean,
                    dtype=torch.float32,
                ),
            )

        if wavelength_weights is None:
            self.wavelength_weights = None
        else:
            wavelength_weights = torch.as_tensor(
                wavelength_weights,
                dtype=torch.float32,
            )

            if wavelength_weights.ndim != 1:
                raise ValueError(
                    "wavelength_weights must have shape "
                    "(nwavelength,)."
                )

            self.register_buffer(
                "wavelength_weights",
                wavelength_weights,
            )

        requires_pca = self.coef_loss_name not in {
            "mse",
            "mae",
            "l1",
        }

        if requires_pca:
            if self.pca_modes is None:
                raise ValueError(
                    "pca_modes is required for flux-space or log-flux-space losses.")

            if self.pca_mean is None:
                raise ValueError(
                    "pca_mean is required for flux-space or log-flux-space losses.")

            if self.pca_modes.ndim != 2:
                raise ValueError(
                    "pca_modes must have shape (ncoefs, nwavelength).")

            if self.pca_mean.ndim != 1:
                raise ValueError(
                    "pca_mean must have shape (nwavelength,).")

            if (self.pca_modes.shape[1] != self.pca_mean.shape[0]):
                raise ValueError(
                    "The wavelength dimensions of pca_modes and pca_mean do not match.")

            if self.wavelength_weights is not None:
                if (self.wavelength_weights.shape[0] != self.pca_mean.shape[0]):
                    raise ValueError(
                        "wavelength_weights and pca_mean have different lengths.")

    def inverse_scale_coef(self, coef_scaled):
        """
        Convert network-space coefficients back into the
        original PCA coefficient units.
        """
        return (coef_scaled * self.coef_scale_std + self.coef_scale_mean)

    def reconstruct_log10_flux(self, coef_scaled):
        """
        Reconstruct base-10 log flux from scaled coefficients.
        """
        coef_physical = self.inverse_scale_coef(coef_scaled)

        return (coef_physical @ self.pca_modes + self.pca_mean)

    def log10_to_linear_flux(self, log10_flux):
        """
        Convert base-10 log flux to linear flux.

        max_log10_flux can optionally prevent numerical overflow,
        but clipping also causes zero gradient beyond the chosen
        clipping threshold.
        """
        if self.max_log10_flux is not None:
            log10_flux = torch.clamp(log10_flux, max=float(self.max_log10_flux))

        return torch.pow(10.0, log10_flux)

    def reduce_residual(self, residual, kind):
        """
        Apply MAE or MSE, with optional wavelength weighting.

        For unweighted losses this returns the mean over the
        batch and wavelength dimensions.

        For weighted losses, weights are normalized so their
        mean is approximately one.
        """
        if kind in {"mae", "l1"}:
            element_loss = torch.abs(residual)

        elif kind == "mse":
            element_loss = residual.square()

        else:
            raise ValueError("Reduction kind must be 'mae' or 'mse'.")

        if self.wavelength_weights is None:
            return torch.mean(element_loss)

        weights = self.wavelength_weights
        weights = weights / torch.mean(weights)

        return torch.mean(element_loss * weights.unsqueeze(0))

    def coefficient_loss(
        self,
        coef_pred_scaled,
        coef_true_scaled,
    ):
        loss_name = self.coef_loss_name

        # ----------------------------------------------
        # Direct scaled-coefficient losses
        # ----------------------------------------------

        if loss_name == "mse":
            residual = coef_pred_scaled - coef_true_scaled
            return torch.mean(residual.square())

        if loss_name == "rmse":
            residual = coef_pred_scaled - coef_true_scaled
            return torch.sqrt(torch.mean(residual.square())) + self.rmse_eps

        if loss_name in {"mae", "l1"}:
            residual = coef_pred_scaled - coef_true_scaled
            return torch.mean(torch.abs(residual))

        # ----------------------------------------------
        # Reconstruct log10 flux
        # ----------------------------------------------

        log10_flux_pred = self.reconstruct_log10_flux(coef_pred_scaled)
        log10_flux_true = self.reconstruct_log10_flux(coef_true_scaled)
        log10_flux_residual = log10_flux_pred - log10_flux_true

        # ----------------------------------------------
        # Log-flux losses
        # ----------------------------------------------

        if loss_name == "logflux_mse":
            return self.reduce_residual(log10_flux_residual, kind="mse")

        if loss_name == "logflux_mae":
            return self.reduce_residual(log10_flux_residual, kind="mae")

        # ----------------------------------------------
        # Convert reconstructed spectra to linear flux
        # ----------------------------------------------

        flux_pred = self.log10_to_linear_flux(log10_flux_pred)
        flux_true = self.log10_to_linear_flux(log10_flux_true)
        flux_residual = flux_pred - flux_true

        # ----------------------------------------------
        # Absolute linear-flux losses
        # ----------------------------------------------

        if loss_name == "flux_mse":
            return self.reduce_residual(flux_residual, kind="mse")

        if loss_name == "flux_mae":
            return self.reduce_residual(flux_residual, kind="mae")

        # ----------------------------------------------
        # Relative linear-flux losses
        # ----------------------------------------------

        relative_residual = (flux_residual / (torch.abs(flux_true) + self.relative_flux_eps))

        if loss_name == "relative_flux_mse":
            return self.reduce_residual(relative_residual, kind="mse")

        if loss_name == "relative_flux_mae":
            return self.reduce_residual(relative_residual, kind="mae")

        raise RuntimeError("Unreachable coefficient-loss branch.")

    def calculate_mfrac_loss(
        self,
        mfrac_pred_scaled,
        mfrac_true_scaled,
    ):
        residual = mfrac_pred_scaled - mfrac_true_scaled

        if self.mfrac_loss_name == "mse":
            return torch.mean(residual.square())

        if self.mfrac_loss_name == "rmse":
            return torch.sqrt(torch.mean(residual.square())) + self.rmse_eps

        if self.mfrac_loss_name in {"mae", "l1"}:
            return torch.mean(torch.abs(residual))

        raise RuntimeError("Unreachable mfrac-loss branch.")

    def forward(
        self,
        prediction,
        coef_true_scaled,
        mfrac_true_scaled=None,
    ):
        loss_coef = self.coefficient_loss(prediction["coef"], coef_true_scaled)

        if 'mfrac' not in prediction:
            loss_mfrac = loss_coef.new_zeros(())
            return {
                "total": loss_coef,
                "coef": loss_coef,
                "mfrac": loss_mfrac,
            }

        if mfrac_true_scaled is None:
            raise ValueError('mfrac_true_scaled is required when the model predicts mfrac')
        
        loss_mfrac = self.calculate_mfrac_loss(prediction["mfrac"], mfrac_true_scaled)
        loss_total = (loss_coef + self.mfrac_lambda * loss_mfrac)

        return {
            "total": loss_total,
            "coef": loss_coef,
            "mfrac": loss_mfrac,
        }


# class to load pre-trained emulator from .pt file and can be used for prediction
class LoadedSPSEmulator:
    def __init__(
        self,
        checkpoint_path,
        device="cpu",
    ):
        # self.device = torch.device(device)
        self.device = efuncs.get_device(device)

        # Load checkpoint once
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        # Reconstruct model
        self.model = SPSEmulator(**self.checkpoint["model_config"])
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Reconstruct fitted scaler
        self.scaler = EmulatorScaler.from_state_dict(self.checkpoint["scaler_state"])

        # PCA reconstruction arrays
        self.pca_lbs = np.asarray(self.checkpoint["pca_lbs"])
        self.lbs = self.pca_lbs
        self.pca_modes = np.asarray(self.checkpoint["pca_modes"])
        self.pca_mean = np.asarray(self.checkpoint["pca_mean"])

        self.train_param_keys = self.checkpoint["train_param_keys"]
        self.default_params = self.checkpoint["default_params"]
        self.prior_dicts = self.checkpoint["prior_dicts"]
        self._setup_scaler_tensors()
        self._setup_pca_tensors()

    def _setup_scaler_tensors(self):
        """
        Cache scaler constants as torch tensors on self.device.

        These are used only by torch-native inference. The original
        NumPy scaler is left unchanged for normal predict()/predict_one().
        """

        model_dtype = next(self.model.parameters()).dtype
        scaler = self.scaler

        # -------------------------------------------------------------
        # x scaling
        # -------------------------------------------------------------

        self._x_mean_t = torch.as_tensor(scaler.x_mean, dtype=model_dtype, device=self.device)
        self._x_std_t = torch.as_tensor(scaler.x_std, dtype=model_dtype, device=self.device)

        # -------------------------------------------------------------
        # coef scaling
        # -------------------------------------------------------------

        self._coef_mean_t = torch.as_tensor(self.scaler.coef_mean, dtype=model_dtype, device=self.device)
        self._coef_std_t = torch.as_tensor(self.scaler.coef_std, dtype=model_dtype, device=self.device)

        self._coef0_mean_t = torch.as_tensor(self.scaler.coef0_mean, dtype=model_dtype, device=self.device)
        self._coef0_std_t = torch.as_tensor(self.scaler.coef0_std, dtype=model_dtype, device=self.device)

        # -------------------------------------------------------------
        # mfrac scaling
        # -------------------------------------------------------------

        if scaler.mfrac_mean is None:
            self._mfrac_mean_t = None
        else:
            self._mfrac_mean_t = torch.as_tensor(scaler.mfrac_mean, dtype=model_dtype, device=self.device)

        if scaler.mfrac_std is None:
            self._mfrac_std_t = None
        else:
            self._mfrac_std_t = torch.as_tensor(scaler.mfrac_std, dtype=model_dtype, device=self.device)

    def _transform_x_torch(self, x):
        return (x - self._x_mean_t) / self._x_std_t
    
    def _inverse_transform_coef_torch(self, coef_scaled):
        scaling = self.scaler.coef_scaling

        if scaling == "none":
            return coef_scaled

        elif scaling == "standard":
            return (coef_scaled * self._coef_std_t + self._coef_mean_t)

        elif scaling == "coef0":
            return (coef_scaled * self._coef0_std_t + self._coef0_mean_t)

        raise ValueError(f"Unknown coef scaling: {scaling!r}")
        

    def _inverse_transform_mfrac_torch(self, mfrac_scaled):
        scaling = self.scaler.mfrac_scaling

        if scaling == "none":
            return mfrac_scaled
        elif scaling == "standard":
            return (mfrac_scaled * self._mfrac_std_t + self._mfrac_mean_t)
        elif scaling == "match_coef0":
            return ((mfrac_scaled - self._coef0_mean_t) / self._coef0_std_t * self._mfrac_std_t + self._mfrac_mean_t)

        raise ValueError(f"Unknown mfrac scaling: {scaling!r}")

    # single object inference method
    @torch.inference_mode()
    def predict_one(
        self,
        x,
        return_spectrum=True,
        return_linear_flux=True,
    ):
        x_scaled = self.scaler.transform_x(x)
        model_dtype = next(self.model.parameters()).dtype

        x_tensor = torch.as_tensor(
            x_scaled,
            dtype=model_dtype,
            device=self.device,
        ).reshape(1, -1)

        prediction = self.model(x_tensor)
        coef_scaled = (prediction["coef"][0].cpu().numpy())
        coef = self.scaler.inverse_transform_coef(coef_scaled)

        result = {
            "coef_scaled": coef_scaled,
            "coef": coef,
        }

        if "mfrac" in prediction:
            mfrac_scaled = (prediction["mfrac"][0].cpu().item())
            result["mfrac_scaled"] = mfrac_scaled
            result["mfrac"] = (self.scaler.inverse_transform_mfrac(mfrac_scaled))

        if return_spectrum:
            log10_flux = (coef @ self.pca_modes + self.pca_mean)
            result["log10_flux"] = log10_flux
            # result["lbs"] = self.pca_lbs

            if return_linear_flux:
                result["flux"] = (10.0 ** log10_flux)

        return result

    # multiple object inference method
    @torch.inference_mode()
    def predict(
        self,
        x,
        batch_size=2000,
        return_spectrum=True,
        return_linear_flux=True,
    ):
        x = np.asarray(x)

        # Redirect one-dimensional input to predict_one().
        if x.ndim == 1:
            return self.predict_one(
                x=x,
                return_spectrum=return_spectrum,
                return_linear_flux=return_linear_flux,
            )

        if x.ndim != 2:
            raise ValueError(
                "x must have shape (n_features,) or "
                "(n_objects, n_features)."
            )

        x_scaled = self.scaler.transform_x(x)
        model_dtype = next(self.model.parameters()).dtype

        x_tensor = torch.as_tensor(x_scaled, dtype=model_dtype)
        coef_batches = []
        mfrac_batches = []

        for start in range(0, x_tensor.shape[0], batch_size):
            x_batch = x_tensor[start:start + batch_size].to(self.device)
            prediction = self.model(x_batch)
            coef_batches.append(prediction["coef"].cpu())

            if "mfrac" in prediction:
                mfrac_batches.append(prediction["mfrac"].cpu())

        coef_scaled = torch.cat(coef_batches, dim=0,).numpy()
        coef = self.scaler.inverse_transform_coef(coef_scaled)

        result = {
            # "coef_scaled": coef_scaled,
            "coef": coef,
        }

        if mfrac_batches:
            mfrac_scaled = torch.cat(mfrac_batches, dim=0).numpy()
            # result["mfrac_scaled"] = mfrac_scaled
            result["mfrac"] = (self.scaler.inverse_transform_mfrac(mfrac_scaled))

        if return_spectrum:
            log10_flux = (coef @ self.pca_modes + self.pca_mean)
            result["log10_flux"] = log10_flux
            # result["lbs"] = self.pca_lbs

            if return_linear_flux:
                result["flux"] = (10.0 ** log10_flux)

        return result

    @torch.inference_mode()
    def predict_torch(self, x, return_spectrum=True, return_linear_flux=True):
        """
        Batched inference that stays entirely in Torch.

        Parameters
        ----------
        x : array-like or torch.Tensor
            Raw, unscaled emulator parameters.

            Shape:
                (n_features,)
                or
                (n_objects, n_features)

        Returns
        -------
        dict
            {
                "flux": physical linear flux tensor,
                "mfrac": physical mfrac tensor,
            }

            Returned tensors remain on self.device.
        """

        model_dtype = next(self.model.parameters()).dtype

        x = torch.as_tensor(x, dtype=model_dtype, device=self.device)
        single_object = (x.ndim == 1)

        if single_object:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise ValueError("x must have shape (n_features,) or (n_objects, n_features).")

        # -------------------------------------------------------------
        # Scale raw parameters
        # -------------------------------------------------------------

        x_scaled = self._transform_x_torch(x)

        # -------------------------------------------------------------
        # Neural network
        # -------------------------------------------------------------

        prediction_scaled = self.model(x_scaled)

        # -------------------------------------------------------------
        # Inverse-transform PCA coefficients
        # -------------------------------------------------------------

        coef = self._inverse_transform_coef_torch(prediction_scaled["coef"])

        # -------------------------------------------------------------
        # Build result
        # -------------------------------------------------------------

        result = {
            # "flux": flux,
            "coef": coef,
            }

        if return_spectrum:
            log10_flux = (coef @ self._pca_modes_t + self._pca_mean_t)
            result["log10_flux"] = log10_flux
            # result["lbs"] = self.pca_lbs
            if return_linear_flux:
                flux = torch.pow(10.0, log10_flux)
                result["flux"] = flux

        if "mfrac" in prediction_scaled:
            mfrac = self._inverse_transform_mfrac_torch(prediction_scaled["mfrac"])
            if mfrac.ndim == 2 and mfrac.shape[1] == 1:
                mfrac = mfrac[:, 0]
            result["mfrac"] = mfrac

        # -------------------------------------------------------------
        # Optional single-object behavior
        # -------------------------------------------------------------

        if single_object:
            for key in ("coef", "log10_flux", "flux", "mfrac"):
                if key in result:
                    result[key] = result[key][0]

        return result

    def _setup_pca_tensors(self):
        model_dtype = next(self.model.parameters()).dtype

        self._pca_modes_t = torch.as_tensor(
            self.pca_modes,
            dtype=model_dtype,
            device=self.device,
        )

        self._pca_mean_t = torch.as_tensor(
            self.pca_mean,
            dtype=model_dtype,
            device=self.device,
        )

    def to(self, device):
        # self.device = torch.device(device)
        self.device = efuncs.get_device(device)
        self.model.to(self.device)
        self.model.eval()
        self._setup_scaler_tensors()
        self._setup_pca_tensors()
        return self



# =============================================================================
# Functions
# =============================================================================

# one full epoch run function that makes data go through MLP and returns loss and gradient if training
def run_epoch(
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
        "coef": 0.0,
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

            coef_true = batch["coef"].to(
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
                coef_true,
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
def fit_emulator(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    max_epochs=500,
    patience=20,
    abs_min_delta=0.0,
    rel_min_delta=1e-5,
    monitor="total",
    verbose=True,
):
    if monitor not in {
        "total",
        "coef",
        "mfrac",
    }:
        raise ValueError("monitor must be 'total', 'coef', or 'mfrac'.")

    # Ensure both model and criterion are on the right device.
    model.to(device)
    criterion.to(device)

    early_stopping = efuncs.EarlyStopping(
        patience=patience,
        abs_min_delta=abs_min_delta,
        rel_min_delta=rel_min_delta
    )

    history = {
        "train_total": [],
        "train_coef": [],
        "train_mfrac": [],
        "valid_total": [],
        "valid_coef": [],
        "valid_mfrac": [],
        "learning_rate": [],
    }

    if verbose:
        print("Best validation loss: not available")
        print("Latest epoch: not started")

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        valid_metrics = run_epoch(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        if scheduler is not None:
            scheduler.step(valid_metrics[monitor])

        for name in {
            "total",
            "coef",
            "mfrac",
        }:
            history[f"train_{name}"].append(train_metrics[name])
            history[f"valid_{name}"].append(valid_metrics[name])
            current_lr = optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

        if verbose:
            # Move to the previous line and clear both displayed lines.
            print("\033[1A\033[2K", end="")
            print("\033[1A\033[2K", end="")

            print(
                f"Best epoch:\t{early_stopping.best_epoch} | "
                f"valid={early_stopping.best_loss:.4e}"
                # f"lr={current_lr}"
                # f"{early_stopping.epochs_without_improvement} epochs without improvement"
                # f"(epoch {early_stopping.best_epoch})"
            )

            print(
                f"Latest epoch:\t{epoch} | "
                f"valid={valid_metrics[monitor]:.4e} | "
                f"coef={valid_metrics['coef']:.4e} | "
                f"mfrac={valid_metrics['mfrac']:.4e} | "
                f"train={train_metrics[monitor]:.4e} | "
                f"lr={current_lr:.2e}"
            )

        # if verbose:
        #     print(
        #         f"Epoch {epoch:4d} | "
        #         f"train loss={train_metrics[monitor]:.6e} | "
        #         f"valid loss={valid_metrics[monitor]:.6e} | "
        #         # f"valid coef={valid_metrics['coef']:.6e} | "
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
def predict_emulator(
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

    coef_predictions = []
    mfrac_predictions = []

    for start in range(0, x_tensor.shape[0], batch_size):

        x_batch = x_tensor[start:start + batch_size].to(device=device, dtype=torch.float32)
        prediction = model(x_batch)
        coef_predictions.append(prediction["coef"].cpu())

        if 'mfrac' in prediction:
            mfrac_predictions.append(prediction["mfrac"].cpu())

    coef_scaled = torch.cat(coef_predictions, dim=0).numpy()
    coef_physical = scaler.inverse_transform_coef(coef_scaled)

    result = {
        'coef_scaled': coef_scaled,
        'coef': coef_physical
    }

    if len(mfrac_predictions) > 0:
        mfrac_scaled = torch.cat(mfrac_predictions, dim=0).numpy()
        mfrac_physical = scaler.inverse_transform_mfrac(mfrac_scaled)
        result['mfrac_scaled'] = mfrac_scaled
        result['mfrac'] = mfrac_physical

    return result

# prediction function for single data entry
@torch.inference_mode()
def predict_one(
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
    coef_scaled = prediction["coef"][0].cpu().numpy()
    coef = scaler.inverse_transform_coef(coef_scaled)
    result = {"coef": coef}

    if "mfrac" in prediction:
        mfrac_scaled = (prediction["mfrac"][0].cpu().item())
        result["mfrac"] = (scaler.inverse_transform_mfrac(mfrac_scaled))

    return result

