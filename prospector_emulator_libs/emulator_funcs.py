import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import h5py




class EarlyStopping:
    def __init__(
        self,
        patience=50,
        abs_min_delta=0.0,
        rel_min_delta=1e-3,
    ):
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if abs_min_delta < 0.0 or rel_min_delta < 0.0:
            raise ValueError("abs_min_delta and rel_min_delta must be nonnegative.")

        self.patience = int(patience)
        self.abs_min_delta = float(abs_min_delta)
        self.rel_min_delta = float(rel_min_delta)
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
            required_improvement = max(self.abs_min_delta, self.rel_min_delta * abs(self.best_loss))
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


def make_scheduler(
    name,
    optimizer,
    **kwargs,
):
    if name is None:
        return None

    name = name.lower()

    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **kwargs,
        )

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            **kwargs,
        )

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **kwargs,
        )

    raise ValueError("Scheduler must be None or 'reduce_on_plateau', 'step', 'cosine'")



