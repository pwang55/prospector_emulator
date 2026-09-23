"""

Usage:
    $ python train_flux_emulator.py --config configs/train_flux_emulator.yaml

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib.cm import ScalarMappable
import yaml
from pathlib import Path
import argparse
import json
import libs.flux_emulator_libs as felibs
import time
from datetime import timedelta


# =============================================================================
# Configuration and command-line handling
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=None,
        metavar='<str>',
        help='Configuration YAML file',
        required=True,
    )
    parser.add_argument(
        # '-tr',
        '--train-data',
        type=str,
        default=None,
        metavar='<str>',
        help='train data containing input x, flux and mfrac'
    )
    parser.add_argument(
        # '-v',
        '--valid-data',
        type=str,
        default=None,
        metavar='<str>',
        help='valid data containing input x, flux and mfrac'
    )
    parser.add_argument(
        # '-vs',
        '--valid-split',
        type=float,
        default=None,
        metavar='<float>',
        help='valid/tran data split, omitted if valid_data is provided'
    )
    parser.add_argument(
        # '-te',
        '--test-data',
        type=str,
        default=None,
        metavar='<str>',
        help='test data containing input x, flux, mfrac and actual spectra'
    )
    parser.add_argument(
        # '-td',
        '--device',
        type=str,
        default=None,
        metavar='<str>',
        help="model training device, can be 'auto', 'mps', 'cuda', 'cpu'"
    )
    parser.add_argument(
        # '-td',
        '--flux-scaling',
        type=str,
        default=None,
        metavar='<str>',
        help="How to scale fluxes in training, can be 'none', 'standard', 'global_standard', 'log10_none', 'log10_standard', 'log10_global_standard'"
    )
    parser.add_argument(
        # '-td',
        '--mfrac-scaling',
        type=str,
        default=None,
        metavar='<str>',
        help="How to scale mfracs in training, can be 'none', 'standard'"
    )
    parser.add_argument(
        # '-td',
        '--train-batch-size',
        type=int,
        default=None,
        metavar='<int>',
        help='training dataset batch size'
    )
    parser.add_argument(
        # '-td',
        '--valid-batch-size',
        type=int,
        default=None,
        metavar='<int>',
        help='valid dataset batch size'
    )
    parser.add_argument(
        # '-td',
        '--train-num-workers',
        type=int,
        default=None,
        metavar='<int>',
        help='train and valid dataset number of workers'
    )
    # parser.add_argument(
    #     # '-td',
    #     '--train-persistent-workers',
    #     type=int,
    #     default=None,
    #     metavar='<int>',
    #     help='train and valid dataset number of persistent workers'
    # )
    parser.add_argument(
        # '-td',
        '--test-batch-size',
        type=int,
        default=None,
        metavar='<int>',
        help='test dataset batch size'
    )
    parser.add_argument(
        # '-td',
        '--test-num-workers',
        type=int,
        default=None,
        metavar='<int>',
        help='test dataset number of workers'
    )
    # parser.add_argument(
    #     # '-td',
    #     '--test-persistent-workers',
    #     type=int,
    #     default=None,
    #     metavar='<int>',
    #     help='test dataset number of persistent workers'
    # )
    parser.add_argument(
        # '-td',
        '--shared-dims',
        type=int,
        default=None,
        nargs="+",
        metavar="N",
        help='flux and mfrac shared hidden layer dimensions; \
              use consecutive integer with space in betweeh \
              to indicate neurons in each layer, ex: --shared_dims 256 256 256'
    )
    parser.add_argument(
        # '-td',
        '--flux-head-dims',
        type=int,
        default=None,
        nargs="+",
        metavar="N",
        help='flux shared hidden layer dimensions after shared_dim'
    )
    parser.add_argument(
        # '-td',
        '--mfrac-head-dims',
        type=int,
        default=None,
        nargs="+",
        metavar="N",
        help='mfrac shared hidden layer dimensions after shared_dim'
    )
    parser.add_argument(
        # '-td',
        '--activation',
        type=str,
        default=None,
        metavar='<str>',
        help="activation function, can be 'gelu', 'relu', 'silu', 'elu', 'tanh', 'leaky_relu'"
    )
    parser.add_argument(
        # '-td',
        '--dropout',
        type=float,
        default=None,
        metavar="<float>",
        help='dropout rate of the emulator neurons'
    )
    parser.add_argument(
        "--predict-mfrac",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--dont-predict-mfrac",
        dest="predict_mfrac",
        action="store_false",
        # default=None,
    )
    parser.add_argument(
        # '-td',
        '--flux-loss',
        type=str,
        default=None,
        metavar="<str>",
        help="mse, mae"
    )
    parser.add_argument(
        # '-td',
        '--flux-loss-space',
        type=str,
        default=None,
        metavar="<str>",
        help="scaled, log10, linear"
    )
    parser.add_argument(
        # '-td',
        '--mfrac-loss',
        type=str,
        default=None,
        metavar="<str>",
        help="mse, rmse, mae"
    )
    parser.add_argument(
        # '-td',
        '--mfrac-lambda',
        type=float,
        default=None,
        metavar="<float>",
        help="weight of mfrac_loss compared to flux_loss"
    )
    parser.add_argument(
        # '-td',
        '--optimizer',
        type=str,
        default=None,
        metavar="<str>",
        help="adam, adamw, sgd, rmsprop"
    )
    parser.add_argument(
        # '-td',
        '--learning-rate',
        type=float,
        default=None,
        metavar="<float>",
        help="learning rate of optimizer"
    )
    parser.add_argument(
        # '-td',
        '--weight-decay',
        type=float,
        default=None,
        metavar="<float>",
        help="regularization strength of optimizer"
    )
    parser.add_argument(
        # '-td',
        '--max-epochs',
        type=int,
        default=None,
        metavar="<int>",
        help="max allowed epochs"
    )
    parser.add_argument(
        # '-td',
        '--patience',
        type=int,
        default=None,
        metavar="<int>",
        help="how many epochs without improvement to stop early"
    )
    parser.add_argument(
        # '-td',
        '--abs-tol',
        type=float,
        default=None,
        metavar="<float>",
        help="absolute tolerance for improvement (new loss has to be smaller than previous loss - tol)"
    )
    parser.add_argument(
        # '-td',
        '--rel-tol',
        type=float,
        default=None,
        metavar="<float>",
        help="relative tolerance for improvement (new loss has to be smaller than previous_loss - tol * previous_loss)"
    )
    parser.add_argument(
        # '-td',
        '--monitor',
        type=str,
        default=None,
        metavar="<str>",
        help="which loss to monitor, can be 'total', 'flux', 'mfrac'"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
    )
    parser.add_argument(
        # '-td',
        '--outputs-dir',
        type=str,
        default=None,
        metavar="<str>",
        help="output files directory"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--dont-save-plots",
        dest="save_plots",
        action="store_false",
    )
    parser.add_argument(
        "--save-emulator",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--dont-save-emulator",
        dest="save_emulator",
        action="store_false",
    )
    parser.add_argument(
        # '-td',
        '--emulator-filename',
        type=str,
        default=None,
        metavar="<str>",
        help="output emulator filename"
    )
    return parser.parse_args()



def load_config(path, args):
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    # defines which args correspond to which config
    # args.keys: (config group, config key)
    override_dicts = {
        # config Data
        "train_data": ("Data", "train_data"),
        "valid_data": ("Data", "valid_data"),
        "valid_split": ("Data", "valid_split"),
        "test_data": ("Data", "test_data"),
        # config Device
        "device": "Device",
        # config Scaler
        "flux_scaling": ("Scaler", "flux_scaling"),
        "mfrac_scaling": ("Scaler", "mfrac_scaling"),
        "train_batch_size": ("Scaler", "train_batch_size"),
        "valid_batch_size": ("Scaler", "valid_batch_size"),
        "train_num_workers": ("Scaler", "train_num_workers"),
        # "train_persistent_workers": ("Scaler", "train_persistent_workers"),
        "test_batch_size": ("Scaler", "test_batch_size"),
        "test_num_workers": ("Scaler", "test_num_workers"),
        # "test_persistent_workers": ("Scaler", "test_persistent_workers"),
        # config Emulator
        "shared_dims": ("Emulator", "shared_dims"),
        "flux_head_dims": ("Emulator", "flux_head_dims"),
        "mfrac_head_dims": ("Emulator", "mfrac_head_dims"),
        "activation": ("Emulator", "activation"),
        "dropout": ("Emulator", "dropout"),
        "predict_mfrac": ("Emulator", "predict_mfrac"),
        # config Loss
        "flux_loss": ("Loss", "flux_loss"),
        "flux_loss_space": ("Loss", "flux_loss_space"),
        "mfrac_loss": ("Loss", "mfrac_loss"),
        "mfrac_lambda": ("Loss", "mfrac_lambda"),
        # config Optimizer
        "optimizer": ("Optimizer", "name"),
        "learning_rate": ("Optimizer", "learning_rate"),
        "weight_decay": ("Optimizer", "weight_decay"),
        # config Training
        "max_epochs": ("Training", "max_epochs"),
        "patience": ("Training", "patience"),
        "abs_tol": ("Training", "abs_tol"),
        "rel_tol": ("Training", "rel_tol"),
        "monitor": ("Training", "monitor"),
        "verbose": ("Training", "verbose"),
        # config Outputs
        "outputs_dir": ("Outputs", "outputs_dir"),
        "save_plots": ("Outputs", "save_plots"),
        "save_emulator": ("Outputs", "save_emulator"),
        "emulator_filename": ("Outputs", "emulator_filename"),

    }

    for arg_name, config_path in override_dicts.items():
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            try:
                section, key = config_path
                config[section][key] = arg_value
            except: # if it is not nested
                key = config_path
                config[key] = arg_value
    
    return config

def format_runtime(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"

    if seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m:{seconds:.1f}s"

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"{int(hours)}h:"
        f"{int(minutes)}m:"
        f"{seconds:.1f}s"
    )


# =============================================================================
# plotting functions
# =============================================================================

def rel_flux_err_plot(
    flux_pred,
    flux_true,
    lamb_obs=None,
    flux_scale='linear',
    yscale='linear',
    save=False,
    filename="rel_flux_err_plot.png",
    figsize=(10, 5),
    output_dirname="",
    dpi=300,
    fill_between_95_kwargs=None,
    fill_between_68_kwargs=None,
    plot_kwargs=None,
    # **kwargs,
    ):
    percentiles = [2.5, 16, 50, 84, 97.5]
    fill_between_95_kwargs = fill_between_95_kwargs or {}
    fill_between_68_kwargs = fill_between_68_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    fill_between_95_kwargs.setdefault("alpha", 0.2)
    fill_between_95_kwargs.setdefault("color", "tab:blue")
    fill_between_68_kwargs.setdefault("alpha", 0.3)
    fill_between_68_kwargs.setdefault("color", "tab:orange")

    plot_kwargs.setdefault("linestyle", "-")
    plot_kwargs.setdefault("marker", ".")
    plot_kwargs.setdefault("color", "tab:green")
    plot_kwargs.setdefault("linewidth", 0.5)
    plot_kwargs.setdefault("markersize", 3)
    plot_kwargs.setdefault("alpha", 0.8)

    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_lamb_obs = flux_pred.shape[1]
    if flux_scale == 'log10':
        flux_pred = np.log10(flux_pred)
        flux_true = np.log10(flux_true)
        y_label = r'$(log_{10}(f_{pred})-log_{10}(f_{true}))/log_{10}(f_{true})$'
    elif flux_scale == 'linear':
        y_label = r'$(f_{pred}-f_{true})/f_{true}$'
    rel_errs = (flux_pred - flux_true) / flux_true

    x_label = r"wavelength [$\mu m$]"
    if lamb_obs is None:
        lamb_obs = np.arange(n_lamb_obs)
        x_label = "datapoints"

    # errs_mean = np.mean(rel_errs, axis=0)
    # errs_std = np.std(rel_errs, axis=0)
    errs_percentiles = np.percentile(rel_errs, percentiles, axis=0)
    fig, ax = plt.subplots(figsize=figsize)
    # ax.errorbar(lamb_obs, errs_mean, errs_std, **kwargs)
    ax.fill_between(lamb_obs, errs_percentiles[0], errs_percentiles[-1], **fill_between_95_kwargs, label="95% percentile")
    ax.fill_between(lamb_obs, errs_percentiles[1], errs_percentiles[-2], **fill_between_68_kwargs, label="68% percentile")
    ax.plot(lamb_obs, errs_percentiles[2], **plot_kwargs, label="median")
    ax.legend()
    ax.grid(alpha=0.7)
    ax.set_title("Relative fiducial flux error")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yscale(yscale)
    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)


def mfrac_plots(
        mfrac_true,
        mfrac_pred,
        figsize=(10, 4),
        gridsize=100,
        extra_margin_ratio=0.005,
        bins='log',
        nbins_hist=200,
        save=False,
        filename="mfrac_plots.png",
        output_dirname="",
        dpi=300,
    ):
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    mfrac_range = np.max((np.max(mfrac_true)-np.min(mfrac_true), np.max(mfrac_pred)-np.min(mfrac_pred)))
    x_min = np.min((np.min(mfrac_true), np.min(mfrac_pred))) - extra_margin_ratio * mfrac_range
    x_max = np.max((np.max(mfrac_true), np.max(mfrac_pred))) + extra_margin_ratio * mfrac_range
    extent = [x_min, x_max, x_min, x_max]
    hb = ax[0].hexbin(mfrac_true, mfrac_pred, gridsize=gridsize, extent=extent, bins=bins)
    ax[0].plot([x_min, x_max], [x_min, x_max], linewidth=0.8, color='tab:red', alpha=0.6)
    fig.colorbar(hb, ax=ax[0], label='counts')
    ax[0].set_xlabel(r'$mfrac_{true}$')
    ax[0].set_ylabel(r'$mfrac_{pred}$')
    ax[0].grid(alpha=0.6)
    frac_diff = (mfrac_pred-mfrac_true)/mfrac_true
    ax[1].hist(frac_diff, bins=nbins_hist, alpha=0.8)
    ax[1].set_xlabel(r"($mfrac_{pred}$-$mfrac_{true}$)/$mfrac_{true}$")
    ax[1].grid(alpha=0.6)
    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)



def each_param_plots(
        x_test,
        flux_pred,
        flux_true,
        flux_scale='log10',
        feature_names=None,
        figsize=(15, 12),
        gridsize=100,
        ymin1=-0.02,
        ymax1=0.5,
        ymin2=0.5,
        ymax2=1.5,
        yscale='linear',
        bins="log",
        save=False,
        filename="param_plots.png",
        output_dirname="",
        dpi=300,
        **kwargs,
    ):
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = x_test.shape[1]
    if feature_names is None:
        feature_names = [f"param{i}" for i in range(n_features)]

    # TODO automatically decide dimension based on input number of templates
    # fig1, ax1 = plt.subplots(2, 5, figsize=figsize)

    if flux_scale == 'log10':
        flux_true = np.log10(flux_true)
        flux_pred = np.log10(flux_pred)
        y1_label = 'relative log10_flux L2 error'
        y2_label = 'median log10_flux ratio'
    elif flux_scale == 'linear':
        y1_label = 'relative flux L2 error'
        y2_label = 'median flux ratio'

    rel_l2_errs = np.linalg.norm(flux_pred-flux_true, axis=1)/np.linalg.norm(flux_true, axis=1)
    median_ratios = np.median(flux_pred/flux_true, axis=1)

    fig, ax = plt.subplots(4, 5, figsize=figsize)
    for i in range(x_test.shape[1]):
        rowi = i // 5
        coli = i % 5
        axi1 = ax[rowi, coli]
        xmin = np.min(x_test[:,i])
        xmax = np.max(x_test[:,i])
        hb1 = axi1.hexbin(x_test[:,i], rel_l2_errs, gridsize=gridsize, extent=[xmin, xmax, ymin1, ymax1], bins=bins, **kwargs)        
        fig.colorbar(hb1, ax=axi1)
        axi1.set_xlabel(feature_names[i])
        axi1.grid()
        axi1.set_yscale(yscale)
        if coli == 0:
            axi1.set_ylabel(y1_label)

        axi2 = ax[rowi+2, coli]
        hb2 = axi2.hexbin(x_test[:,i], median_ratios, gridsize=gridsize, extent=[xmin, xmax, ymin2, ymax2], bins=bins, **kwargs)
        fig.colorbar(hb2, ax=axi2)
        axi2.set_xlabel(feature_names[i])
        axi2.grid()
        axi2.set_yscale(yscale)
        if coli == 0:
            axi2.set_ylabel(y2_label)

    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)
        # plt.close()

# =============================================================================
# main
# =============================================================================



def main():
    args = parse_args()
    config = args.config

    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    # read config, then override with args if given CLI inputs
    config = load_config(args.config, args=args)

    device = felibs.get_device(config['Device'])
    print(f"Using device: {device}")

    verbose = config["Training"]["verbose"]

    # load data
    train_data = felibs.load_data(config["Data"]["train_data"])
    x_train = train_data["x"]
    flux_train = train_data["flux_fiducial"]
    mfrac_train = train_data["mfrac"]
    lamb_obs = train_data["lamb_obs"]

    train_param_keys = train_data['train_param_keys']
    default_params = train_data["default_params"]
    prior_dicts = train_data["prior_dicts"]

    if config["Data"]["valid_data"] is not None:
        valid_data = felibs.load_data(config["Data"]["valid_data"])
        x_valid = valid_data["x"]
        flux_valid = valid_data["flux_fiducial"]
        mfrac_valid = valid_data["mfrac"]
    else:
        # if valid_data is not provided, use data split
        rng = np.random.default_rng(seed=config["Data"]["valid_split_seed"])
        total_train_idx = np.arange(len(x_train))
        split_ratio = config["Data"]["valid_split"]
        idx_valid = rng.choice(total_train_idx, size=int(len(x_train)*split_ratio), replace=False)
        idx_train = np.setdiff1d(total_train_idx, idx_valid)
        x_valid = x_train[idx_valid]
        flux_valid = flux_train[idx_valid]
        mfrac_valid = mfrac_train[idx_valid]
        x_train = x_train[idx_train]
        flux_train = flux_train[idx_train]
        mfrac_train = mfrac_train[idx_train]

    test_data = felibs.load_data(config["Data"]["test_data"])
    x_test = test_data["x"]
    flux_test = test_data["flux_fiducial"]
    mfrac_test = test_data["mfrac"]

    # TEMP check negative or zero fluxes and interpolate them
    if verbose:
        print("check negative or zero fluxes...")
    for i in range(len(flux_train)):
        flux_i = flux_train[i]
        h = flux_i <= 0
        flux_i_interp = np.interp(lamb_obs[h], lamb_obs[~h], flux_i[~h])
        flux_i[h] = flux_i_interp
        flux_train[i] = flux_i

    for i in range(len(flux_valid)):
        flux_i = flux_valid[i]
        h = flux_i <= 0
        flux_i_interp = np.interp(lamb_obs[h], lamb_obs[~h], flux_i[~h])
        flux_i[h] = flux_i_interp
        flux_valid[i] = flux_i

    for i in range(len(flux_test)):
        flux_i = flux_test[i]
        h = flux_i <= 0
        flux_i_interp = np.interp(lamb_obs[h], lamb_obs[~h], flux_i[~h])
        flux_i[h] = flux_i_interp
        flux_test[i] = flux_i

    # first print out some settings on screen
    print(f"Train data:\t{config['Data']['train_data']}")
    if config["Data"]["valid_data"] is not None:
        print(f"valid data:\t{config['Data']['valid_data']}")
    else:
        print(f"valid/Train split: {config['Data']['valid_split']} (seed={config['Data']['valid_split_seed']})")
    print(f"Test data:\t{config['Data']['test_data']}")
    
    print(f"Scaling method: flux={config['Scaler']['flux_scaling']}, mfrac={config['Scaler']['mfrac_scaling']}") 
    # print(f"\tflux: ")
    # print(f"\tmfrac:")
    # print(f"Batch size: train={config["DataLoader"]["train_batch_size"]}, valid={config["DataLoader"]["valid_batch_size"]}")
    print("")
    print(f"Emulator::")
    print(f"\tshared dims: {config['Emulator']['shared_dims']}")
    print(f"\tflux head dims: {config['Emulator']['flux_head_dims']}")
    print(f"\tmfrac head dims: {config['Emulator']['mfrac_head_dims']}")
    print(f"\tactivation: {config['Emulator']['activation']}")
    print(f"\tpredict mfrac: {config['Emulator']['predict_mfrac']}")
    if config["Emulator"]["dropout"] != 0:
        print(f"\tdropout: {config['Emulator']['dropout']}")
    # print("")
    print(f"Loss:")
    print(f"\tflux loss: {config['Loss']['flux_loss']}")
    print(f"\tmfrac loss: {config['Loss']['mfrac_loss']}")
    print(f"\tmfrac lambda: {config['Loss']['mfrac_lambda']}")
    print(f"Optimizer: {config['Optimizer']['name']}, learning rate={config['Optimizer']['learning_rate']}, weight decay={config['Optimizer']['weight_decay']}")
    # print("")
    print(f"Training settings: ")
    print(f"\tmax_epochs={config['Training']['max_epochs']}, patience={config['Training']['patience']}, abs_tol={config['Training']['abs_tol']}, rel_tol={config['Training']['rel_tol']}")
    print(f"\ttrain batch size={config['DataLoader']['train_batch_size']}, valid batch size={config['DataLoader']['valid_batch_size']}, num worker={config['DataLoader']['train_num_workers']}")
    print("")


    # create Scaler
    scaler = felibs.FluxEmulatorScaler()
    (x_train_scaled, 
     flux_train_scaled, 
     mfrac_train_scaled
     ) = scaler.fit_transform(x_train, 
                              flux_train, 
                              mfrac_train, 
                              flux_scaling=config["Scaler"]["flux_scaling"],
                              mfrac_scaling=config["Scaler"]["mfrac_scaling"])
    (x_valid_scaled, 
     flux_valid_scaled, 
     mfrac_valid_scaled
    ) = scaler.transform(x_valid, 
                         flux_valid, 
                         mfrac_valid)
    (x_test_scaled, 
     flux_test_scaled, 
     mfrac_test_scaled
    ) = scaler.transform(x_test, 
                         flux_test, 
                         mfrac_test)

    x_train_scaled = x_train_scaled.astype(np.float32)
    flux_train_scaled = flux_train_scaled.astype(np.float32)
    mfrac_train_scaled = mfrac_train_scaled.astype(np.float32)

    x_valid_scaled = x_valid_scaled.astype(np.float32)
    flux_valid_scaled = flux_valid_scaled.astype(np.float32)
    mfrac_valid_scaled = mfrac_valid_scaled.astype(np.float32)

    x_test_scaled = x_test_scaled.astype(np.float32)
    flux_test_scaled = flux_test_scaled.astype(np.float32)
    mfrac_test_scaled = mfrac_test_scaled.astype(np.float32)

    # Create Dataset objects and DataLoaders
    train_dataset = felibs.FluxSPSDataset(x_train_scaled, flux_train_scaled, mfrac_train_scaled)
    valid_dataset = felibs.FluxSPSDataset(x_valid_scaled, flux_valid_scaled, mfrac_valid_scaled)
    test_dataset = felibs.FluxSPSDataset(x_test_scaled, flux_test_scaled, mfrac_test_scaled)

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["DataLoader"]["train_batch_size"],
        shuffle=config["DataLoader"]["train_shuffle"],
        num_workers=config["DataLoader"]["train_num_workers"],
        pin_memory=use_cuda,
        # persistent_workers=config["DataLoader"]["train_persistent_workers"],
        persistent_workers=(config["DataLoader"]["train_num_workers"]>0)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["DataLoader"]["valid_batch_size"],
        shuffle=config["DataLoader"]["valid_shuffle"],
        num_workers=config["DataLoader"]["train_num_workers"],
        pin_memory=use_cuda,
        # persistent_workers=config["DataLoader"]["train_persistent_workers"],
        persistent_workers=(config["DataLoader"]["train_num_workers"]>0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["DataLoader"]["test_batch_size"],
        shuffle=config["DataLoader"]["test_shuffle"],
        num_workers=config["DataLoader"]["test_num_workers"],
        pin_memory=use_cuda,
        # persistent_workers=config["DataLoader"]["test_persistent_workers"],
        persistent_workers=(config["DataLoader"]["test_num_workers"]>0),
    )

    # construct the emulator architecture with config settings
    shared_dims = tuple(config["Emulator"]["shared_dims"])
    if config["Emulator"]["flux_head_dims"] is not None:
        flux_head_dims = tuple(config["Emulator"]["flux_head_dims"])
    else:
        flux_head_dims = ()
    if config["Emulator"]["mfrac_head_dims"] is not None:
        mfrac_head_dims = tuple(config["Emulator"]["mfrac_head_dims"])
    else:
        mfrac_head_dims = ()

    n_features = x_train.shape[1]
    n_fluxes = flux_train.shape[1]

    # Create emulator model object with config architecture and send to device
    model = felibs.FluxSPSEmulator(
        n_features=n_features,
        n_fluxes=n_fluxes,
        shared_dims=shared_dims,
        flux_head_dims=flux_head_dims,
        mfrac_head_dims=mfrac_head_dims,
        activation=config["Emulator"]["activation"],
        dropout=config["Emulator"]["dropout"],
        predict_mfrac=config["Emulator"]["predict_mfrac"]
    ).to(device)

    # create criterion object from Loss class and send to device
    criterion = felibs.FluxEmulatorLoss(
        flux_loss=config["Loss"]["flux_loss"],
        flux_loss_space=config["Loss"]["flux_loss_space"],
        mfrac_loss=config["Loss"]["mfrac_loss"],
        mfrac_lambda=config["Loss"]["mfrac_lambda"],
        flux_scaling=scaler.flux_scaling,
        flux_scale_mean=scaler.flux_mean,
        flux_scale_std=scaler.flux_std,
        max_log10_flux=config["Loss"]["max_log10_flux"],
        wavelength_weights=config["Loss"]["wavelength_weights"]
    ).to(device)

    # create optimizer from optimizer class, AFTER creating model
    optimizer = felibs.make_optimizer(
        name=config["Optimizer"]["name"],
        parameters=model.parameters(),
        learning_rate=config["Optimizer"]["learning_rate"],
        weight_decay=config["Optimizer"]["weight_decay"]
    )

    train_start_time = time.perf_counter()
    # Actual training run
    history = felibs.fit_flux_emulator(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_epochs=config["Training"]["max_epochs"],
        patience=config["Training"]["patience"],
        abs_tol=config["Training"]["abs_tol"],
        rel_tol=config["Training"]["rel_tol"],
        monitor=config["Training"]["monitor"],
        verbose=verbose,
        )
    train_elapsed_time = time.perf_counter() - train_start_time
    if verbose:
        # print(f"Training time", timedelta(seconds=round(train_elapsed_time)))
        print(f"Training time: {format_runtime(train_elapsed_time)}")

    # prediction with test set
    predictions = felibs.predict_flux_emulator(
        model=model,
        x_unscaled=x_test,
        scaler=scaler,
        device=device,
        batch_size=config["DataLoader"]["test_batch_size"],
    )

    flux_pred = predictions["flux"]
    if 'mfrac' in predictions:
        mfrac_pred = predictions["mfrac"]
    else:
        mfrac_pred = np.zeros(mfrac_test.shape[0])

    # flux_pred = np.log10(flux_pred)
    # flux_test = np.log10(flux_test)
    # flux_rel_l2_errs = np.linalg.norm(flux_pred-flux_test, axis=1)/np.linalg.norm(flux_test, axis=1)
    # flux_median_ratios = np.median(flux_pred/flux_test, axis=1)


    test_metrics = felibs.run_flux_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )
    if verbose:
        print(f"test set loss={test_metrics[config['Training']['monitor']]}")
        # print(test_metrics)

    if config["Outputs"]["save_plots"]:
        # make plots
        rel_flux_err_plot(
            flux_pred=flux_pred,
            flux_true=flux_test,
            lamb_obs=lamb_obs,
            # flux_scale='log10',
            flux_scale='linear',
            yscale='linear',
            save=True,
            output_dirname=config["Outputs"]["outputs_dir"],
            filename="rel_flux_err_plot.png"
        )

        mfrac_plots(
            mfrac_test,
            mfrac_pred,
            save=True,
            output_dirname=config["Outputs"]["outputs_dir"],
            filename="mfrac_plots.png"
        )

        each_param_plots(
            x_test,
            flux_pred=flux_pred,
            flux_true=flux_test,
            # flux_scale='log10',
            flux_scale='linear',
            yscale='linear',
            feature_names=train_param_keys,
            save=True,
            output_dirname=config["Outputs"]["outputs_dir"],
            filename="param_plots.png"
        )


    # Saving emulator model and scaler, as well as other metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,

        "scaler_state": scaler.state_dict(),

        "lamb_obs": np.asarray(lamb_obs),

        "optimizer_name": optimizer.__class__.__name__,
        "optimizer_defaults": optimizer.defaults.copy(),
        "optimizer_state_dict": optimizer.state_dict(),

        "history": history,
        "loss_config": config["Loss"],
        "data_config": config["DataLoader"],

        "train_param_keys": train_param_keys,
        "default_params": default_params,
        "prior_dicts": prior_dicts
    }

    checkpoint_dir = Path(config["Outputs"]["outputs_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_dir / config["Outputs"]["emulator_filename"])
    if verbose:
        print(f"Trained emulator saved: {checkpoint_dir / config['Outputs']['emulator_filename']}")


    # time one iteration with the emulator
    model_cpu = model.to("cpu")
    model_cpu.eval()

    x_test_one = x_test[0]

    # Warm up before timing
    for _ in range(20):
        _ = felibs.predict_flux_one(
            model=model_cpu,
            x=x_test_one,
            device=torch.device("cpu"),
            scaler=scaler,
        )

    n_repeats = 1000
    start_time = time.perf_counter()
    for _ in range(n_repeats):
        _ = felibs.predict_flux_one(
            model=model_cpu,
            x=x_test_one,
            device=torch.device("cpu"),
            scaler=scaler,
        )
    elapsed_time = time.perf_counter() - start_time
    mean_seconds = elapsed_time / n_repeats

    if verbose:
        print(f"CPU single-object mean inference time: {mean_seconds*1e6:.4f} us (from {n_repeats} calls)")

if __name__ == '__main__':
    main()




