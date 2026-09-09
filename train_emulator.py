"""

Usage:
    $ python train_emulator.py --config training.yaml

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
import libs.emulator_libs as libs
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
        help='training data containing input x, coef and mfrac'
    )
    parser.add_argument(
        # '-v',
        '--validation-data',
        type=str,
        default=None,
        metavar='<str>',
        help='validation data containing input x, coef and mfrac'
    )
    parser.add_argument(
        # '-vs',
        '--validation-split',
        type=float,
        default=None,
        metavar='<float>',
        help='validation/tran data split, omitted if validation_data is provided'
    )
    parser.add_argument(
        # '-te',
        '--test-data',
        type=str,
        default=None,
        metavar='<str>',
        help='test data containing input x, coef, mfrac and actual spectra'
    )
    parser.add_argument(
        # '-td',
        '--pca-file',
        type=str,
        default=None,
        metavar='<str>',
        help='file containing PCA wavelength grid lbs, modes and mean'
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
        '--coef-scaling',
        type=str,
        default=None,
        metavar='<str>',
        help="How to scale coefs in training, can be 'none', 'standard', 'coef0'"
    )
    parser.add_argument(
        # '-td',
        '--mfrac-scaling',
        type=str,
        default=None,
        metavar='<str>',
        help="How to scale mfracs in training, can be 'none', 'standard', 'match_coef0'"
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
        '--validation-batch-size',
        type=int,
        default=None,
        metavar='<int>',
        help='validation dataset batch size'
    )
    parser.add_argument(
        # '-td',
        '--train-num-workers',
        type=int,
        default=None,
        metavar='<int>',
        help='train and validation dataset number of workers'
    )
    # parser.add_argument(
    #     # '-td',
    #     '--train-persistent-workers',
    #     type=int,
    #     default=None,
    #     metavar='<int>',
    #     help='train and validation dataset number of persistent workers'
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
        help='coef and mfrac shared hidden layer dimensions; \
              use consecutive integer with space in betweeh \
              to indicate neurons in each layer, ex: --shared_dims 256 256 256'
    )
    parser.add_argument(
        # '-td',
        '--coef-head-dims',
        type=int,
        default=None,
        nargs="+",
        metavar="N",
        help='coef shared hidden layer dimensions after shared_dim'
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
        '--coef-loss',
        type=str,
        default=None,
        metavar="<str>",
        help="mse, rmse, mae, logflux_mse, logflux_mae, flux_mse, flux_mae, relative_flux_mse, relative_flux_mae"
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
        help="weight of mfrac_loss compared to coef_loss"
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
        help="which loss to monitor, can be 'total', 'coef', 'mfrac'"
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
        "validation_data": ("Data", "validation_data"),
        "validation_split": ("Data", "validation_split"),
        "test_data": ("Data", "test_data"),
        "pca_file": ("Data", "pca_file"),
        # config Device
        "device": "Device",
        # config Scaler
        "coef_scaling": ("Scaler", "coef_scaling"),
        "mfrac_scaling": ("Scaler", "mfrac_scaling"),
        "train_batch_size": ("Scaler", "train_batch_size"),
        "validation_batch_size": ("Scaler", "validation_batch_size"),
        "train_num_workers": ("Scaler", "train_num_workers"),
        # "train_persistent_workers": ("Scaler", "train_persistent_workers"),
        "test_batch_size": ("Scaler", "test_batch_size"),
        "test_num_workers": ("Scaler", "test_num_workers"),
        # "test_persistent_workers": ("Scaler", "test_persistent_workers"),
        # config Emulator
        "shared_dims": ("Emulator", "shared_dims"),
        "coef_head_dims": ("Emulator", "coef_head_dims"),
        "mfrac_head_dims": ("Emulator", "mfrac_head_dims"),
        "activation": ("Emulator", "activation"),
        "dropout": ("Emulator", "dropout"),
        "predict_mfrac": ("Emulator", "predict_mfrac"),
        # config Loss
        "coef_loss": ("Loss", "coef_loss"),
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

# def parse_value(text):
#     """Parse CLI values using YAML syntax: true, 3, 1e-3, [512, 256], etc."""
#     return yaml.safe_load(text)

# def set_nested(config, dotted_key, value):
#     """Set a nested YAML entry such as model.activation=relu."""
#     keys = dotted_key.split(".")
#     current = config

#     for key in keys[:-1]:
#         if key not in current or not isinstance(current[key], dict):
#             current[key] = {}
#         current = current[key]

#     current[keys[-1]] = value

# def load_config(path, overrides):
#     with open(path, "r", encoding="utf-8") as file:
#         config = yaml.safe_load(file) or {}

#     for override in overrides:
#         if "=" not in override:
#             raise ValueError(f"Override must be KEY=VALUE, received: {override!r}")
#         key, value = override.split("=", 1)
#         set_nested(config, key, parse_value(value))

#     return config



# =============================================================================
# plotting functions
# =============================================================================

def rel_err_ratio_plots(
        x, 
        flux_rel_l2_err, 
        coef_rel_l2_err, 
        flux_median_ratio, 
        coef0_ratio, 
        save=False,
        filename="rel_err_ratio.png",
        output_dirname="",
        dpi=300,
        x_min=-0.02,
        x_max=3.02,
        x_label="z",
        l2_err_min=-0.05,
        l2_err_max=0.6,
        ratio_min=0.5,
        ratio_max=1.5,
        bins='log',
        gridsize=100,
        markersize=3,
        markeralpha=0.3,
        figsize=(8, 6),
        style='hexbin',
        **kwargs):
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    if style == "hexbin":
        hb00 = ax[0,0].hexbin(x, flux_rel_l2_err, gridsize=gridsize, extent=[x_min, x_max, l2_err_min, l2_err_max], bins=bins, **kwargs)
        fig.colorbar(hb00, ax=ax[0,0])
    elif style == "scatter":
        ax[0,0].scatter(x, flux_rel_l2_err, s=markersize, alpha=markeralpha, **kwargs)
    ax[0,0].set_xlabel(x_label)
    ax[0,0].set_ylim(l2_err_min, l2_err_max)
    ax[0,0].grid(alpha=0.3)
    ax[0,0].set_title('flux relative L2 error')

    if style == "hexbin":
        hb01 = ax[0,1].hexbin(x, coef_rel_l2_err, gridsize=gridsize, extent=[x_min, x_max, l2_err_min, l2_err_max], bins=bins, **kwargs)
        fig.colorbar(hb01, ax=ax[0,1])
    elif style == "scatter":
        ax[0,1].scatter(x, coef_rel_l2_err, s=markersize, alpha=markeralpha, **kwargs)
    ax[0,1].set_xlabel(x_label)
    ax[0,1].set_ylim(l2_err_min, l2_err_max)
    ax[0,1].grid(alpha=0.3)
    ax[0,1].set_title('coefs relative L2 error')

    if style == "hexbin":
        hb10 = ax[1,0].hexbin(x, flux_median_ratio, gridsize=gridsize, extent=[x_min, x_max, ratio_min, ratio_max], bins=bins, **kwargs)
        fig.colorbar(hb10, ax=ax[1,0])
    elif style == "scatter":
        ax[1,0].scatter(x, flux_median_ratio, s=markersize, alpha=markeralpha, **kwargs)
    ax[1,0].grid(alpha=0.6)
    ax[1,0].set_ylim(ratio_min, ratio_max)
    ax[1,0].set_xlabel(x_label)
    ax[1,0].set_title('flux median ratio')

    if style == "hexbin":
        hb10 = ax[1,1].hexbin(x, coef0_ratio, gridsize=gridsize, extent=[x_min, x_max, ratio_min, ratio_max], bins=bins, **kwargs)
        fig.colorbar(hb10, ax=ax[1,1])
    elif style == "scatter":
        ax[1,1].scatter(x, coef0_ratio, s=markersize, alpha=markeralpha, **kwargs)
    ax[1,1].grid(alpha=0.6)
    ax[1,1].set_ylim(ratio_min, ratio_max)
    ax[1,1].set_xlabel(x_label)
    ax[1,1].set_title('coef0 ratio')

    # fig.suptitle(f'test metric = {test_metrics['total']:.4f}')
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


def coef_plots(
        coef_true,
        coef_pred,
        figsize=(12, 8),
        style="hexbin",
        markercolor_data=None,
        markersize=4,
        markeralpha=0.2,
        markercolor_label="",
        gridsize=100,
        bins="log",
        save=False,
        filename="coef_plots.png",
        output_dirname="",
        dpi=300,
        **kwargs,
    ):
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(4, 5, figsize=figsize)
    for i in range(coef_true.shape[1]):
        rowi = i // 5
        coli = i % 5
        if style == "scatter":
            sc = ax[rowi, coli].scatter(coef_true[:,i], coef_pred[:,i], s=markersize, c=markercolor_data[:,0], alpha=markeralpha, **kwargs)
            xmin, xmax = ax[rowi, coli].get_xlim()

        elif style == "hexbin":
            ax[rowi, coli].scatter(coef_true[:,i], coef_pred[:,i], s=markersize, alpha=0.0)
            xmin, xmax = ax[rowi, coli].get_xlim()
            ax[rowi, coli].clear()
            hb = ax[rowi, coli].hexbin(coef_true[:,i], coef_pred[:,i], gridsize=gridsize, extent=[xmin, xmax, xmin, xmax], bins=bins, **kwargs)
            fig.colorbar(hb, ax=ax[rowi, coli])

        ax[rowi, coli].set_xlim(xmin, xmax)
        ax[rowi, coli].set_ylim(xmin, xmax)
        ax[rowi, coli].plot([xmin, xmax], [xmin, xmax], linewidth=0.8, alpha=0.6, color='tab:red')
        ax[rowi, coli].set_title(f'coef {i}')
        ax[rowi, coli].grid(alpha=0.4)

    if style == "scatter":
        fig.subplots_adjust(right=0.85, wspace=0.1, hspace=0.25)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        sm = ScalarMappable(norm=sc.norm, cmap=sc.cmap)
        cbar = fig.colorbar(sm, ax=cbar_ax, label=markercolor_label)

    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)

def each_param_plots(
        x_test,
        rel_l2_err,
        med_ratio,
        feature_names=None,
        figsize=(14, 12),
        gridsize=100,
        ymin1=-0.02,
        ymax1=0.5,
        ymin2 = 0.5,
        ymax2 = 1.5,
        bins="log",
        save=False,
        filename="param_plots.png",
        # filename1="param_plots_rel_l2_err.png",
        # filename2="param_plots_med_ratio.png",
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

    # for i in range(x_test.shape[1]):
    #     rowi = i // 5
    #     coli = i % 5
    #     axi = ax1[rowi, coli]
    #     xmin = np.min(x_test[:,i])
    #     xmax = np.max(x_test[:,i])
    #     hb = axi.hexbin(x_test[:,i], rel_l2_err, gridsize=gridsize, extent=[xmin, xmax, ymin1, ymax1], bins=bins, **kwargs)
    #     fig1.colorbar(hb, ax=axi)
    #     axi.set_xlabel(feature_names[i])
    #     axi.grid()
    # fig1.tight_layout()
    # if save:
    #     plt.savefig(output_dir / filename1, dpi=dpi)
    #     plt.close()

    # fig2, ax2 = plt.subplots(2, 5, figsize=figsize)
    # for i in range(x_test.shape[1]):
    #     rowi = i // 5
    #     coli = i % 5
    #     axi = ax2[rowi, coli]
    #     xmin = np.min(x_test[:,i])
    #     xmax = np.max(x_test[:,i])
    #     hb = axi.hexbin(x_test[:,i], med_ratio, gridsize=gridsize, extent=[xmin, xmax, ymin2, ymax2], bins=bins, **kwargs)
    #     fig2.colorbar(hb, ax=axi)
    #     axi.set_xlabel(feature_names[i])
    #     axi.grid()
    # fig2.tight_layout()
    # if save:
    #     plt.savefig(output_dir / filename2, dpi=dpi)
    #     plt.close()

    fig, ax = plt.subplots(4, 5, figsize=figsize)
    for i in range(x_test.shape[1]):
        rowi = i // 5
        coli = i % 5
        axi1 = ax[rowi, coli]
        xmin = np.min(x_test[:,i])
        xmax = np.max(x_test[:,i])
        hb1 = axi1.hexbin(x_test[:,i], rel_l2_err, gridsize=gridsize, extent=[xmin, xmax, ymin1, ymax1], bins=bins, **kwargs)        
        fig.colorbar(hb1, ax=axi1)
        axi1.set_xlabel(feature_names[i])
        axi1.grid()
        axi2 = ax[rowi+2, coli]
        hb2 = axi2.hexbin(x_test[:,i], med_ratio, gridsize=gridsize, extent=[xmin, xmax, ymin2, ymax2], bins=bins, **kwargs)
        fig.colorbar(hb2, ax=axi2)
        axi2.set_xlabel(feature_names[i])
        axi2.grid()
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

    device = libs.get_device(config['Device'])
    print(f"Using device: {device}")

    verbose = config["Training"]["verbose"]

    # load data
    # x_train, coef_train, mfrac_train, _ = load_data(config["Data"]["train_data"])
    train_data = libs.load_data(config["Data"]["train_data"])
    x_train = train_data["x"]
    coef_train = train_data["coef"]
    mfrac_train = train_data["mfrac"]
    train_param_keys = train_data['train_param_keys']
    default_params = train_data["default_params"]
    prior_dicts = train_data["prior_dicts"]

    if config["Data"]["validation_data"] is not None:
        # x_validation, coef_validation, mfrac_validation, _ = load_data(config["Data"]["validation_data"])
        validation_data = libs.load_data(config["Data"]["validation_data"])
        x_validation = validation_data["x"]
        coef_validation = validation_data["coef"]
        mfrac_validation = validation_data["mfrac"]
    else:
        # if validation_data is not provided, use data split
        rng = np.random.default_rng(seed=config["Data"]["validation_split_seed"])
        total_train_idx = np.arange(len(x_train))
        split_ratio = config["Data"]["validation_split"]
        idx_validation = rng.choice(total_train_idx, size=int(len(x_train)*split_ratio), replace=False)
        idx_train = np.setdiff1d(total_train_idx, idx_validation)
        x_validation = x_train[idx_validation]
        coef_validation = coef_train[idx_validation]
        mfrac_validation = mfrac_train[idx_validation]
        x_train = x_train[idx_train]
        coef_train = coef_train[idx_train]
        mfrac_train = mfrac_train[idx_train]

    # x_test, coef_test, mfrac_test, flux_test = load_data(config["Data"]["train_data"])
    test_data = libs.load_data(config["Data"]["test_data"])
    x_test = test_data["x"]
    coef_test = test_data["coef"]
    mfrac_test = test_data["mfrac"]
    flux_test = test_data["flux"]

    # load PCA
    # if config["Data"]["pca_file"] is not None:
    pca_data = np.load(config["Data"]["pca_file"])
    lbs = pca_data['lbs']
    pca_modes = pca_data['modes']
    pca_mean = pca_data['mean']
    # else:
        # lbs = None
        # pca_modes = None
        # pca_mean = None

    # first print out some settings on screen
    print(f"Train data:\t{config["Data"]["train_data"]}")
    if config["Data"]["validation_data"] is not None:
        print(f"Validation data:\t{config["Data"]["validation_data"]}")
    else:
        print(f"Validation/Train split: {config["Data"]["validation_split"]} (seed={config["Data"]["validation_split_seed"]})")
    print(f"Test data:\t{config["Data"]["test_data"]}")
    print(f"PCA file:\t{config["Data"]["pca_file"]}")
    
    print(f"Scaling method: coef={config["Scaler"]["coef_scaling"]}, mfrac={config["Scaler"]["mfrac_scaling"]}") 
    # print(f"\tcoef: ")
    # print(f"\tmfrac:")
    # print(f"Batch size: train={config["DataLoader"]["train_batch_size"]}, validation={config["DataLoader"]["validation_batch_size"]}")
    print("")
    print(f"Emulator::")
    print(f"\tshared dims: {config["Emulator"]["shared_dims"]}")
    print(f"\tcoef head dims: {config["Emulator"]["coef_head_dims"]}")
    print(f"\tmfrac head dims: {config["Emulator"]["mfrac_head_dims"]}")
    print(f"\tactivation: {config["Emulator"]["activation"]}")
    print(f"\tpredict mfrac: {config["Emulator"]["predict_mfrac"]}")
    if config["Emulator"]["dropout"] != 0:
        print(f"\tdropout: {config["Emulator"]["dropout"]}")
    # print("")
    print(f"Loss:")
    print(f"\tcoef loss: {config["Loss"]["coef_loss"]}")
    print(f"\tmfrac loss: {config["Loss"]["mfrac_loss"]}")
    print(f"\tmfrac lambda: {config["Loss"]["mfrac_lambda"]}")
    print(f"Optimizer: {config["Optimizer"]["name"]}, learning rate={config["Optimizer"]["learning_rate"]}, weight decay={config["Optimizer"]["weight_decay"]}")
    # print("")
    print(f"Training settings: ")
    print(f"\tmax_epochs={config["Training"]["max_epochs"]}, patience={config["Training"]["patience"]}, abs_tol={config["Training"]["abs_tol"]}, rel_tol={config["Training"]["rel_tol"]}")
    print(f"\ttrain batch size={config["DataLoader"]["train_batch_size"]}, validation batch size={config["DataLoader"]["validation_batch_size"]}, num worker={config["DataLoader"]["train_num_workers"]}")
    print("")

    # create Scaler
    scaler = libs.EmulatorScaler()
    (x_train_scaled, 
     coef_train_scaled, 
     mfrac_train_scaled
     ) = scaler.fit_transform(x_train, 
                              coef_train, 
                              mfrac_train, 
                              coef_scaling=config["Scaler"]["coef_scaling"],
                              mfrac_scaling=config["Scaler"]["mfrac_scaling"])
    (x_validation_scaled, 
     coef_validation_scaled, 
     mfrac_validation_scaled
    ) = scaler.transform(x_validation, 
                         coef_validation, 
                         mfrac_validation)
    (x_test_scaled, 
     coef_test_scaled, 
     mfrac_test_scaled
    ) = scaler.transform(x_test, 
                         coef_test, 
                         mfrac_test)

    x_train_scaled = x_train_scaled.astype(np.float32)
    coef_train_scaled = coef_train_scaled.astype(np.float32)
    mfrac_train_scaled = mfrac_train_scaled.astype(np.float32)

    x_validation_scaled = x_validation_scaled.astype(np.float32)
    coef_validation_scaled = coef_validation_scaled.astype(np.float32)
    mfrac_validation_scaled = mfrac_validation_scaled.astype(np.float32)

    x_test_scaled = x_test_scaled.astype(np.float32)
    coef_test_scaled = coef_test_scaled.astype(np.float32)
    mfrac_test_scaled = mfrac_test_scaled.astype(np.float32)

    # Create Dataset objects and DataLoaders
    train_dataset = libs.SPSDataset(x_train_scaled, coef_train_scaled, mfrac_train_scaled)
    validation_dataset = libs.SPSDataset(x_validation_scaled, coef_validation_scaled, mfrac_validation_scaled)
    test_dataset = libs.SPSDataset(x_test_scaled, coef_test_scaled, mfrac_test_scaled)

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
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["DataLoader"]["validation_batch_size"],
        shuffle=config["DataLoader"]["validation_shuffle"],
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

    shared_dims = tuple(config["Emulator"]["shared_dims"])
    if config["Emulator"]["coef_head_dims"] is not None:
        coef_head_dims = tuple(config["Emulator"]["coef_head_dims"])
    else:
        coef_head_dims = ()
    if config["Emulator"]["mfrac_head_dims"] is not None:
        mfrac_head_dims = tuple(config["Emulator"]["mfrac_head_dims"])
    else:
        mfrac_head_dims = ()

    n_features = x_train.shape[1]
    n_coefs = coef_train.shape[1]

    # Create emulator model object with config architecture and send to device
    model = libs.SPSEmulator(
        n_features=n_features,
        n_coefs=n_coefs,
        shared_dims=shared_dims,
        coef_head_dims=coef_head_dims,
        mfrac_head_dims=mfrac_head_dims,
        activation=config["Emulator"]["activation"],
        dropout=config["Emulator"]["dropout"],
        predict_mfrac=config["Emulator"]["predict_mfrac"]
    ).to(device)

    # create criterion object from Loss class and send to device
    criterion = libs.EmulatorLoss(
        coef_loss=config["Loss"]["coef_loss"],
        mfrac_loss=config["Loss"]["mfrac_loss"],
        mfrac_lambda=config["Loss"]["mfrac_lambda"],
        pca_modes=pca_modes,
        pca_mean=pca_mean,
        coef_scale_mean=scaler.coef_mean,
        coef_scale_std=scaler.coef_std,
        relative_flux_eps=config["Loss"]["relative_flux_eps"],
        rmse_eps=config["Loss"]["rmse_eps"],
        max_log10_flux=config["Loss"]["max_log10_flux"],
        wavelength_weights=config["Loss"]["wavelength_weights"]
    ).to(device)

    # create optimizer from optimizer class, AFTER creating model
    optimizer = libs.make_optimizer(
        name=config["Optimizer"]["name"],
        parameters=model.parameters(),
        learning_rate=config["Optimizer"]["learning_rate"],
        weight_decay=config["Optimizer"]["weight_decay"]
    )

    train_start_time = time.perf_counter()
    # Actual training run
    history = libs.fit_emulator(
        model=model,
        train_loader=train_loader,
        valid_loader=validation_loader,
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
    predictions = libs.predict_emulator(
        model=model,
        x_unscaled=x_test,
        scaler=scaler,
        device=device,
        batch_size=config["DataLoader"]["test_batch_size"],
    )

    coef_pred = predictions["coef"]
    if 'mfrac' in predictions:
        mfrac_pred = predictions["mfrac"]
    else:
        mfrac_pred = np.zeros(mfrac_test.shape[0])

    log10_flux_pred = coef_pred @ pca_modes + pca_mean
    flux_pred = 10.0 ** log10_flux_pred

    rel_l2_errs = np.linalg.norm(flux_pred-flux_test, axis=1)/np.linalg.norm(flux_test, axis=1)
    median_ratios = np.median(flux_pred/flux_test, axis=1)

    coefs_rel_l2_errs = np.linalg.norm(coef_pred-coef_test, axis=1)/np.linalg.norm(coef_test, axis=1)
    coef0_ratios = coef_pred[:,0]/coef_test[:,0]

    test_metrics = libs.run_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )
    if verbose:
        print(f"test set loss={test_metrics[config["Training"]["monitor"]]}")
        # print(test_metrics)

    if config["Outputs"]["save_plots"]:
        # make plots
        rel_err_ratio_plots(
            x=x_test[:,0],
            flux_rel_l2_err=rel_l2_errs,
            coef_rel_l2_err=coefs_rel_l2_errs,
            flux_median_ratio=median_ratios,
            coef0_ratio=coef0_ratios,
            save=True,
            output_dirname=config["Outputs"]["outputs_dir"],
            filename="rel_err_ratio.png",
        )

        mfrac_plots(
            mfrac_test,
            mfrac_pred,
            save=True,
            output_dirname=config["Outputs"]["outputs_dir"],
            filename="mfrac_plots.png"
        )

        coef_plots(
            coef_test,
            coef_pred,
            style="hexbin",
            save=True,
            output_dirname=config["Outputs"]["outputs_dir"],
            filename="coef_plots.png"
        )

        each_param_plots(
            x_test,
            rel_l2_errs,
            median_ratios,
            feature_names=train_param_keys,
            save=True,
            output_dirname=config["Outputs"]["outputs_dir"],
            filename="param_plots.png"
            # filename1="param_plots_rel_l2_err.png",
            # filename2="param_plots_med_ratio.png"
        )


    # Saving emulator model and scaler, as well as other metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,

        "scaler_state": scaler.state_dict(),

        "pca_lbs": np.asarray(lbs),
        "pca_modes": np.asarray(pca_modes),
        "pca_mean": np.asarray(pca_mean),

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
        print(f"Trained emulator saved: {checkpoint_dir / config["Outputs"]["emulator_filename"]}")


    # time one iteration with the emulator
    model_cpu = model.to("cpu")
    model_cpu.eval()

    x_test_one = x_test[0]

    # Warm up before timing
    for _ in range(20):
        _ = libs.predict_one(
            model=model_cpu,
            x=x_test_one,
            device=torch.device("cpu"),
            scaler=scaler,
        )

    n_repeats = 1000
    start_time = time.perf_counter()
    for _ in range(n_repeats):
        _ = libs.predict_one(
            model=model_cpu,
            x=x_test_one,
            device=torch.device("cpu"),
            scaler=scaler,
        )
    elapsed_time = time.perf_counter() - start_time
    mean_seconds = elapsed_time / n_repeats

    if verbose:
        print(f"CPU single-object mean inference time: {mean_seconds*1e6:.4f} us (from {n_repeats} calls)")
        # print(f"Number of calls: {n_repeats}")
        # print(f"Total time:      {elapsed_time:.6f} s")
        # print(f"Mean per call:   {mean_seconds * 1e3:.6f} ms")
        # print(f"Rate:            {1.0 / mean_seconds:.2f} calls/s")

if __name__ == '__main__':
    main()

