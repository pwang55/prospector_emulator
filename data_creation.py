"""

Usage:
    $ python data_creation.py -c configs/data_creation.yaml

"""
import numpy as np
# from importlib import reload
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# from scipy.stats import t, truncnorm
import joblib
import pandas as pd
import json
import libs.emulator_libs as elibs
import libs.param_libs as plibs
import libs.custom_prospector_tools as cpt
from sklearn.decomposition import PCA
import yaml
import argparse


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
    return parser.parse_args()


# def load_config(filename):
args = parse_args()
config_filename = args.config
with open(config_filename, "r") as file:
    config = yaml.safe_load(file)

sfh_type = config["sfh_type"]
train_param_keys = config["train_param_keys"]
wl_min = config["wl_min"]
wl_max = config["wl_max"]

ntrain = config["ntrain"]
nvalid = config["nvalid"]
ntest = config["ntest"]
seed_train = config["seed_train"]
seed_valid = config["seed_valid"]
seed_test = config["seed_test"]

npca = config["npca"]

output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
train_flux_filename = config["train_flux_filename"]
train_filename = config["train_filename"]
valid_filename = config["valid_filename"]
test_filename = config["test_filename"]
pca_filename = config["pca_filename"]

prior_dicts = config["prior_dicts"]

# modify filenames Ndat
train_flux_filename = train_flux_filename.replace("Ndat", f"{int(ntrain/1000)}k")
train_filename = train_filename.replace("Ndat", f"{int(ntrain/1000)}k")
pca_filename = pca_filename.replace("Ndat", f"{int(ntrain/1000)}k")

valid_filename = valid_filename.replace("Ndat", f"{int(nvalid/1000)}k")
test_filename = test_filename.replace("Ndat", f"{int(ntest/1000)}k")

# this section automatically creates default_params and train_param_keys, 
# train_param_keys and default_params will be propragated all the way to the emulator mcmc code
default_params = plibs.get_default_params(sfh_type, train_param_keys)

# create random values based on input prior settings and allowed train_param_keys
x_train = plibs.generate_random_values(
    prior_dicts=prior_dicts,
    train_param_keys=train_param_keys,
    nsamples=ntrain,
    rng=np.random.default_rng(seed_train)
)

x_valid = plibs.generate_random_values(
    prior_dicts=prior_dicts,
    train_param_keys=train_param_keys,
    nsamples=nvalid,
    rng=np.random.default_rng(seed_valid)
)

x_test = plibs.generate_random_values(
    prior_dicts=prior_dicts,
    train_param_keys=train_param_keys,
    nsamples=ntest,
    rng=np.random.default_rng(seed_test)
)

# based on generated rand_vals which are 2d arrays of shape (nsamples, n_params), which is the format for emulator,
# covert them to prospector compatible dictionary format, based on train_param_keys and default_params
all_rand_myparams_train = plibs.rand_vals_to_all_params(
    rand_vals=x_train, 
    train_param_keys=train_param_keys, 
    default_params=default_params)
all_rand_myparams_valid = plibs.rand_vals_to_all_params(
    rand_vals=x_valid, 
    train_param_keys=train_param_keys, 
    default_params=default_params)
all_rand_myparams_test = plibs.rand_vals_to_all_params(
    rand_vals=x_test, 
    train_param_keys=train_param_keys, 
    default_params=default_params)


# run prospector just once (extremely slowly) to get lbs
print("initializing Prospector...")
pros_obj = cpt.custom_prospector(sfh_type='continuity_sfh')
lbs, flux, flux_conv, mfrac = pros_obj.generate_spectra(myparams=all_rand_myparams_train[0], wl_min=wl_min, wl_max=wl_max)

# create train flux and mfrac
flux_train = np.zeros((ntrain, len(lbs)))
mfrac_train = np.zeros(ntrain)
for i in range(ntrain):
    print(f"\rCreating train spectra {i+1}/{ntrain}", end="")
    my_param_i = all_rand_myparams_train[i]
    lbs, flux, _, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max)
    h = flux < 0
    flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
    flux[h] = flux_interp
    flux_train[i] = flux
    mfrac_train[i] = mfrac
np.savez_compressed(output_dir / train_flux_filename, 
                    lbs=lbs, 
                    flux=flux_train, 
                    # logflux=logflux_train, 
                    mfrac=mfrac_train,
                    train_param_keys=np.asarray(train_param_keys), 
                    default_params=json.dumps(default_params),
                    prior_dicts=json.dumps(prior_dicts))

log10_flux_train = np.log10(flux_train)

# create PCA modes and mean based on training spectra
print(f"\nCreating PCA modes...")
pca = PCA(n_components=npca)
pca_coef = pca.fit_transform(log10_flux_train)
pca_modes = pca.components_
pca_mean = pca.mean_

print("explained variance ratio:", pca.explained_variance_ratio_)

# delete flux_train to save some memory
del flux_train, log10_flux_train

np.savez_compressed(output_dir / pca_filename, 
                    lbs=lbs, 
                    modes=pca_modes, 
                    mean=pca_mean,
                    train_param_keys=np.asarray(train_param_keys), 
                    default_params=json.dumps(default_params),
                    prior_dicts=json.dumps(prior_dicts))

# save x, coef, mfrac to a separate file that can serve as input for emulator_training.py
np.savez(output_dir / train_filename, 
         x=x_train, 
         coef=pca_coef, 
         mfrac=mfrac_train, 
         train_param_keys=np.asarray(train_param_keys), 
         default_params=json.dumps(default_params),
         prior_dicts=json.dumps(prior_dicts))

# create valid set and save x, coef, mfrac, flux to a single file
flux_valid = np.zeros((nvalid, len(lbs)))
mfrac_valid = np.zeros(nvalid)
for i in range(nvalid):
    print(f"\rCreating valid spectra {i+1}/{nvalid}", end="")
    my_param_i = all_rand_myparams_valid[i]
    lbs, flux, _, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max)
    h = flux < 0
    flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
    flux[h] = flux_interp
    flux_valid[i] = flux
    mfrac_valid[i] = mfrac

log10_flux_valid = np.log10(flux_valid)
coef_valid = (log10_flux_valid - pca_mean) @ pca_modes.T

np.savez_compressed(output_dir / valid_filename, 
                    lbs=lbs, 
                    x=x_valid, 
                    flux=flux_valid, 
                    coef=coef_valid, 
                    mfrac=mfrac_valid,
                    train_param_keys=np.asarray(train_param_keys), 
                    default_params=json.dumps(default_params),
                    prior_dicts=json.dumps(prior_dicts))

del flux_valid, log10_flux_valid
print("")
# create test set and save x, coef, mfrac, flux to a single file
flux_test = np.zeros((ntest, len(lbs)))
mfrac_test = np.zeros(ntest)
for i in range(ntest):
    print(f"\rCreating test spectra {i+1}/{ntest}", end="")
    my_param_i = all_rand_myparams_test[i]
    lbs, flux, _, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max)
    h = flux < 0
    flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
    flux[h] = flux_interp
    flux_test[i] = flux
    mfrac_test[i] = mfrac

log10_flux_test = np.log10(flux_test)
coef_test = (log10_flux_test - pca_mean) @ pca_modes.T

np.savez_compressed(output_dir / test_filename, 
                    lbs=lbs, 
                    x=x_test, 
                    flux=flux_test, 
                    coef=coef_test, 
                    mfrac=mfrac_test,
                    train_param_keys=np.asarray(train_param_keys), 
                    default_params=json.dumps(default_params),
                    prior_dicts=json.dumps(prior_dicts))
print("")

