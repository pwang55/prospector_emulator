"""

This code reads config and save prospector log10flux to file,
so that they can be used to make PCA modes/mean and compile x/coef/scale/mfracs into training ready files format.

Usage:
    $ python create_train_valid_test_spectra.py -c configs/data_creation.yaml

"""
import numpy as np
# from importlib import reload
# import matplotlib.pyplot as plt
# import sys
from pathlib import Path
# from scipy.stats import t, truncnorm
# import joblib
# import pandas as pd
import json
# import libs.emulator_libs as elibs
import libs.param_libs as plibs
import libs.data_libs as dlibs
import libs.custom_prospector_tools as cpt
# from sklearn.decomposition import PCA
import yaml
import argparse
import h5py

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
filter_list = config["filters"]
filters = dlibs.read_filters(filter_list, return_unique_inverse=True, return_lamb_obs=True)
lamb_obs = filters[-1]
nfilt = lamb_obs.shape[0]
ndim = len(train_param_keys)

ntrain = config["ntrain"]
nvalid = config["nvalid"]
ntest = config["ntest"]
seed_train = config["seed_train"]
seed_valid = config["seed_valid"]
seed_test = config["seed_test"]

output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
train_log10flux_filename = config["train_log10flux_filename"]
valid_log10flux_filename = config["valid_log10flux_filename"]
test_log10flux_filename = config["test_log10flux_filename"]

chunk_size = config["chunk_size"]
prior_dicts = config["prior_dicts"]

# modify filenames Ndat
train_log10flux_filename = train_log10flux_filename.replace("Ndat", f"{int(ntrain/1000)}k")
valid_log10flux_filename = valid_log10flux_filename.replace("Ndat", f"{int(nvalid/1000)}k")
test_log10flux_filename = test_log10flux_filename.replace("Ndat", f"{int(ntest/1000)}k")


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
nlbs = len(lbs)


# create train flux and mfrac
# flux_train = np.zeros((ntrain, len(lbs)))
# mfrac_train = np.zeros(ntrain)

with h5py.File(output_dir / train_log10flux_filename, "w") as f:
    dset_lbs = f.create_dataset(
        "lbs",
        data=lbs
    )
    dset_lamb_obs = f.create_dataset(
        "lamb_obs",
        data=lamb_obs
    )
    dset_x = f.create_dataset(
        "x",
        shape=(ntrain, ndim),
        dtype=np.float32,
        chunks=(chunk_size, ndim),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_log10flux = f.create_dataset(
        "log10flux",
        shape=(ntrain, nlbs),
        dtype=np.float32,
        chunks=(chunk_size, nlbs),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_flux_fiducial = f.create_dataset(
        "flux_fiducial",
        shape=(ntrain, nfilt),
        dtype=np.float32,
        chunks=(chunk_size, nfilt),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_mfrac = f.create_dataset(
        "mfrac",
        shape=ntrain,
        dtype=np.float32,
        chunks=chunk_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    f.attrs["train_param_keys"] = train_param_keys
    f.attrs["default_params"] = json.dumps(default_params)
    f.attrs["prior_dicts"] = json.dumps(prior_dicts)

    for start in range(0, ntrain, chunk_size):
        end = np.min((start+chunk_size, ntrain))

        flux_chunk = np.zeros((end-start, nlbs))
        flux_conv_chunk = np.zeros((end-start, nfilt))
        mfrac_chunk = np.zeros(end-start)
        for ci, i in enumerate(range(start, end)):

            print(f"\rCreating train spectra {i+1}/{ntrain}", end="")
            my_param_i = all_rand_myparams_train[i]
            lbs, flux, flux_conv, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max, filters=filters)
            h = flux <= 0
            flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
            flux[h] = flux_interp
            flux_chunk[ci] = flux
            flux_conv_chunk[ci] = flux_conv
            mfrac_chunk[ci] = mfrac

        log10flux_chunk = np.log10(flux_chunk)
        dset_log10flux[start:end] = log10flux_chunk
        dset_flux_fiducial[start:end] = flux_conv_chunk
        dset_mfrac[start:end] = mfrac_chunk
        dset_x[start:end] = x_train[start:end]

# for i in range(ntrain):
#     print(f"\rCreating train spectra {i+1}/{ntrain}", end="")
#     my_param_i = all_rand_myparams_train[i]
#     lbs, flux, _, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max)
#     h = flux < 0
#     flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
#     flux[h] = flux_interp
#     flux_train[i] = flux
#     mfrac_train[i] = mfrac

# log10flux_train = np.log10(flux_train)

# np.savez_compressed(output_dir / train_log10flux_filename, 
#                     lbs=lbs, 
#                     # flux=flux_train, 
#                     log10flux=log10flux_train, 
#                     mfrac=mfrac_train,
#                     train_param_keys=np.asarray(train_param_keys), 
#                     default_params=json.dumps(default_params),
#                     prior_dicts=json.dumps(prior_dicts))


# delete flux_train to save some memory
# del flux_train, log10flux_train
print("")

with h5py.File(output_dir / valid_log10flux_filename, "w") as f:
    dset_lbs = f.create_dataset(
        "lbs",
        data=lbs
    )
    dset_lamb_obs = f.create_dataset(
        "lamb_obs",
        data=lamb_obs
    )
    dset_x = f.create_dataset(
        "x",
        shape=(nvalid, ndim),
        dtype=np.float32,
        chunks=(chunk_size, ndim),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_log10flux = f.create_dataset(
        "log10flux",
        shape=(nvalid, nlbs),
        dtype=np.float32,
        chunks=(chunk_size, nlbs),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_flux_fiducial = f.create_dataset(
        "flux_fiducial",
        shape=(nvalid, nfilt),
        dtype=np.float32,
        chunks=(chunk_size, nfilt),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_mfrac = f.create_dataset(
        "mfrac",
        shape=nvalid,
        dtype=np.float32,
        chunks=chunk_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    f.attrs["train_param_keys"] = train_param_keys
    f.attrs["default_params"] = json.dumps(default_params)
    f.attrs["prior_dicts"] = json.dumps(prior_dicts)

    for start in range(0, nvalid, chunk_size):
        end = np.min((start+chunk_size, nvalid))

        flux_chunk = np.zeros((end-start, nlbs))
        flux_conv_chunk = np.zeros((end-start, nfilt))
        mfrac_chunk = np.zeros(end-start)
        for ci, i in enumerate(range(start, end)):

            print(f"\rCreating valid spectra {i+1}/{nvalid}", end="")
            my_param_i = all_rand_myparams_valid[i]
            lbs, flux, flux_conv, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max, filters=filters)
            h = flux <= 0
            flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
            flux[h] = flux_interp
            flux_chunk[ci] = flux
            flux_conv_chunk[ci] = flux_conv
            mfrac_chunk[ci] = mfrac

        log10flux_chunk = np.log10(flux_chunk)
        dset_log10flux[start:end] = log10flux_chunk
        dset_flux_fiducial[start:end] = flux_conv_chunk
        dset_mfrac[start:end] = mfrac_chunk
        dset_x[start:end] = x_valid[start:end]

# create valid set and save x, coef, mfrac, flux to a single file
# flux_valid = np.zeros((nvalid, len(lbs)))
# mfrac_valid = np.zeros(nvalid)
# for i in range(nvalid):
#     print(f"\rCreating valid spectra {i+1}/{nvalid}", end="")
#     my_param_i = all_rand_myparams_valid[i]
#     lbs, flux, _, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max)
#     h = flux < 0
#     flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
#     flux[h] = flux_interp
#     flux_valid[i] = flux
#     mfrac_valid[i] = mfrac

# log10flux_valid = np.log10(flux_valid)

# np.savez_compressed(output_dir / valid_log10flux_filename, 
#                     lbs=lbs, 
#                     # flux=flux_train, 
#                     log10flux=log10flux_valid, 
#                     mfrac=mfrac_valid,
#                     train_param_keys=np.asarray(train_param_keys), 
#                     default_params=json.dumps(default_params),
#                     prior_dicts=json.dumps(prior_dicts))

# del flux_valid, log10flux_valid
print("")
with h5py.File(output_dir / test_log10flux_filename, "w") as f:
    dset_lbs = f.create_dataset(
        "lbs",
        data=lbs
    )
    dset_lamb_obs = f.create_dataset(
        "lamb_obs",
        data=lamb_obs
    )
    dset_x = f.create_dataset(
        "x",
        shape=(ntest, ndim),
        dtype=np.float32,
        chunks=(chunk_size, ndim),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_log10flux = f.create_dataset(
        "log10flux",
        shape=(ntest, nlbs),
        dtype=np.float32,
        chunks=(chunk_size, nlbs),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_flux_fiducial = f.create_dataset(
        "flux_fiducial",
        shape=(ntest, nfilt),
        dtype=np.float32,
        chunks=(chunk_size, nfilt),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_mfrac = f.create_dataset(
        "mfrac",
        shape=ntest,
        dtype=np.float32,
        chunks=chunk_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    f.attrs["train_param_keys"] = train_param_keys
    f.attrs["default_params"] = json.dumps(default_params)
    f.attrs["prior_dicts"] = json.dumps(prior_dicts)

    for start in range(0, ntest, chunk_size):
        end = np.min((start+chunk_size, ntest))

        flux_chunk = np.zeros((end-start, nlbs))
        flux_conv_chunk = np.zeros((end-start, nfilt))
        mfrac_chunk = np.zeros(end-start)
        for ci, i in enumerate(range(start, end)):

            print(f"\rCreating test spectra {i+1}/{ntest}", end="")
            my_param_i = all_rand_myparams_test[i]
            lbs, flux, flux_conv, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max, filters=filters)
            h = flux <= 0
            flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
            flux[h] = flux_interp
            flux_chunk[ci] = flux
            flux_conv_chunk[ci] = flux_conv
            mfrac_chunk[ci] = mfrac

        log10flux_chunk = np.log10(flux_chunk)
        dset_log10flux[start:end] = log10flux_chunk
        dset_flux_fiducial[start:end] = flux_conv_chunk
        dset_mfrac[start:end] = mfrac_chunk
        dset_x[start:end] = x_test[start:end]


# create test set and save x, coef, mfrac, flux to a single file
# flux_test = np.zeros((ntest, len(lbs)))
# mfrac_test = np.zeros(ntest)
# for i in range(ntest):
#     print(f"\rCreating test spectra {i+1}/{ntest}", end="")
#     my_param_i = all_rand_myparams_test[i]
#     lbs, flux, _, mfrac = pros_obj.generate_spectra(myparams=my_param_i, wl_min=wl_min, wl_max=wl_max)
#     h = flux < 0
#     flux_interp = np.interp(lbs[h], lbs[~h], flux[~h])
#     flux[h] = flux_interp
#     flux_test[i] = flux
#     mfrac_test[i] = mfrac

# log10flux_test = np.log10(flux_test)

# np.savez_compressed(output_dir / test_log10flux_filename, 
#                     lbs=lbs, 
#                     # flux=flux_train, 
#                     log10flux=log10flux_test, 
#                     mfrac=mfrac_test,
#                     train_param_keys=np.asarray(train_param_keys), 
#                     default_params=json.dumps(default_params),
#                     prior_dicts=json.dumps(prior_dicts))

print("")

