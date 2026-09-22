"""

Usage:
    $ python make_x_coef_files.py -c configs/make_x_coef_files.yaml

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
# import libs.custom_prospector_tools as cpt
# from sklearn.decomposition import PCA
import yaml
import argparse
import h5py
from scipy.sparse.linalg import eigsh


default_batch_size = 2000

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

npca = config["npca"]
output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
pca_filename = config["pca_filename"]

batch_size = config["batch_size"]
scaling = config["scaling"]

# list of filenames
train_log10flux_filenames = config["train_log10flux_filenames"]
valid_log10flux_filenames = config["valid_log10flux_filenames"]
test_log10flux_filenames = config["test_log10flux_filenames"]

out_train_filename = config["train_filename"]
out_valid_filename = config["valid_filename"]
out_test_filename = config["test_filename"]


# first grab necessary dimensions and compile bounds in prior info
ntrain_total = 0
for i, filename in enumerate(train_log10flux_filenames):
    with h5py.File(filename, "r") as f:
        lbs = f["lbs"][()]
        lamb_obs = f["lamb_obs"][()]
        train_param_keys = f.attrs["train_param_keys"].tolist()
        default_params = json.loads(f.attrs["default_params"])

        x = f["x"]
        ntrain = x.shape[0]
        ntrain_total += ntrain

        nlbs = len(lbs)
        ndim = x.shape[1]
        nfilt = len(lamb_obs)

        if i == 0:
            prior_dicts = json.loads(f.attrs["prior_dicts"]).copy()
        else:
            prior_dicts_new = json.loads(f.attrs["prior_dicts"]).copy()
            for _, key in enumerate(prior_dicts.keys()):
                low0, high0 = prior_dicts[key]["bounds"]
                low_new, high_new = prior_dicts_new[key]["bounds"]
                if low_new < low0:
                    low0 = low_new
                if high_new > high0:
                    high0 = high_new
                prior_dicts[key]["bounds"] = [low0, high0]


train_log10flux_sum = np.empty((nlbs))
Cov = np.empty((nlbs, nlbs))

# TODO scaling?
# print("Calculating PCA modes and mean...")
# batch read training file and build PCA
# first iteration pass to get the mean log10flux
global_start = 0
for i, filename in enumerate(train_log10flux_filenames):
    with h5py.File(filename, "r") as f:
        lbs = f["lbs"][()]
        train_log10flux = f["log10flux"]

        ntrain = train_log10flux.shape[0]
        # ntrain_total = ntrain_total + ntrain

        if batch_size is None:
            if train_log10flux.chunks is not None:
                batch_size = train_log10flux.chunks[0]
            else:
                batch_size = default_batch_size

        for start in range(0, ntrain, batch_size):
            print(f"\rBatch calculating pca mean {(global_start+start) // batch_size + 1}/{ntrain_total // batch_size + ((ntrain_total % batch_size)>0)}", end="")
            end = np.min((start+batch_size, ntrain))

            log10flux_chunk = train_log10flux[start:end]
            train_log10flux_sum += np.sum(log10flux_chunk, axis=0)
        global_start += ntrain

pca_mean = train_log10flux_sum / ntrain_total
print("")

global_start = 0
# second iteration pass to calculate covariance matrix
for i, filename in enumerate(train_log10flux_filenames):
    with h5py.File(filename, "r") as f:
        # lbs = f["lbs"][()]
        # x = f["x"]
        mfrac = f["mfrac"]
        train_log10flux = f["log10flux"]

        ntrain = train_log10flux.shape[0]

        if batch_size is None:
            if train_log10flux.chunks is not None:
                batch_size = train_log10flux.chunks[0]
            else:
                batch_size = default_batch_size

        for start in range(0, ntrain, batch_size):
            print(f"\rBatch calculating X.T @ X {(global_start+start) // batch_size + 1}/{ntrain_total // batch_size + ((ntrain_total % batch_size)>0)}", end="")
            end = np.min((start+batch_size, ntrain))

            chunk = train_log10flux[start:end] - pca_mean
            Cov += chunk.T @ chunk
        global_start += ntrain

Cov = Cov / (ntrain_total-1)

evals, evecs = eigsh(Cov, k=npca, which="LA")
eidx = np.argsort(evals)[::-1]
evals = evals[eidx]
evecs = evecs[:, eidx]
pca_modes = evecs.T

pca_explained_variance = evals
pca_explained_variance_ratio = evals / np.sum(evals)
print("")
print(pca_explained_variance_ratio)

pca_filename = pca_filename.replace("Ndat", f"{int(ntrain_total/1000)}k")

np.savez(output_dir / pca_filename, 
         lbs=lbs, 
         modes=pca_modes, 
         mean=pca_mean, 
         pca_explained_variance=evals,
         pca_explained_variance_ratio=pca_explained_variance_ratio,
         train_param_keys=np.asarray(train_param_keys),
         default_params=json.dumps(default_params),
         prior_dicts=json.dumps(prior_dicts))

# print("Calculate PCA coefs for train data...")

out_train_filename = out_train_filename.replace("Ndat", f"{int(ntrain_total/1000)}k")

with h5py.File(output_dir / out_train_filename, "w") as f_out:
    f_out.attrs["train_param_keys"] = train_param_keys
    f_out.attrs["default_params"] = json.dumps(default_params)
    f_out.attrs["prior_dicts"] = json.dumps(prior_dicts)

    dset_lbs = f_out.create_dataset(
        "lbs",
        data=lbs
    )
    dset_lamb_obs = f_out.create_dataset(
        "lamb_obs",
        data=lamb_obs
    )
    dset_x = f_out.create_dataset(
        "x",
        shape=(ntrain_total, ndim),
        dtype=np.float32,
        chunks=(batch_size, ndim),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_coef = f_out.create_dataset(
        "coef",
        shape=(ntrain_total, npca),
        dtype=np.float32,
        chunks=(batch_size, npca),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_flux_fiducial = f_out.create_dataset(
        "flux_fiducial",
        shape=(ntrain_total, nfilt),
        dtype=np.float32,
        chunks=(batch_size, nfilt),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_mfrac = f_out.create_dataset(
        "mfrac",
        shape=ntrain_total,
        dtype=np.float32,
        chunks=batch_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_scale = f_out.create_dataset(
        "scale",
        shape=ntrain_total,
        dtype=np.float32,
        chunks=batch_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    global_start = 0
    # iterate over input training file to calculate the coefficients
    for i, filename in enumerate(train_log10flux_filenames):
        with h5py.File(filename, "r") as f_in:
            x = f_in["x"]
            mfrac = f_in["mfrac"]
            flux_fiducial = f_in["flux_fiducial"]
            train_log10flux = f_in["log10flux"]

            ntrain = train_log10flux.shape[0]

            if batch_size is None:
                if train_log10flux.chunks is not None:
                    batch_size = train_log10flux.chunks[0]
                else:
                    batch_size = default_batch_size

            for start in range(0, ntrain, batch_size):
                print(f"\rBatch calculate PCA coefs for train data {(global_start+start) // batch_size + 1}/{ntrain_total // batch_size + ((ntrain_total % batch_size)>0)}", end="")
                end = np.min((start+batch_size, ntrain))

                train_log10flux_chunk = train_log10flux[start:end]
                # TODO scaling?
                coef_train = (train_log10flux_chunk - pca_mean) @ pca_modes.T

                out_start = global_start + start
                out_end = out_start + end - start

                dset_coef[out_start:out_end] = coef_train
                dset_x[out_start:out_end] = x[start:end]
                dset_flux_fiducial[out_start:out_end] = flux_fiducial[start:end]
                dset_mfrac[out_start:out_end] = mfrac[start:end]
                dset_scale[out_start:out_end] = np.ones(end-start)  # TODO

            global_start += ntrain


print("")
# print("Calculate PCA coefs for valid data...")

# get nvalid_total first
nvalid_total = 0
for i, filename in enumerate(valid_log10flux_filenames):
    with h5py.File(filename, "r") as f:
        x = f["x"]
        nvalid = x.shape[0]
        nvalid_total += nvalid

out_valid_filename = out_valid_filename.replace("Ndat", f"{int(nvalid_total/1000)}k")

with h5py.File(output_dir / out_valid_filename, "w") as f_out:
    f_out.attrs["train_param_keys"] = train_param_keys
    f_out.attrs["default_params"] = json.dumps(default_params)
    f_out.attrs["prior_dicts"] = json.dumps(prior_dicts)

    dset_lbs = f_out.create_dataset(
        "lbs",
        data=lbs
    )
    dset_lamb_obs = f_out.create_dataset(
        "lamb_obs",
        data=lamb_obs
    )
    dset_x = f_out.create_dataset(
        "x",
        shape=(nvalid_total, ndim),
        dtype=np.float32,
        chunks=(batch_size, ndim),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_coef = f_out.create_dataset(
        "coef",
        shape=(nvalid_total, npca),
        dtype=np.float32,
        chunks=(batch_size, npca),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_flux_fiducial = f_out.create_dataset(
        "flux_fiducial",
        shape=(nvalid_total, nfilt),
        dtype=np.float32,
        chunks=(batch_size, nfilt),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_mfrac = f_out.create_dataset(
        "mfrac",
        shape=nvalid_total,
        dtype=np.float32,
        chunks=batch_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_scale = f_out.create_dataset(
        "scale",
        shape=nvalid_total,
        dtype=np.float32,
        chunks=batch_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_log10flux = f_out.create_dataset(
        "log10flux",
        shape=(nvalid_total, nlbs),
        dtype=np.float32,
        chunks=(batch_size, nlbs),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    global_start = 0
    # iterate over input valid file to calculate the coefficients
    for i, filename in enumerate(valid_log10flux_filenames):
        with h5py.File(filename, "r") as f_in:
            x = f_in["x"]
            mfrac = f_in["mfrac"]
            flux_fiducial = f_in["flux_fiducial"]
            valid_log10flux = f_in["log10flux"]

            nvalid = valid_log10flux.shape[0]

            if batch_size is None:
                if valid_log10flux.chunks is not None:
                    batch_size = valid_log10flux.chunks[0]
                else:
                    batch_size = default_batch_size

            for start in range(0, nvalid, batch_size):
                print(f"\rBatch calculate PCA coefs for valid data {(global_start+start) // batch_size + 1}/{nvalid_total // batch_size + ((nvalid_total % batch_size)>0)}", end="")

                end = np.min((start+batch_size, nvalid))

                valid_log10flux_chunk = valid_log10flux[start:end]
                # TODO scaling?
                coef_valid = (valid_log10flux_chunk - pca_mean) @ pca_modes.T

                out_start = global_start + start
                out_end = out_start + end - start

                dset_log10flux[out_start:out_end] = valid_log10flux_chunk
                dset_coef[out_start:out_end] = coef_valid
                dset_x[out_start:out_end] = x[start:end]
                dset_flux_fiducial[out_start:out_end] = flux_fiducial[start:end]
                dset_mfrac[out_start:out_end] = mfrac[start:end]
                dset_scale[out_start:out_end] = np.ones(end-start)  # TODO

            global_start += nvalid



print("")
# print("Calculate PCA coefs for test data...")

# get ntest_total first
ntest_total = 0
for i, filename in enumerate(test_log10flux_filenames):
    with h5py.File(filename, "r") as f:
        x = f["x"]
        ntest = x.shape[0]
        ntest_total += ntest

out_test_filename = out_test_filename.replace("Ndat", f"{int(ntest_total/1000)}k")

with h5py.File(output_dir / out_test_filename, "w") as f_out:
    f_out.attrs["train_param_keys"] = train_param_keys
    f_out.attrs["default_params"] = json.dumps(default_params)
    f_out.attrs["prior_dicts"] = json.dumps(prior_dicts)

    dset_lbs = f_out.create_dataset(
        "lbs",
        data=lbs
    )
    dset_lamb_obs = f_out.create_dataset(
        "lamb_obs",
        data=lamb_obs
    )
    dset_x = f_out.create_dataset(
        "x",
        shape=(ntest_total, ndim),
        dtype=np.float32,
        chunks=(batch_size, ndim),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_coef = f_out.create_dataset(
        "coef",
        shape=(ntest_total, npca),
        dtype=np.float32,
        chunks=(batch_size, npca),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_flux_fiducial = f_out.create_dataset(
        "flux_fiducial",
        shape=(ntest_total, nfilt),
        dtype=np.float32,
        chunks=(batch_size, nfilt),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_mfrac = f_out.create_dataset(
        "mfrac",
        shape=ntest_total,
        dtype=np.float32,
        chunks=batch_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_scale = f_out.create_dataset(
        "scale",
        shape=ntest_total,
        dtype=np.float32,
        chunks=batch_size,
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    dset_log10flux = f_out.create_dataset(
        "log10flux",
        shape=(ntest_total, nlbs),
        dtype=np.float32,
        chunks=(batch_size, nlbs),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )
    global_start = 0
    # iterate over input test file to calculate the coefficients
    for i, filename in enumerate(test_log10flux_filenames):
        with h5py.File(filename, "r") as f_in:
            x = f_in["x"]
            mfrac = f_in["mfrac"]
            flux_fiducial = f_in["flux_fiducial"]
            test_log10flux = f_in["log10flux"]

            ntest = test_log10flux.shape[0]

            if batch_size is None:
                if test_log10flux.chunks is not None:
                    batch_size = test_log10flux.chunks[0]
                else:
                    batch_size = default_batch_size

            for start in range(0, ntest, batch_size):
                print(f"\rBatch calculate PCA coefs for test data {(global_start+start) // batch_size + 1}/{ntest_total // batch_size + ((ntest_total % batch_size)>0)}", end="")
                end = np.min((start+batch_size, ntest))

                test_log10flux_chunk = test_log10flux[start:end]
                # TODO scaling?
                coef_test = (test_log10flux_chunk - pca_mean) @ pca_modes.T

                out_start = global_start + start
                out_end = out_start + end - start

                dset_log10flux[out_start:out_end] = test_log10flux_chunk
                dset_coef[out_start:out_end] = coef_test
                dset_x[out_start:out_end] = x[start:end]
                dset_flux_fiducial[out_start:out_end] = flux_fiducial[start:end]
                dset_mfrac[out_start:out_end] = mfrac[start:end]
                dset_scale[out_start:out_end] = np.ones(end-start)  # TODO

            global_start += ntest

print("")
