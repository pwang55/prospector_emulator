# Introduction

This repo contains collection of tools/libraries/scripts related to ASU's Prospector emulator for SPHEREx project.

Notable python packages:
- h5py
- emcee
- astropy
- numba
- torch
- corner.py
- prospector and FSPS, python-fsps (ONLY if you want to run `create_train_valid_test_spectra.py` or if you want to compare to true spectra)

All the relevant libraries & function sit in `prospector_emulator_libs`, which after installation you can import from.  
There are 3 major categories of scripts, `create_data/`, `train_emulator/` and `run_mcmc/`, with scripts in each folders.  

---

# Installation

In the repo folder, do
```
pip install -e .
```
This allows your environment to find `prospector_emulator_libs` anywhere.

---

# Usages

## 1. Data Creation

Data creation is a 2 step process now with two separate scripts.  
First, to make train/valid/test high resolutiton spectra from `Prospector` (requires installing prospector, FSPS, python-fsps),  
modify `configs/create_spectra.yaml` for prospector related settings, number of spectra, priors etc, and run

```
$ python create_data/create_train_valid_test_spectra.py -c configs/create_spectra.yaml
```
This will create 3 large `.h5` files containing the native rest frame spectra.

Second, `make_x_coef_files.py` and its config `configs/make_x_coef_files.yaml` compile the files from above step  
to training required formats, also `.h5` files.  

```
$ python create_data/make_x_coef_files.py -c configs/make_x_coef_files.yaml
```
This script accepts multiple file inputs and will combine them together, creating `train/valid/test_x_coef_mfrac_flux_Ndat.h5` files.

---

## 2. Train an emulator

### a. PCA high-resolution SED emulator

Once you have a set of train/test data (or train/valid/test), edit `configs/train_pca_emulator.yaml` for training related settings, then
```
$ python train_emulators/train_pca_emulator.py -c configs/train_pca_emulator.yaml
```
You can use CLI arguments to override the config file; to check available options, run  
`python train_emulators/train_pca_emulator.py -h`

This code creates an `output_dir` that contains all the outputs from the script.

### b. Direct fiducial flux emulator
```
$ python train_emulators/train_flux_emulator.py -c configs/train_flux_emulator.yaml
```
You can use CLI arguments to override the config file; to check available options, run  
`python train_emulators/train_flux_emulator.py -h`

This code creates an `output_dir` that contains all the outputs from the script.

---

## 3. Run emulator MCMC

If you have an L4 parquet file ready, and know the `SPHERExRefID`, you can set it up in `configs/pca_mcmc_config.yaml` or `configs/flux_mcmc_config.yaml`, then 

```
$ python run_mcmc/run_pca_emulator_mcmc.py -c configs/pca_mcmc_config.yaml  
```
or  
```
$ python run_mcmc/run_flux_emulator_mcmc.py -c configs/flux_mcmc_config.yaml
```
There are also many CLI flags you can use to override the config file. To check what are available use `-h`. 

Emulator MCMC routine can be custom built in jupyter notebooks or python scripts.  
Check `tutorials/create_mcmc_routines.ipynb` to see how do do so.  


