# Introduction

This repo contains collection of tools/libraries/scripts related to ASU's Prospector emulator for SPHEREx project.

Notable python packages:
- h5py
- emcee
- astropy
- numba
- torch
- corner.py
- prospector and FSPS, python-fsps (ONLY if you want to run `data_creation.py` or if you want to compare to true spectra)

# Usages

## Data Creation

Noted: `data_creation.py` is now deprecated.  

Data creation is a 2 step process now with two separate scripts.  
First, to make train/valid/test high resolutiton spectra from `Prospector` (requires installing prospector, FSPS, python-fsps),  
modify `configs/create_spectra.yaml` for prospector related settings, number of spectra, priors etc, and run


> $ python create_train_valid_test_spectra.py -c configs/create_spectra.yaml

This will create 3 large `.h5` files containing the native rest frame spectra.

Second, `make_x_coef_files.py` and its config `configs/make_x_coef_files.yaml` compile the files from above step  
to training required formats, also `.h5` files.  


> $ python make_x_coef_files.py -c configs/make_x_coef_files.yaml

This script accepts multiple file inputs and will combine them together, creating `train/valid/test_x_coef_mfrac_flux_Ndat.h5` files.

## Train an emulator

Once you have a set of train/test data (or train/valid/test), edit `configs/train_config.yaml` for training related settings, then


> $ python train_emulator.py -c configs/train_config.yaml


You can use CLI arguments to override the config file; to check available options, run  
`python train_emulator.py -h`

This code creates an `output_dir` that contains all the outputs from the script.

## Run emulator MCMC

If you have an L4 parquet file ready, and know the `SPHERExRefID`, you can set it up in `configs/mcmc_config.yaml`, then 


> $ python emulator_mcmc.py -c configs/mcmc_config.yaml


There are also many CLI flags you can use to override the config file. To check what are available use `-h`. 

Emulator MCMC routine can be custom built in jupyter notebooks or python scripts.  
Check `tutorials/create_mcmc_routines.ipynb` to see how do do so.  


