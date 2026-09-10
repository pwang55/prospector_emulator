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

To make train/valid/test data from `Prospector` (requires installing prospector, FSPS, python-fsps),  
modify settings in `configs/data_creation.yaml`, including allowed parameters, prior settings and nsamples, then  

```
$ python data_creation.py -c configs/data_creation_config.yaml
```

## Train an emulator

Once you have a set of train/test data (or train/valid/test), edit `configs/train_config.yaml` for training related settings, then

```
$ python train_emulator.py -c configs/train_config.yaml
```

You can use CLI arguments to override the config file; to check available options, run  
`python train_emulator.py -h`

This code creates an `output_dir` that contains all the outputs from the script.

## 

