"""
Run prospector emulator+MCMC on a single selected SPHEREx spectra, and save figures/parameter medians & 16, 84 percentiles

Example:
$ python emulator_mcmc.py --config configs/mcmc_config.yaml --spherex_id 1663027938116763658 -f spherex_gals.parq -n 32 -o PLOTS

Notes:
    - a yaml configuration file is required as an input in CLI mode
    - any following keywords overrides the configuration settings
    - by default, this script saves MCMC settings/metadata in "mcmc_results_{spherex_id}.yaml" 
      and MCMC key ressults in "mcmc_results_{spherex_id}.npz"
    - to check available override keywords, use -h
"""
# import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
# import emcee
# import multiprocessing
# from scipy.stats import norm, t, lognorm, loguniform
# from functools import partial
# import custom_prospector_tools as cpt
# from types import SimpleNamespace
# import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from pathlib import Path
# import corner
# from IPython.display import display, Math
# from IPython import get_ipython
# import sedpy
import argparse
# import gc
import time
from datetime import datetime
# import h5py
import libs.mcmc_libs as mlibs
import libs.data_libs as dlibs
import libs.sps_libs as slibs

# default cosmology
cosmology = Planck18


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
        '-si',
        '--spherex-id',
        type=int,
        default=None,
        metavar='<int>',
        help='SPHERExRefID, will override the id saved in config file'
    )
    parser.add_argument(
        '-f',
        '--filename',
        type=str,
        default=None,
        metavar='<str>',
        help='SPHEREx catalog name'
    )
    parser.add_argument(
        '-fl',
        '--filter-list',
        type=str,
        default=None,
        metavar='<str>',
        help='SPHEREx fiducial_filters.txt path'
    )
    parser.add_argument(
        '-e',
        '--emulator',
        type=str,
        default=None,
        metavar='<str>',
        help='Trained pytorch emulator.pt path'
    )
    parser.add_argument(
        '-nw',
        '--nwalkers',
        type=int,
        default=None,
        metavar='<int>',
        help='number of emcee ensemble walkers'
    )
    parser.add_argument(
        '-j',
        '--jitter',
        type=float,
        default=None,
        metavar='<float>',
        help='random jitter from MCMC initial position'
    )
    parser.add_argument(
        '-ns',
        '--nsteps',
        type=int,
        default=None,
        metavar='<int>',
        help='number of emcee chain steps'
    )
    parser.add_argument(
        '-d',
        '--discard',
        type=int,
        default=None,
        metavar='<int>',
        help='number of emcee steps to discard'
    )
    parser.add_argument(
        '-t',
        '--thin',
        type=int,
        default=None,
        metavar='<int>',
        help='number of steps to skip to thin the chains'
    )
    parser.add_argument(
        '-zp',
        '--zprior',
        action='store_true',
        default=None,
        help="Use photoz and sigma_photz as gaussian prior for zred"
    )
    parser.add_argument(
        '-nzp',
        '--no-zprior',
        dest="zprior",
        action='store_false',
        default=None,
        help="Don't use photoz and sigma_photz as gaussian prior for zred"
    )
    parser.add_argument(
        '-p',
        '--parallel',
        action='store_true',
        default=None,
        help='Use python multiprocessing'
    )
    parser.add_argument(
        '-np',
        '--nprocesses',
        type=int,
        default=None,
        metavar='<int>',
        help='How many Pool process if multiprocessing is True'
    )
    parser.add_argument(
        '-ss',
        '--save-sampler',
        action='store_true',
        default=None,
        help='Save sampler to a .h5 file'
    )
    parser.add_argument(
        '-sf',
        '--sampler-filename',
        type=str,
        default=None,
        metavar='<str>',
        help='If --save_sampler, save sampler backend to this filename (if "use_id", filename will be "mcmc_results_sampler_[spherex_id].h5")'
    )
    parser.add_argument(
        '-o',
        '--output-filename',
        type=str,
        default=None,
        metavar='<str>',
        help='Output report and results file name (if "use_id", filename will be "mcmc_results_[spherex_id].h5")'
    )
    parser.add_argument(
        # '-np',
        '--save-plots',
        action='store_true',
        default=None,
        help="Save plots to plots_dir"
    )
    parser.add_argument(
        # '-np',
        '--no-plots',
        dest="save_plots",
        action='store_false',
        default=None,
        help="Don't save plots to plots_dir"
    )
    parser.add_argument(
        '-pd',
        '--plots-dir',
        type=str,
        default=None,
        metavar='<str>',
        help='Output plots directory'
    )
    parser.add_argument(
        '-od',
        '--output-dir',
        type=str,
        default=None,
        metavar='<str>',
        help='Output report file and mcmc result directory")'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        '-q', 
        '--quiet', 
        dest="verbose",
        action='store_false',
        default=None,
        help='Silence the verbose outputs'
    )

    return parser.parse_args()


def main():

    start_datetime = datetime.now().isoformat(timespec='seconds')
    args = parse_args()
    config = args.config
    # read config files first
    with open(config, 'r') as file:
        yaml_config = yaml.safe_load(file)

    spherex_id = yaml_config['Files']['SPHERExRefID']
    filename = yaml_config['Files']['catalog']

    save_plots = yaml_config['Outputs']['save_plots'] if args.save_plots is None else args.save_plots
    plots_dir = yaml_config['Outputs']['plots_dir'] if args.plots_dir is None else args.plots_dir
    output_dir = yaml_config['Outputs']['output_dir'] if args.output_dir is None else args.output_dir    
    output_filename = yaml_config['Outputs']['output_filename'] if args.output_filename is None else args.output_filename
    save_sampler = yaml_config['Outputs']['save_sampler'] if args.save_sampler is None else args.save_sampler
    sampler_filename = yaml_config['Outputs']['sampler_filename'] if args.sampler_filename is None else args.sampler_filename

    verbose = yaml_config['MCMC']['verbose'] if args.verbose is None else args.verbose

    if sampler_filename == 'use_id':
        sampler_filename = f"emulator_mcmc_results_sampler_{spherex_id}.h5"
    if output_filename == 'use_id':
        output_filename = f"emulator_mcmc_results_report_{spherex_id}.h5"

    if verbose:
        print("Read input catalog...")
    cat = dlibs.catalog_dataset(filename=filename)
    cat.get_row(SPHERExRefID=spherex_id)

    if verbose:
        print("Creating MCMC instance...")
    tstart = time.time()
    emcmc_obj = mlibs.emulator_mcmc(
        config_filename=config,
        emulator=args.emulator,
        nwalkers=args.nwalkers,
        jitter=args.jitter,
        nsteps=args.nsteps,
        discard=args.discard,
        thin=args.thin,
        zprior=args.zprior,
        filters=args.filter_list,
        parallel=args.parallel,
        n_processes=args.nprocesses,
        verbose=args.verbose,
        output_dir=args.output_dir,
        save_sampler=args.save_sampler,
        sampler_filename=args.sampler_filename,
        output_filename=args.output_filename,
        save_plots=args.save_plots,
        plots_dir=args.plots_dir,
    )

    if verbose:
        print(f'SPHERExRefID:\t{spherex_id}')
        print(f'nwalkers:\t{emcmc_obj.nwalkers} \
            \n\tjitter:  \t{emcmc_obj.jitter} \
            \n\tnsteps:  \t{emcmc_obj.nsteps} \
            \n\tdiscard: \t{emcmc_obj.discard} \
            \n\tthin:    \t{emcmc_obj.thin} \
            \n\tzprior:  \t{emcmc_obj.zprior} \
            \n\tparallel:\t{emcmc_obj.parallel}')
        
        print('Run MCMC...')

    emcmc_obj.run_mcmc(
        cat.spec,
        cat.err,
        redshift=cat.zphot,
        redshift_sigma=(cat.zphot_u68-cat.zphot_l68)/2,
        results=True,
    )
    results = emcmc_obj.results
    tend = time.time()


    if verbose:
        mlibs.display_fits(theta_percentiles=results["theta_percentiles"], keys=emcmc_obj.keys)


    # get lamb_obs for plotting purpose
    filter_list = yaml_config['Files']['filters'] if args.filter_list is None else args.filter_list
    try:
        filter_list_filename = filter_list.split('/')[-1]
        filter_central_wavelengths = filter_list.replace(filter_list_filename, 'fiducial_filters_cent_waves.txt')
        lamb_obs = np.genfromtxt(filter_central_wavelengths, delimiter=' ')[:,1]
    except:
        filters = dlibs.read_filters(filter_list)
        nfilt = filters.shape[0]
        lamb_obs = np.zeros(nfilt)
        threshold = 0.1
        for i in range(nfilt):
            lamb_i = filters[i][0]
            res_i = filters[i][1] / np.max(filters[i][1])
            mask = res_i > threshold
            lamb_obs[i] = np.sum(lamb_i[mask]*res_i[mask])/np.sum(res_i[mask])



    # ------------- save plot block ---------------
    if save_plots:
        mlibs.plot_chain(emcmc_obj.full_samples,
                   ylabels=emcmc_obj.keys,
                   save=True,
                   filename=f'mcmc_results_chains_{spherex_id}.png',
                   output_dirname=plots_dir)

        mlibs.plot_corner(results['flat_samples'],
                    ylabels=emcmc_obj.keys,
                    save=True,
                    filename=f'mcmc_results_corner_{spherex_id}.png',
                    output_dirname=plots_dir
                    )
        
        # get model and SFH percentiles for sed_sfh plot
        if emcmc_obj.sfh_type == 'continuity_sfh':
            qs_agelims, qs_agebins_all_sfrs = slibs.continuity_sfh_percentiles_steps(
                results['flat_samples'],
                zred_idx=emcmc_obj.zred_index,
                logmass_idx=emcmc_obj.logmass_index,
                logsfr_ratios_idx=emcmc_obj.logsfr_ratios_index,
                n_transition=50,
                transition_start_idx=3,
                percentiles=[16,50,84]
            )
            mlibs.plot_sed_sfh(lamb_obs,
                         cat.spec,
                         cat.err,
                         lamb_model=results["lbs_med"],
                         spec_model=results['flux_med'],
                         agelims_model=results['agebins_med'],
                         sfrsteps_model=results['sfrs_med'],
                         qs_agelims=qs_agelims,
                         qs_sfrsteps=qs_agebins_all_sfrs,
                         external_phots=cat.external_phots,
                         save=True,
                         filename=f"mcmc_results_sed_sfh_{spherex_id}.png",
                         output_dirname=plots_dir,
                         title_kwargs={
                            'spherex_id': spherex_id,
                            'zspec': cat.zspec,
                            'zphot': cat.zphot,
                            'zphot_u68': cat.zphot_u68,
                            'zphot_l68': cat.zphot_l68,
                            'zmcmc_med': results["theta_percentiles"][1, emcmc_obj.zred_index],
                            'zmcmc_16': results["theta_percentiles"][0, emcmc_obj.zred_index],
                            'zmcmc_84': results["theta_percentiles"][2, emcmc_obj.zred_index],
                            'frac102': cat.frac102,
                            'fontsize': 9,
                            }
                         )

    # ------------- save plot block ---------------


    elapsed_time = tend - tstart
    end_datetime = datetime.now().isoformat(timespec='seconds')

    # saving data to .h5 file
    dlibs.save_h5_results(results, 
                    output_filename=output_filename, 
                    output_dir=output_dir,
                    metadata={
                        'keys': results['keys'],
                        'start_time': start_datetime,
                        'end_time': end_datetime,
                        'SPHERExRefID': spherex_id,
                        'zspec': cat.zspec,
                        'zphot': cat.zphot,
                        'zphot_u68': cat.zphot_u68,
                        'zphot_l68': cat.zphot_l68,
                        'zphot_std': cat.zphot_std,
                        'nwalkers': emcmc_obj.nwalkers,
                        'jitter': emcmc_obj.jitter,
                        'nsteps': emcmc_obj.nsteps,
                        'discard': emcmc_obj.discard,
                        'thin': emcmc_obj.thin,
                        'zprior': emcmc_obj.zprior,
                        'parallel': emcmc_obj.parallel
                        }
                    )

    print(f'MCMC runtime = {elapsed_time} seconds')

if __name__ == '__main__':
    main()
