import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import sedpy
import h5py
import pyarrow.dataset as ds


# ===========================================
# Catalog and data handling
# ===========================================

filters_ls = sedpy.observate.load_filters(['bass_g', 'bass_r', 'mzls_z'])
filters_ps1 = sedpy.observate.load_filters(['panstarrs_g', 'panstarrs_r', 'panstarrs_i', 'panstarrs_z', 'panstarrs_y'])
filters_2mass = sedpy.observate.load_filters(['twomass_J', 'twomass_H', 'twomass_Ks'])
filters_wise = sedpy.observate.load_filters(['wise_w1', 'wise_w2', 'wise_w3', 'wise_w4'])

lbs_ls = np.array([filters_ls[i].wave_effective for i in range(len(filters_ls))])/10000
lbs_ps1 = np.array([filters_ps1[i].wave_effective for i in range(len(filters_ps1))])/10000
lbs_2mass = np.array([filters_2mass[i].wave_effective for i in range(len(filters_2mass))])/10000
lbs_wise = np.array([filters_wise[i].wave_effective for i in range(len(filters_wise))])/10000
external_phot_lbs = {
    'LS': lbs_ls,
    'PS1': lbs_ps1,
    '2MASS': lbs_2mass,
    'WISE': lbs_wise
}

# Reference catalog external photometry column names
LS_cols = ['LS_g', 'LS_r', 'LS_z']
PS1_cols = ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']
twomass_cols = ['2MASS_J', '2MASS_H', '2MASS_Ks']
wise_cols = ['WISE_W1', 'WISE_W2', 'WISE_W3', 'WISE_W4']
all_refcat_surveys = [LS_cols, PS1_cols, twomass_cols, wise_cols]


class catalog:
    """
    Read input SPHEREx data format and convert to useful arrays

    Attributes
    ----------
    dat : pandas.DataFrame
        Full input data table
    spherex_ids : ndarray
        All SPHEREx unique reference catalog ids in this table
    zspecs : ndarray
        All spectroscopic redshifts
    zphots : ndarray
        All photometric redshifts from L4 catalog
    zphots_u68 : ndarray
        All upper 1-sigma photo-zs
    zphots_l68 : ndarray
        All lower 1-sigma photo-zs
    zphots_std : ndarray
        All standard deviation of photo-z pdfs
    spectra : ndarray of shape (nrows, nfilts)
        All spectra in this table
    error : ndarray of shape (nrows, nfilts)
        All uncertainty in this table
    frac102 : ndarray
        All frac102 values in this table

    Methods
    -------
    get_external_phots(SPHERExRefID, idx)
        Grab all external photometry for a given source, whether with SPHERExRefID or row index
        Returns a nested dictionary with 'LS', 'PS1', '2MASS', 'WISE', 
        each with keys 'wavelength', 'flux', 'flux_error'
    """
    def __init__(self, filename=''):
        self.dat = pd.read_parquet(filename)
        self.spherex_ids = np.array(self.dat['SPHERExRefID'])
        self.zspecs = np.array(self.dat['z_specz'])
        self.zphots = np.array(self.dat['z_best_gals'])
        self.zphots_u68 = np.array(self.dat['z_err_u68_gals'])
        self.zphots_l68 = np.array(self.dat['z_err_l68_gals'])
        self.zphots_std = np.array(self.dat['z_err_std_gals'])
        self.spectra = np.stack(self.dat['flux_dered_fiducial'])
        self.error = np.stack(self.dat['flux_err_dered_fiducial'])
        self.frac102 = np.array(self.dat['frac_sampled_102'])

    def get_external_phots(self, SPHERExRefID=None, idx=None):
        if SPHERExRefID is not None:
            idx = np.where(self.spherex_ids == SPHERExRefID)[0][0]

        external_phots = {}
        for i, survey_cols in enumerate(all_refcat_surveys):
            ndat = len(survey_cols)
            survey_name = survey_cols[0].split('_')[0]
            wavelength = external_phot_lbs[survey_name]
            flux = np.zeros(ndat)
            flux_error = np.zeros(ndat)
            for j, colname in enumerate(survey_cols):
                flux[j] = self.dat[colname].iloc[idx]
                flux_error[j] = self.dat[colname+'_error'].iloc[idx]
            external_phots[survey_name] = {
                                            'wavelength':wavelength,
                                            'flux': flux,
                                            'flux_error': flux_error
                                            }
        return external_phots

class catalog_dataset:
    """
    Use pyarrow.dataset to access SPHEREx L4 Catalog

    Parameters
    ----------
    filename : str
        L4 parquet catalog filename
    
    Methods
    -------
    get_row(SPHERExRefID) : 
        Get useful information from a single row with SPHERExRefID
        Creates

    """
    def __init__(self, filename):
        self.dataset = ds.dataset(filename, format='parquet')

    def get_row(self, SPHERExRefID):
        ds_filters = ds.field('SPHERExRefID') == SPHERExRefID
        tab = self.dataset.to_table(filter=ds_filters)
        self.zspec = tab['z_specz'][0].as_py()
        self.zphot = tab['z_best_gals'][0].as_py()
        self.zphot_u68 = tab['z_err_u68_gals'][0].as_py()
        self.zphot_l68 = tab['z_err_l68_gals'][0].as_py()
        self.zphot_std = tab['z_err_std_gals'][0].as_py()
        self.frac102 = tab['frac_sampled_102'][0].as_py()
        self.spec = tab['flux_dered_fiducial'].to_numpy()[0]
        self.err = tab['flux_err_dered_fiducial'].to_numpy()[0]

        self.external_phots = {}
        for i, survey_cols in enumerate(all_refcat_surveys):
            ndat = len(survey_cols)
            survey_name = survey_cols[0].split('_')[0]
            wavelength = external_phot_lbs[survey_name]
            flux = np.zeros(ndat)
            flux_error = np.zeros(ndat)
            for j, colname in enumerate(survey_cols):
                flux[j] = tab[colname][0].as_py()
                flux_error[j] = tab[colname+'_error'][0].as_py()
            self.external_phots[survey_name] = {
                                            'wavelength':wavelength,
                                            'flux': flux,
                                            'flux_error': flux_error
                                            }


def save_h5_results(mcmc_results, 
                    output_filename='mcmc_results_report.h5', 
                    output_dir='.',
                    metadata=None):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compressed_cols = ['flat_samples', 'flat_mfracs', 'med_lbs', 'med_spectra']
    
    with h5py.File(Path(output_dir) / output_filename, 'w') as f:
        for i, key in enumerate(mcmc_results):
            if type(mcmc_results[key]) is not dict:
                if key not in compressed_cols:
                    dset = f.create_dataset(key, data=mcmc_results[key])
                else:
                    dset = f.create_dataset(key, data=mcmc_results[key], compression='gzip')
            else:
                group = f.create_group(key)
                for j, keyj in enumerate(mcmc_results[key]):
                    dset2 = group.create_dataset(keyj, data=mcmc_results[key][keyj])

        if metadata is not None:
            for i, key in enumerate(metadata):
                f.attrs[key] = metadata[key]



def read_h5_results(h5file):
    """
    Convenience function to read back the saved mcmc_results_report.h5 file
    Returns the mcmc_results dictionary, and add "attribute" key to the dictionary that includes metadata for the MCMC run
    """
    mcmc_results = {}
    with h5py.File(h5file, 'r') as f:
        for i, key in enumerate(f):
            if key != 'med_myparam':
                mcmc_results[key] = f[key][()]
            else:
                mcmc_results[key] = {}
                for j, keyj in enumerate(f['med_myparam']):
                    mcmc_results[key][keyj] = f['med_myparam'][keyj][()]
        mcmc_results['attributes'] = {}
        for i, key in enumerate(f.attrs) :
            mcmc_results['attributes'][key] = f.attrs[key]
    return mcmc_results

# Read SPHEREx filters
def read_filters(filter_list, half_length=105, return_lamb_obs=False, response_threshold=0.1):
    '''
    Read SPHEREx official filters and return ndarray of (nfilt, 2, 2*half_length)
    
    Parameters
    ----------
    filter_list : str
        Full path to the filter_list.txt that contains all the filter names in order
    half_length : int, default=105
        Half size of each filter's length.
        Currently, official SPHEREx filters don't have the same size; half_length of 105 is a good middle ground for 306 bands
        to cut off unimportant outside parts while preserving response to <3% accuracy.
        If half_length*2 > existing wavelength grid, fill longer wavelength side with zeros
    return_lamb_obs : bool, default=False
        If True, also returns the lamb_obs by finding 'fiducial_filters_cent_waves.txt' at the same directory as the filter_list
        If the cent_wave file can't be found, calculate by weight average of normalized response>0.1
        
    Returns
    -------
    filters : ndarray of shape (nfilt, 2, 2*half_length)
        Full array of all the filters' wavelength coverage (micron) and response
        For example, n_filter=i has wavelength filters[i][0] and response filters[i][1]
    lamb_obs : ndarray of shape (nfilt, )
        Only return when return_lamb_obs=True
    '''
    filter_dir = Path(filter_list).parent
    filter_names = np.loadtxt(filter_list, dtype=str)
    Nf = filter_names.shape[0]
    filters = np.zeros((Nf, 2, half_length*2))

    for i in range(Nf):
        filter_path_name = filter_dir / Path(filter_names[i])
        filt_i = np.loadtxt(filter_path_name)
        wavelength_i = filt_i[:,0] * 1e-4   # convert from AA to micron
        response_i = filt_i[:,1]
        arg_peak = np.argmax(response_i)
        if len(wavelength_i) < half_length*2:
            wavelength_i1 = np.zeros(half_length*2)
            response_i1 = np.zeros(half_length*2)
            wavelength_i1[:len(wavelength_i)] = wavelength_i
            dw = wavelength_i[-1] - wavelength_i[-2]
            wavelength_i1[len(wavelength_i):] = wavelength_i[-1] + dw * np.arange(1, len(wavelength_i1)-len(wavelength_i)+1)
            response_i1[:len(response_i)] = response_i
        else:
            if arg_peak - half_length < 0:
                istart = 0
                ifinish = half_length*2
            elif arg_peak + half_length > len(wavelength_i):
                ifinish = len(wavelength_i)
                istart = ifinish - half_length*2
            else:
                istart = arg_peak - half_length
                ifinish = arg_peak + half_length
            wavelength_i1 = wavelength_i[istart:ifinish]
            response_i1 = response_i[istart:ifinish]
        tot_response_i1 = np.trapezoid(response_i1, wavelength_i1)
        response_i1 = response_i1 / tot_response_i1    # divide by total response now so that when convolving SED with filters no normalization is needed
        filters[i][0] = wavelength_i1
        filters[i][1] = response_i1
        # filters.append((wavelength_i1, response_i1))
    if return_lamb_obs:
        try:
            filter_central_wavelengths = filter_list.replace('fiducial_filters.txt', 'fiducial_filters_cent_waves.txt')
            lamb_obs = np.genfromtxt(filter_central_wavelengths)[:,1]
        except:
            lamb_obs = np.zeros(Nf)
            for i in range(Nf):
                wav_i = filters[i][0]
                res_i = filters[i][1] / np.max(filters[i][1])
                mask = res_i > response_threshold
                lamb_obs[i] = np.sum(wav_i[mask] * res_i[mask]) / np.sum(res_i[mask])
        return filters, lamb_obs
    else:
        return filters
    
@njit(fastmath=True)
def convolve_filter(wl, flux, filters=None):
    '''
    Convolve spectra with filters read from read_filters()
    filters need to have shape (nfilt, 2, ns)

    Parameters
    ----------
    wl : ndarray of shape n
        wavelength points in micron
    flux : ndarray of shape n
        flux in f_nu
    filters : ndarray of shape (nfilt, 2, ns)
        filters array from read_filters()

    Returns
    -------
    flux_conv : ndarray of shape (nfilt, )
        Convolved fluxes in each passband
    '''
    if filters is not None:
        Nf = len(filters)
        flux_conv = np.zeros(Nf)
        for i in range(Nf):
            lb = filters[i][0]
            ftrans = filters[i][1]
            f_interp = np.interp(lb, wl, flux)
            fnu_i = np.trapezoid(f_interp*ftrans, lb)
            flux_conv[i] = fnu_i
    else:
        flux_conv = flux
    return flux_conv

