"""
================================================================================
Functions to compute plasma parameters from MMS data.

Functions:
- compute_beta_profiles: Compute the plasma beta profiles.

Author: Regis John
Created: 2026-06-23
================================================================================
"""

import os
import h5py
import numpy as np
from pyspedas.projects import mms
from pyspedas import tplot_rename, get_data, fac_matrix_make
from fipcore.utils.helper_utils import downsample_cad, compute_beta_par, \
    compute_beta_perp, compute_beta_scalar
import fipcore.utils.helper_utils as hutil
import fipcore.utils.coord_utils as coord
import fipcore.utils.io_utils as iout
import fipcore.analysis.jdotE_proc as je
import fipcore.analysis.jvec_proc as jvec


def compute_beta_profiles(trange, species='e', probe='1', data_rate='brst', 
        level='l2', get_support_data=True, no_update=True):
    """
    Compute the plasma beta profiles for a given species and data.

    Parameters:
    - trange (list of str): [start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - species (str): Species of particle to analyze. Default is 'e'.
    - probe (str): Probe number to analyze. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - level (str): Level of the data. Default is 'l2'.
    - get_support_data (bool): If True, load in support data. Default is True.
    - no_update (bool): If True, the data will not be updated from the server. 
        Default is True.

    Returns:
    - beta_par (numpy array): Parallel plasma beta profile.
    - beta_perp (numpy array): Perpendicular plasma beta profile.
    - beta_scalar (numpy array): Scalar plasma beta profile.
    """
    

    spec_tag = 'des' if species == 'e' else 'dis'

    # Variables to load:
    den_var = f'mms{probe}_{spec_tag}_numberdensity_{data_rate}'
    t_para_var = f'mms{probe}_{spec_tag}_temppara_{data_rate}'
    t_perp_var = f'mms{probe}_{spec_tag}_tempperp_{data_rate}'
    bvec_var = f'mms{probe}_fgm_b_gse_{data_rate}_{level}'

    # Loading in density and temperature data from fpi
    mms.fpi(trange= trange, probe=probe, data_rate=data_rate, level=level,
        datatype=f'{spec_tag}-moms', 
        time_clip=True, varnames=[den_var, t_para_var, t_perp_var], 
        get_support_data=get_support_data, no_update=no_update)

    # Loading in magnetic field data from fgm
    mms.fgm(trange=trange, probe=probe, data_rate=data_rate, level=level,
        varnames=bvec_var, time_clip=True, 
        get_support_data=get_support_data, no_update=no_update)

    # Renaming the tplot variables
    tplot_rename(den_var, 'den')
    tplot_rename(t_para_var, 't_para')
    tplot_rename(t_perp_var, 't_perp')
    tplot_rename(bvec_var, 'bvec')

    # Downsampling the fgm data to the fpi cadence
    downsample_cad('bvec', 'den', trange, newname='bvec_ds')

    # Extracting data from tplot variables
    ntime, den = get_data('den')
    _, t_para = get_data('t_para')
    _, t_perp = get_data('t_perp')
    _, bvec = get_data('bvec_ds') # bvec is [Bx, By, Bz, Bmag]
    bmag = bvec[:, 3]

    # Calling the helper functions to compute the individual beta profiles
    beta_par = compute_beta_par(den, t_para, bmag)
    beta_perp = compute_beta_perp(den, t_perp, bmag)
    beta_scalar = compute_beta_scalar(den, t_para, t_perp, bmag)
    
    return beta_par, beta_perp, beta_scalar, ntime


def compute_jvec_all(trange, species='e', probe='1', data_rate='brst', bin_width_frac=0.25, 
    level='l2', subtract_f0=False, no_update=True):
    """
    Function to compute MMS current density in various coordinates and 
    save to an existing .h5 file.

    Parameters:
    - trange (list of str): [start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - species (str): Species of particle to analyze. Default is 'e'.
    - probe (str): Probe number to analyze. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - bin_width_frac (float): Fraction of the thermal velocity used to 
        determine the bin width. Default is 0.25.
    - level (str): Level of the data. Default is 'l2'.
    - subtract_f0 (bool): if True, use the df suffix in the filename. Default is False.
    - no_update (bool): if True, do not download new data. Default is True. Turn 
        it to False to download new data if it is not available locally.

    Returns:
    - jvec_gse (ndarray): Current density vector in GSE coordinates.
    - jvec_fac (ndarray): Current density vector in FAC coordinates.
    - jvec_lmn (ndarray): Current density vector in LMN coordinates.
    """

    # Computing current density vector in GSE coordinates
    jvec_gse = jvec.compute_jvec(trange, probe=probe, data_rate=data_rate, 
                level=level)

    # Read in magnetic field and electron bulf flow data
    mms.fpi(trange= trange, probe=probe, data_rate=data_rate, level=level,
    datatype=['des-moms'], time_clip=True, varnames=f'mms{probe}_des_bulkv_gse_brst', 
    get_support_data=True, no_update=no_update)
    mms.fgm(trange=trange, probe=probe, data_rate=data_rate, level=level,
        varnames=f'mms{probe}_fgm_b_gse_brst_l2', time_clip=True, 
        get_support_data=True, no_update=no_update)

    tplot_rename(f'mms{probe}_des_bulkv_gse_brst', 'bulk_ve_gse')
    tplot_rename(f'mms{probe}_fgm_b_gse_brst_l2_bvec', 'bvec_gse')

    # Downsample magnetic field to electron bulk flow cadence
    hutil.downsample_cad('bvec_gse', 'bulk_ve_gse', trange, newname='bvec_gse_dwn')

    # Rotate to FAC coordinates
    fac_mat_name = 'fac_mat_var'
    fac_matrix_make(mag_var_name='bvec_gse_dwn', other_dim='Xgse', 
                             newname=fac_mat_name) # Creating FAC matrix

    jvec_fac = jvec.jvec_to_fac(jvec_gse, fac_mat_name)       

    # Rotate to LMN coordinates
    lmn_mat_name = 'lmn_mat_var'
    coord.lmn_matrix_make('bvec_gse_dwn', trange, newname=lmn_mat_name)
    jvec_lmn = jvec.jvec_to_lmn(jvec_gse, lmn_mat_name)

    return jvec_gse, jvec_fac, jvec_lmn


def compute_jdotE_all(trange, species='e', probe='1', data_rate='brst', 
    bin_width_frac=0.25, subtract_f0=False):
    """
    Computes the J·E dot product for a given species and time range.
    
    Parameters:
    - trange (list of str): [start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - species (str): Species of particle to analyze. Default is 'e'.
    - probe (str): Probe number to analyze. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - bin_width_frac (float): Fraction of the thermal velocity used to 
        determine the bin width. Default is 0.25.
    - subtract_f0 (bool): if True, use the df suffix in the filename. Default is False.
    
    Returns:
    - jdotE_gse (ndarray): J.E vector in GSE coordinates.
    - jdotE_fac (ndarray): J.E vector in FAC coordinates.
    - jdotE_lmn (ndarray): J.E vector in LMN coordinates.
    """

    # Data path setup
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    type_tag = 'df' if subtract_f0 else 'f'
    hfile_pref = f'mms{probe}_{data_rate}_{type_tag}_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, iout.mms_name_make(hfile_pref, trange[0], trange[1]))
    
    # Read in required data
    with h5py.File(hfile, "r") as f:
        jvec_gse = f["gse"]["jvec"][:]
        jvec_fac = f["fac"]["jvec"][:]
        jvec_lmn = f["lmn"]["jvec"][:]
        evec_gse = f["gse"]["evec_lor"][:]
        evec_fac = f["fac"]["evec_lor"][:]
        evec_lmn = f["lmn"]["evec_lor"][:]

    # GSE
    _, jdotE_gse, jdotE_gse0, jdotE_gse1, jdotE_gse2 = je.compute_jdotE(jvec_gse, evec_gse)
    jdotE_gse = np.column_stack([jdotE_gse, jdotE_gse0, jdotE_gse1, jdotE_gse2])
    
    # FAC
    _, jdotE_fac, jdotE_fac0, jdotE_fac1, jdotE_fac2 = je.compute_jdotE(jvec_fac, evec_fac)
    jdotE_fac = np.column_stack([jdotE_fac, jdotE_fac0, jdotE_fac1, jdotE_fac2])
    
    # LMN
    _, jdotE_lmn, jdotE_lmn0, jdotE_lmn1, jdotE_lmn2 = je.compute_jdotE(jvec_lmn, evec_lmn)
    jdotE_lmn = np.column_stack([jdotE_lmn, jdotE_lmn0, jdotE_lmn1, jdotE_lmn2])

    return jdotE_gse, jdotE_fac, jdotE_lmn