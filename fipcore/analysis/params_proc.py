"""
================================================================================
Functions to compute plasma parameters from MMS data.

Functions:
- compute_beta_profiles: Compute the plasma beta profiles.
- compute_upstream_params: Average plasma parameters over an upstream window.

Author: Regis John
Created: 2026-06-23
================================================================================
"""

import os
import h5py
import numpy as np
from pyspedas.projects import mms
from pyspedas import tplot_rename, get_data, del_data, fac_matrix_make
import scipy.constants as const
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


def compute_upstream_params(trange_upstream, probe='1', data_rate='brst', 
    level='l2', species='e', bin_width_frac=0.25, get_support_data=True, 
    no_update=True):
    """
    Average plasma parameters over an upstream window and print them.

    Gives single values averaged over the window (e.g. the d_e, d_i used to
    normalize lengths). Nothing is written to a file. For time series over the
    event written to the .h5 file, use fpc_mrx_main.fpc_mrx_plasma_params instead.

    Parameters:
    - trange_upstream (list of str): [start time, end time] of the upstream window 
        in the format ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss'].
    - probe (str): Probe number. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - level (str): Level of the data. Default is 'l2'.
    - species (str): Not used. Default is 'e'.
    - bin_width_frac (float): Not used. Default is 0.25.
    - get_support_data (bool): If True, load in support data. Default is True.
    - no_update (bool): If True, the data will not be updated from the server. 
        Default is True.

    Returns:
    - params (dict): Window averages and derived parameters: b_vec, bmag (nT); 
        den_e, den_i (cm^-3); te, ti (eV); ti_te_ratio, beta_e, beta_i, bg_ratio; 
        ue_gse, ui_gse, vA, vti, vte, c_s (km/s); rho_i, rho_e, d_i, d_e, rho_s (km); 
        lambda_D (m); t_ci, t_ce, t_pi, t_pe (s).
    """
    # Physical constants (bound here so they are not exported by `import *`)
    m_p, m_e, e, mu_0, epsilon_0, c = (const.m_p, const.m_e, const.e, const.mu_0,
                                       const.epsilon_0, const.c)

    # Clean workspace
    del_data('*')

    # 1. Load FPI and FGM data for the upstream window
    fpi_varlist = [
    f'mms{probe}_des_numberdensity_brst', f'mms{probe}_dis_numberdensity_brst', 
    f'mms{probe}_des_temppara_brst', f'mms{probe}_dis_temppara_brst', 
    f'mms{probe}_des_tempperp_brst', f'mms{probe}_dis_tempperp_brst', 
    f'mms{probe}_des_bulkv_gse_brst', f'mms{probe}_dis_bulkv_gse_brst'
    ]
    
    mms.fpi(trange=trange_upstream, probe=probe, data_rate=data_rate, level=level, 
        datatype=['des-moms', 'dis-moms'], time_clip=True, 
        varnames=fpi_varlist, get_support_data=get_support_data, no_update=no_update)

    mms.fgm(trange=trange_upstream, probe=probe, data_rate=data_rate, level=level,
        varnames=f'mms{probe}_fgm_b_gse_brst_l2', time_clip=True, 
        get_support_data=get_support_data, no_update=no_update)

    # Extract the numpy arrays
    _, b_vec = get_data(f'mms{probe}_fgm_b_gse_brst_l2_bvec')
    _, bmag = get_data(f'mms{probe}_fgm_b_gse_brst_l2_btot')
    _, den_e = get_data(f'mms{probe}_des_numberdensity_brst')
    _, den_i = get_data(f'mms{probe}_dis_numberdensity_brst')
    _, te_para = get_data(f'mms{probe}_des_temppara_brst')
    _, ti_para = get_data(f'mms{probe}_dis_temppara_brst')
    _, te_perp = get_data(f'mms{probe}_des_tempperp_brst')
    _, ti_perp = get_data(f'mms{probe}_dis_tempperp_brst')
    _, ue_gse = get_data(f'mms{probe}_des_bulkv_gse_brst')
    _, ui_gse = get_data(f'mms{probe}_dis_bulkv_gse_brst')

    # Guide Field Ratio
    bg_ratio = 0.5

    # Computing averages
    b_vec_mean = np.nanmean(b_vec, axis=0)   # shape: (3,) in nT
    bmag_mean = np.nanmean(bmag)             # nT

    den_e_mean = np.nanmean(den_e)           # cm^-3
    den_i_mean = np.nanmean(den_i)           # cm^-3

    te_para_mean = np.nanmean(te_para)       # eV
    te_perp_mean = np.nanmean(te_perp)       # eV
    te_mean = (te_para_mean + 2.0 * te_perp_mean) / 3.0

    ti_para_mean = np.nanmean(ti_para)       # eV
    ti_perp_mean = np.nanmean(ti_perp)       # eV
    ti_mean = (ti_para_mean + 2.0 * ti_perp_mean) / 3.0

    ue_gse_mean = np.nanmean(ue_gse, axis=0) # shape: (3,) in km/s
    ui_gse_mean = np.nanmean(ui_gse, axis=0) # shape: (3,) in km/s

    # Convert mean values to SI units
    B0_T = bmag_mean * 1e-9                  # T
    n_e_m3 = den_e_mean * 1e6                # m^-3
    n_i_m3 = den_i_mean * 1e6                # m^-3
    Te_J = te_mean * e                       # J
    Ti_J = ti_mean * e                       # J

    # 1. Temperature Ratio & Plasma Betas
    ti_te_ratio = ti_mean / te_mean
    beta_e = (2.0 * mu_0 * n_e_m3 * Te_J) / (B0_T**2)
    beta_i = (2.0 * mu_0 * n_i_m3 * Ti_J) / (B0_T**2)

    # 2. Speeds in km/s
    vA = (B0_T / np.sqrt(mu_0 * n_i_m3 * m_p)) / 1e3    # km/s
    vti = (np.sqrt(2.0 * Ti_J / m_p)) / 1e3             # km/s
    vte = (np.sqrt(2.0 * Te_J / m_e)) / 1e3             # km/s
    # Ion sound speed (using electron temperature and proton mass)
    c_s = (np.sqrt(Te_J / m_p)) / 1e3                   # km/s

    # 3. Angular frequencies (rad/s)
    omega_ci = (e * B0_T) / m_p
    omega_ce = (e * B0_T) / m_e
    omega_pi = np.sqrt((n_i_m3 * e**2) / (m_p * epsilon_0))
    omega_pe = np.sqrt((n_e_m3 * e**2) / (m_e * epsilon_0))
    # Frequencies in Hz
    f_ci = omega_ci / (2.0 * np.pi)          # Hz
    f_ce = omega_ce / (2.0 * np.pi)          # Hz
    f_pi = omega_pi / (2.0 * np.pi)          # Hz
    f_pe = omega_pe / (2.0 * np.pi)          # Hz

    # Characteristic Timescales / Periods (seconds)
    t_ci = 1.0 / f_ci  # Ion cyclotron period (s)
    t_ce = 1.0 / f_ce  # Electron cyclotron period (s)
    t_pi = 1.0 / f_pi  # Ion plasma timescale (s)
    t_pe = 1.0 / f_pe  # Electron plasma timescale (s)


    # 4. Length Scales (km)
    rho_i = (vti * 1e3 / omega_ci) / 1e3     # km
    rho_e = (vte * 1e3 / omega_ce) / 1e3     # km
    d_i = (c / omega_pi) / 1e3               # km
    d_e = (c / omega_pe) / 1e3               # km
    # Ion sound gyroradius
    rho_s = (c_s * 1e3 / omega_ci) / 1e3     # km

    lambda_D = np.sqrt((epsilon_0 * Te_J) / (n_e_m3 * e**2)) # m

    # --- Print Results ---
    print("\n" + "="*70)
    print("UPSTREAM PLASMA PARAMETERS")
    print("="*70)

    print("\n--- Raw Plasma Parameters ---")
    print(f"Average B_vec (nT): {b_vec_mean}")
    print(f"Average |B| (nT): {bmag_mean:.3f}")
    print(f"Average Electron Density (cm^-3): {den_e_mean:.3f}")
    print(f"Average Ion Density (cm^-3): {den_i_mean:.3f}")
    print(f"Average Te (para, perp, scalar) (eV): {te_para_mean:.3f}, {te_perp_mean:.3f}, {te_mean:.3f}")
    print(f"Average Ti (para, perp, scalar) (eV): {ti_para_mean:.3f}, {ti_perp_mean:.3f}, {ti_mean:.3f}")
    
    print("\n--- Derived Plasma Parameters ---")
    print(f"Beta Electron (beta_e): {beta_e:.4f}")
    print(f"Beta Ion (beta_i): {beta_i:.4f}")
    print(f"Ti / Te Ratio: {ti_te_ratio:.3f}")
    print(f"b_g = {bg_ratio:.3f}")

    print("\n--- Characteristic Velocities ---")
    print(f"Average Ue_gse (km/s): {ue_gse_mean}")
    print(f"Average Ui_gse (km/s): {ui_gse_mean}")
    print(f"Alfvén Speed (vA): {vA:.2f} km/s")
    print(f"Ion Thermal Speed (vti): {vti:.2f} km/s")
    print(f"Electron Thermal Speed (vte): {vte:.2f} km/s")
    print(f"Ion Sound Speed (c_s): {c_s:.2f} km/s")

    print("\n--- Characteristic Lengths ---")
    print(f"Ion Larmor Radius (rho_i): {rho_i:.2f} km")
    print(f"Electron Larmor Radius (rho_e): {rho_e:.2f} km")
    print(f"Ion Skin Depth (d_i): {d_i:.2f} km")
    print(f"Electron Skin Depth (d_e): {d_e:.2f} km")
    print(f"Ion Sound Gyroradius (rho_s): {rho_s:.2f} km")
    print(f"Debye Length (lambda_D): {lambda_D:.2f} m")

    print("\n--- Characteristic Frequencies ---")
    print(f"Ion Cyclotron Period (t_ci): {t_ci:.4f} s")
    print(f"Electron Cyclotron Period (t_ce): {t_ce * 1e3:.4f} ms")
    print(f"Ion Plasma Period (t_pi): {t_pi * 1e3:.4f} ms")
    print(f"Electron Plasma Period (t_pe): {t_pe * 1e6:.4f} \\mu s")

    print("="*70)

    return {
        "b_vec": b_vec_mean, "bmag": bmag_mean,
        "den_e": den_e_mean, "den_i": den_i_mean,
        "te": te_mean, "ti": ti_mean, "ti_te_ratio": ti_te_ratio,
        "beta_e": beta_e, "beta_i": beta_i, "bg_ratio": bg_ratio,
        "ue_gse": ue_gse_mean, "ui_gse": ui_gse_mean,
        "vA": vA, "vti": vti, "vte": vte, "c_s": c_s,                 # km/s
        "rho_i": rho_i, "rho_e": rho_e, "d_i": d_i, "d_e": d_e,       # km
        "rho_s": rho_s, "lambda_D": lambda_D,                         # km (lambda_D in m)
        "t_ci": t_ci, "t_ce": t_ce, "t_pi": t_pi, "t_pe": t_pe,       # s
    }
