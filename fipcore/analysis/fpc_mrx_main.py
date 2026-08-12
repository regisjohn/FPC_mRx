"""
================================================================================
Master function to perform MMS FPC Analysis and ancillary plasma parameters.

Functions:
- fpc_mrx_main: Master function to perform MMS FPC Analysis.
- fpc_mrx_fold: Function to compute the folded FPC and save to the same h5 file.
- fpc_mrx_plasma_params: Master function to compute ancillary plasma parameters 

Author: Regis John
Created: 2026-03-31
================================================================================
"""


import numpy as np
import os
from scipy.constants import elementary_charge as q_e
from pyspedas.projects import mms
from pyspedas import get_data, tplot_rename, tvector_rotate, fac_matrix_make, tplot_save
import h5py

import fipcore.analysis.vel_proc as vel
import fipcore.analysis.vdf_proc as vdf
import fipcore.analysis.evec_proc as evec
import fipcore.analysis.fpc_proc as fpc
import fipcore.utils.helper_utils as hutil
import fipcore.utils.coord_utils as coord
import fipcore.utils.io_utils as iout
import fipcore.analysis.jdotE_proc as je


def fpc_mrx_main(trange, species='e', vth_lim=3.5, bin_width_frac=0.25, mean_phi=False, 
        probe='1', data_rate='brst', level='l2', get_support_data=True, no_update=True, 
        lmn_mat_name='lmn_matrix', fac_mat_name='fac_matrix', subtract_f0=False, 
        v_flow=None, **kwargs):
    """
    Master function to perform MMS FPC Analysis by calling the necessary functions.

    Parameters:
    - trange (list of str): [start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - species (str): Species of particle to analyze. Default is 'e'.
    - vth_lim (float): Thermal velocity limit used to determine the bin edges. 
        Default is 3.5.
    - bin_width_frac (float): Fraction of the thermal velocity used to 
        determine the bin width. Default is 0.25.
    - mean_phi (bool): mean or instantaneous phi values. Default: True.
    - probe (str): Probe number to analyze. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - level (str): Level of the data. Default is 'l2'.
    - get_support_data (bool): If True, load in support data. Default is True.
    - no_update (bool): If True, the data will not be updated from the server. 
        Default is True.
    - lmn_mat_name (str): Name of the LMN matrix. Default is 'lmn_matrix'.
    - fac_mat_name (str): Name of the FAC matrix. Default is 'fac_matrix'.
    - subtract_f0 (bool): If True, subtract the equilibrium VDF from the raw VDF. 
        Default is False.
    - v_flow (ndarray or list): Mean flow velocity in m/s to use for equilibrium VDF. 
        Default is None. 
    - **kwargs: Additional keyword arguments to pass to the other functions.

    Returns:
    - hfile (str): Full path of the .h5 file containing the results of the analysis.
    """
    # --- MMS Data Loading --- 
    # Read in FPI data
    fpi_vars = mms.fpi(trange= trange, probe=probe, data_rate=data_rate, level=level,
        datatype=['des-dist', 'dis-dist', 'des-moms', 'dis-moms'], 
        time_clip=True, varnames=[f'mms{probe}_des_energy_brst', f'mms{probe}_des_phi_brst', 
        f'mms{probe}_des_bulkv_gse_brst', f'mms{probe}_dis_bulkv_gse_brst', 
        f'mms{probe}_des_temppara_brst', f'mms{probe}_des_tempperp_brst', 
        f'mms{probe}_des_numberdensity_brst', f'mms{probe}_des_dist_brst', 
        f'mms{probe}_des_disterr_brst', f'mms{probe}_des_errorflags_brst_dist'], 
        get_support_data=get_support_data, no_update=no_update)
    
    # Read in EDP data
    edp_vars = mms.edp(trange=trange, probe=probe, data_rate=data_rate, level=level,
        datatype=['dce', 'scpot'],
        time_clip=True, varnames=[f'mms{probe}_edp_scpot_brst_l2', 
        f'mms{probe}_edp_dce_gse_brst_l2', f'mms{probe}_edp_dce_par_epar_brst_l2'], 
        get_support_data=get_support_data, no_update=no_update)
    
    # Read in FGM data
    fgm_vars = mms.fgm(trange=trange, probe=probe, data_rate=data_rate, level=level,
        varnames=f'mms{probe}_fgm_b_gse_brst_l2', time_clip=True, 
        get_support_data=get_support_data, no_update=no_update)
    
    # Renaming tplot variables
    tplot_rename(f'mms{probe}_des_energy_brst', 'nrgy')
    tplot_rename(f'mms{probe}_edp_scpot_brst_l2', 'scpot')
    tplot_rename(f'mms{probe}_des_bulkv_gse_brst', 'bulk_ve_gse')
    tplot_rename(f'mms{probe}_dis_bulkv_gse_brst', 'bulk_vi_gse')
    tplot_rename(f'mms{probe}_des_phi_brst', 'phi')
    tplot_rename(f'mms{probe}_fgm_b_gse_brst_l2_bvec', 'bvec_gse')
    tplot_rename(f'mms{probe}_des_temppara_brst', 'te_para')
    tplot_rename(f'mms{probe}_des_tempperp_brst', 'te_perp')
    tplot_rename(f'mms{probe}_des_numberdensity_brst', 'den') 
    tplot_rename(f'mms{probe}_des_dist_brst', 'vdf_raw')
    tplot_rename(f'mms{probe}_des_disterr_brst', 'vdf_err')
    tplot_rename(f'mms{probe}_des_errorflags_brst_dist', 'dq_flags')
    tplot_rename(f'mms{probe}_edp_dce_gse_brst_l2', 'evec_gse')

    print("Loaded MMS data!")

    # Downsampling to DES cadence of 30 ms
    hutil.downsample_cad('scpot', 'bulk_ve_gse', trange, newname='scpot_dwn')
    hutil.downsample_cad('bvec_gse', 'bulk_ve_gse', trange, newname='bvec_gse_dwn')
    hutil.downsample_cad('evec_gse', 'bulk_ve_gse', trange, newname='evec_gse_dwn')

    # Interpolating to DES cadence of 30 ms
    hutil.upsample_cad('bulk_vi_gse', 'bulk_ve_gse', newname='bulk_vi_gse_int')

    # --- Spacecraft Potential & Interleave Correction ---
    vbin, grids_corr, interleave_info = vel.nrgy_to_vbin('nrgy', 'scpot_dwn', species=species)

    # --- Obtain Velocity Coordinates ---
    vv = vel.vbin_to_vv(vbin, 'phi', mean_phi=mean_phi)
    
    # --- Shift into Reconnection Frame ---
    coord.lmn_matrix_make('bvec_gse_dwn', trange, newname=lmn_mat_name) # Creating LMN matrix
    tvector_rotate(mat_var_in=lmn_mat_name,vec_var_in='bvec_gse_dwn',
                        newname='bvec_lmn') # Rotating B vector into LMN frame
    
    idx = hutil.bl_recx_idx('bvec_lmn')# Finding index of X-line passing
    bulk_vi_rev = vel.vi_bl_flip(idx, 'bulk_vi_gse_int') # Bulk ion velocity at X-line

    vv_recx = vel.vv_to_recx(vv, bulk_vi_rev) # Shift into reconnection frame
    
    # --- Rotate velocity bins into FAC Coordinates ---
    fac_matrix_make(mag_var_name='bvec_gse_dwn', other_dim='Xgse', 
                             newname=fac_mat_name) # Creating FAC matrix
    time, fac_matrix  = get_data(fac_mat_name)

    vv_fac = vel.vv_to_fac(vv_recx, fac_matrix) # Shift into FAC coordinates  

    # --- Rotate velocity bins into LMN Coordinates ---
    _, lmn_matrix  = get_data(lmn_mat_name)
    vv_lmn = vel.vv_to_lmn(vv_recx, lmn_matrix) # Shift into LMN coordinates

    # --- Computing FAC to LMN unit vector projections ---
    fac2lmn, lmn2fac = coord.fac_lmn_proj(fac_mat_name, lmn_mat_name)

    # --- Computing Thermal Velocities ---
    vthe, vthe_mean = hutil.compute_vth('te_para', 'te_perp', species=species)

    # --- Create Coordinate Maps ---
    vmap_fac, binc_fac, binc_facn, edges_fac, edges_facn, n_vbins_fac = vel.vv_to_vmap(vv_fac, 
                                vthe_mean, vth_lim=vth_lim, bin_width_frac=bin_width_frac)
    vmap_lmn, binc_lmn, binc_lmnn, edges_lmn, edges_lmnn, n_vbins_lmn = vel.vv_to_vmap(vv_lmn, 
                                vthe_mean, vth_lim=vth_lim, bin_width_frac=bin_width_frac)
    
    print("Velocity maps created!")

    # --- Compute Volume Element --- 
    vvol = vel.compute_vbin_vol(vbin)

    # --- Process VDF ---
    vdf_raw, vdf_vol = vdf.process_vdf('vdf_raw', 'vdf_err', 'dq_flags', vvol)

    # --- Create equilibrium VDF if subtract_f0 ---
    if subtract_f0:
        vdf_eq = vdf.equilibrium_vdf(vv_fac, 'den', 'te_para', 'te_perp', 
                        species=species, v_flow=v_flow)
        vdf_vol -= vdf_eq*vvol # delta_f

    # --- Binning the VDF ---
    # in FAC:
    bvdf_vol_fac, npts_vol_fac = vdf.bin_vdf_vol(vdf_vol, vmap_fac, n_vbins_fac, 
                                    n_vbins_fac, n_vbins_fac)
    bvdf_avg_fac, _ = vdf.bin_vdf_avg(vdf_raw, vmap_fac, n_vbins_fac, 
                                    n_vbins_fac, n_vbins_fac)
    
    # in LMN:
    bvdf_vol_lmn, npts_vol_lmn = vdf.bin_vdf_vol(vdf_vol, vmap_lmn, n_vbins_lmn, 
                                    n_vbins_lmn, n_vbins_lmn)
    bvdf_avg_lmn, _ = vdf.bin_vdf_avg(vdf_raw, vmap_lmn, n_vbins_lmn, 
                                    n_vbins_lmn, n_vbins_lmn)
    
    print("VDF processed and binned!")

    # --- Lorentz Transform & rotate Electric Field ---
    evec.evec_to_recx('evec_gse_dwn', 'bvec_gse_dwn', bulk_vi_rev, newname='evec_gse_dwns_lor')
    # rotate electric field into FAC coordinates
    evec.evec_to_fac('evec_gse_dwns_lor', fac_mat_name, newname='evec_lor_fac')
    # rotate electric field into LMN coordinates
    evec.evec_to_lmn('evec_gse_dwns_lor', lmn_mat_name, newname='evec_lor_lmn')

    # --- Compute Alternative FPC ---
    # cprime in fac
    cpfac_raw = fpc.compute_cprime(vv_fac, 'evec_lor_fac', vdf_raw, species=species)
    cpfac_vol = fpc.compute_cprime(vv_fac, 'evec_lor_fac', vdf_vol, species=species)
    # cprime in lmn
    cplmn_raw = fpc.compute_cprime(vv_lmn, 'evec_lor_lmn', vdf_raw, species=species)
    cplmn_vol = fpc.compute_cprime(vv_lmn, 'evec_lor_lmn', vdf_vol, species=species)

    # binning cprime in fac
    bcpfac_vol, npts_vol_cpfac = fpc.bin_cprime_vol(cpfac_vol, vmap_fac, n_vbins_fac, 
                                n_vbins_fac, n_vbins_fac)
    bcpfac_avg, _ = fpc.bin_cprime_avg(cpfac_raw, vmap_fac, n_vbins_fac, n_vbins_fac, 
                                    n_vbins_fac)
    
    # binning cprime in lmn
    bcplmn_vol, npts_vol_cplmn = fpc.bin_cprime_vol(cplmn_vol, vmap_lmn, n_vbins_lmn, 
                                n_vbins_lmn, n_vbins_lmn)
    bcplmn_avg, _ = fpc.bin_cprime_avg(cplmn_raw, vmap_lmn, n_vbins_lmn, n_vbins_lmn, 
                                    n_vbins_lmn)
    
    # computing energization
    jefac_tot = fpc.jEtot_cprime(bcpfac_vol)
    jelmn_tot = fpc.jEtot_cprime(bcplmn_vol)
    
    print("Cprime computed and binned!")

    # --- Compute Standard FPC ---
    # C in FAC
    c_fac_vol = fpc.compute_fpc(bcpfac_vol, binc_fac, binc_fac, binc_fac)
    c_fac_avg = fpc.compute_fpc(bcpfac_avg, binc_fac, binc_fac, binc_fac)
    # C in LMN
    c_lmn_vol = fpc.compute_fpc(bcplmn_vol, binc_lmn, binc_lmn, binc_lmn)
    c_lmn_avg = fpc.compute_fpc(bcplmn_avg, binc_lmn, binc_lmn, binc_lmn)

    print("Computed Standard FPC!")

    # Extracting Field data for saving
    _, bvec_gse = get_data('bvec_gse_dwn')
    _, bvec_lmn = get_data('bvec_lmn')
    _, evec_lor_gse = get_data('evec_gse_dwns_lor')
    _, evec_lor_fac = get_data('evec_lor_fac')
    _, evec_lor_lmn = get_data('evec_lor_lmn')

    dat_grps = {
        "meta": {
            "subtract_f0":      subtract_f0,
            "species":          species,
            "trange":           trange,
            "time":             time,
            "bin_width_frac":   bin_width_frac,
            "vthe":             vthe,
            "vth_mean":         vthe_mean,
            "vth_lim":          vth_lim,
            "interleave_info": interleave_info,
            "vvol":             vvol,
            "fac2lmn":          fac2lmn,
            "lmn2fac":          lmn2fac,
            "probe":            probe,
            "data_rate":        data_rate,
            "level":            level,
            "vdf_raw":          vdf_raw,
            "vdf_vol":          vdf_vol,
        },
        "gse": {
            "vv":            vv,
            "bvec":          bvec_gse,
            "evec_lor":      evec_lor_gse,
        },
        "fac": {
            "rot_mat":      fac_matrix,
            "vv":           vv_fac,
            "vmap":         vmap_fac,
            "bvdf_vol":     bvdf_vol_fac,
            "npts_vol":     npts_vol_fac,
            "bvdf_avg":     bvdf_avg_fac,
            "evec_lor":     evec_lor_fac,
            "Cprime_vol":   cpfac_vol,
            "Cprime_raw":   cpfac_raw,
            "bCprime_avg":  bcpfac_avg,
            "bCprime_vol":  bcpfac_vol,
            "npts_vol_cp":  npts_vol_cpfac,
            "JE_tot":       jefac_tot,
            "c_vol":        c_fac_vol,
            "c_avg":        c_fac_avg,
            "binc":         binc_fac,
            "binc_n":       binc_facn,
            "edges":        edges_fac,
            "edges_n":      edges_facn,
        },
        "lmn": {
            "rot_mat":      lmn_matrix,
            "vv":           vv_lmn,
            "vmap":         vmap_lmn,
            "bvec":         bvec_lmn,
            "bvdf_vol":     bvdf_vol_lmn,
            "npts_vol":     npts_vol_lmn,
            "bvdf_avg":     bvdf_avg_lmn,
            "evec_lor":     evec_lor_lmn,
            "Cprime_vol":   cplmn_vol,
            "Cprime_raw":   cplmn_raw,
            "bCprime_avg":  bcplmn_avg,
            "bCprime_vol":  bcplmn_vol,
            "npts_vol_cp":  npts_vol_cplmn,
            "JE_tot":       jelmn_tot,
            "c_vol":        c_lmn_vol,
            "c_avg":        c_lmn_avg,
            "binc":         binc_lmn,
            "binc_n":       binc_lmnn,
            "edges":        edges_lmn,
            "edges_n":      edges_lmnn,
        }
    }
    if subtract_f0:
        dat_grps["fac"]["vdf_eq"] = vdf_eq
    
    # --- Saving to a .h5 file ---
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    type_tag = 'df' if subtract_f0 else 'f'
    hfile_pref = f'mms{probe}_{data_rate}_{type_tag}_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, iout.mms_name_make(hfile_pref, trange[0], trange[1]))
    iout.h5sav(hfile, dat_grps)

    print(f"Data saved to file: {hfile}")
    
    return hfile


def fpc_mrx_fold(trange, species='e', probe='1', data_rate='brst', 
    bin_width_frac=0.25, coord_type="fac", subtract_f0=False):
    """
    Computes folded FPC (Field Particle Correlation) data for a given time range
    and species (currently FAC only).

    Parameters:
    - trange (list of str): [start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - species (str): Species of particle to analyze. Default is 'e'. 
    - probe (str): Probe number. Default is '1'.
    - data_rate (str): Data rate. Default is 'brst'.
    - bin_width_frac (float): Fraction of the thermal velocity used to 
        determine the bin width. Default is 0.25.   
    - coord_type (str): The type of coordinate system to use. Default is "fac".
    - subtract_f0 (bool): if True, use the df suffix in the filename. Default is False.

    Returns:
    - hfile (str): Full path of the .h5 file containing the results of the 
        folded analysis.
    """

    # Data path setup
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    type_tag = 'df' if subtract_f0 else 'f'
    hfile_pref = f'mms{probe}_{data_rate}_{type_tag}_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, iout.mms_name_make(hfile_pref, trange[0], trange[1]))


    # Read in the computed fpc
    with h5py.File(hfile, "r") as f:
        c_vol = f[coord_type]["c_vol"][...] # (3, nbin, nbin, nbin, ntimes)
    ntimes = c_vol.shape[4] 
    nbin = c_vol.shape[1]
    idx = nbin // 2 # Find the folding over index:

    # Allocating memory for folded fpc
    cx_folds = np.zeros((2, nbin, nbin, ntimes))
    cy_folds = np.zeros((2, nbin, nbin, ntimes))
    cz_folds = np.zeros((2, nbin, nbin, ntimes))

    for t in range(ntimes):
        # Slicing the fpc data
        cx_xy, cx_yz, cx_xz = hutil.slice3d_to_2d(c_vol[0,:,:,:,t])
        cy_xy, cy_yz, cy_xz = hutil.slice3d_to_2d(c_vol[1,:,:,:,t])
        cz_xy, cz_yz, cz_xz = hutil.slice3d_to_2d(c_vol[2,:,:,:,t])
        # Create a tuple of fpc data
        cx_2d_list = (cx_xy, cx_xz, cx_yz.T) # The order is xy, xz and zy
        cy_2d_list = (cy_xy, cy_xz, cy_yz.T)
        cz_2d_list = (cz_xy, cz_xz, cz_yz.T)        
    
        # --- Fold Cx ---
        # Fold Cx(x,y) along +ve x-axis, flip the left and add to the right
        cx_folds[0, idx:, :, t] = cx_2d_list[0][idx:, :] + cx_2d_list[0][:idx, :][::-1, :]
        # Fold Cx(x,z) along +ve x-axis, flip the left and add to the right
        cx_folds[1, idx:, :, t] = cx_2d_list[1][idx:, :] + cx_2d_list[1][:idx, :][::-1, :]

        # --- Fold Cy ---
        # Fold Cy(x,y) along +ve y-axis, flip the bottom and add to the top
        cy_folds[0, :, idx:, t] = cy_2d_list[0][:, idx:] + cy_2d_list[0][:, :idx][:, ::-1]
        # Fold Cy(z,y) along +ve x-axis, flip the bottom and add to the top
        cy_folds[1, :, idx:, t] = cy_2d_list[2][:, idx:] + cy_2d_list[2][:, :idx][:, ::-1]

        # --- Fold Cz ---
        # Fold Cz(x,z) along +ve z-axis, flip the bottom and add to the top
        cz_folds[0,:, idx:, t] = cz_2d_list[1][:, idx:] + cz_2d_list[1][:, :idx][:, ::-1]
        # Fold Cz(z,y) along +ve z-axis, flip the left and add to the right
        cz_folds[1, idx:, :, t] = cz_2d_list[2][idx:,:] + cz_2d_list[2][:idx, :][::-1, :]

    dat_grps = {
        "fac": {
            "cx_folds": cx_folds,
            "cy_folds": cy_folds,
            "cz_folds": cz_folds,
            }
        }
    
    # --- Saving to a .h5 file ---
    iout.h5sav(hfile, dat_grps)

    print(f"Data saved to file: {hfile}")
    
    return hfile


def fpc_mrx_plasma_params(trange, probe='1', data_rate='brst', level='l2', species='e', 
            bg_ratio=None, bin_width_frac=0.25, subtract_f0=False, get_support_data=True, 
            no_update=True):
    """
    Master function to compute ancillary plasma parameters (density, temperature, bulk flow,
    current density, and J·E) using data from FPI and FGM instruments.

    Parameters:
    - trange (list of str): [start time, end time] in the format:
            ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - probe (str): Probe number. Default is '1'.
    - data_rate (str): Data rate. Default is 'brst'.
    - level (str): Level of the data. Default is 'l2'.
    - species (str): Species of particle to analyze. Default is 'e'. 
    - bg_ratio (float): Guide Field strength, default is None.
    - bin_width_frac (float): Fraction of the thermal velocity used to 
        determine the bin width, default is 0.25.
    - subtract_f0 (bool): if True, use the df suffix in the filename. Default is False.
    - get_support_data (bool): If True, load in support data. Default is True.
    - no_update (bool): If True, the data will not be updated from the server. 
        Default is True.

    Returns:
     - hfile (str): Full path of the .h5 file containing computed plasma 
        parameters for the specified time range.
    """
    

    # --- MMS Data Loading --- 
    # Read in FPI Data
    fpi_varlist = [f'mms{probe}_des_energyspectr_omni_brst', 
        f'mms{probe}_dis_energyspectr_omni_brst', f'mms{probe}_des_numberdensity_brst', 
        f'mms{probe}_dis_numberdensity_brst', f'mms{probe}_des_temppara_brst', 
        f'mms{probe}_dis_temppara_brst', f'mms{probe}_des_tempperp_brst', 
        f'mms{probe}_dis_tempperp_brst', f'mms{probe}_des_bulkv_gse_brst', 
        f'mms{probe}_dis_bulkv_gse_brst']
    
    fpi_vars = mms.fpi(trange=trange, probe=probe, data_rate=data_rate, level=level, 
                    datatype=['des-moms', 'dis-moms'], time_clip=True, 
                    varnames=fpi_varlist, get_support_data=get_support_data, 
                    no_update=no_update)

    # Read in FGM Data
    fgm_vars = mms.fgm(trange=trange, probe=probe, data_rate=data_rate, level=level,
        varnames=f'mms{probe}_fgm_b_gse_brst_l2', time_clip=True, 
        get_support_data=get_support_data, no_update=no_update)


    # Renaming tplot variabless
    tplot_rename(f'mms{probe}_des_energyspectr_omni_brst', 'energy_e')
    tplot_rename(f'mms{probe}_dis_energyspectr_omni_brst', 'energy_i')
    tplot_rename(f'mms{probe}_des_numberdensity_brst', 'den_e')
    tplot_rename(f'mms{probe}_dis_numberdensity_brst', 'den_i')
    tplot_rename(f'mms{probe}_des_temppara_brst', 'te_para')
    tplot_rename(f'mms{probe}_dis_temppara_brst', 'ti_para')
    tplot_rename(f'mms{probe}_des_tempperp_brst', 'te_perp')
    tplot_rename(f'mms{probe}_dis_tempperp_brst', 'ti_perp')
    tplot_rename(f'mms{probe}_des_bulkv_gse_brst', 'bulk_ve_gse')
    tplot_rename(f'mms{probe}_dis_bulkv_gse_brst', 'bulk_vi_gse')
    tplot_rename(f'mms{probe}_fgm_b_gse_brst_l2', 'bvec_gse')

    # Matching vars to DES cadence of 30 ms
    hutil.downsample_cad('bvec_gse', 'bulk_ve_gse', trange=trange, newname='bvec_gse_dwn')
    # hutil.upsample_cad('energy_i', 'energy_e', newname='energy_i_up')
    hutil.upsample_cad('den_i', 'bulk_ve_gse', newname='den_i_up')
    hutil.upsample_cad('ti_para', 'bulk_ve_gse', newname='ti_para_up')
    hutil.upsample_cad('ti_perp', 'bulk_ve_gse', newname='ti_perp_up')
    hutil.upsample_cad('bulk_vi_gse', 'bulk_ve_gse', newname='bulk_vi_gse_up')

    # Extract the numpy arrays from the tplot vars
    _, den_e = get_data('den_e')
    _, den_i = get_data('den_i_up')
    _, te_para = get_data('te_para')
    _, ti_para = get_data('ti_para_up')
    _, te_perp = get_data('te_perp')
    _, ti_perp = get_data('ti_perp_up')
    ntime_e, bulk_ve_gse = get_data('bulk_ve_gse')
    _, bulk_vi_gse = get_data('bulk_vi_gse_up')
    _, bvec_gse = get_data('bvec_gse_dwn')
    bmag = bvec_gse[:,3]
    print(bvec_gse.shape, bmag.shape)

    # Data path setup
    project_root= os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    type_tag = 'df' if subtract_f0 else 'f'
    hfile_pref = f'mms{probe}_{data_rate}_{type_tag}_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, iout.mms_name_make(hfile_pref, trange[0], trange[1]))

    # Reading in the E-field, fac & lmn matrices
    with h5py.File(hfile, "r") as f:
        evec_gse = f["gse"]["evec_lor"][:]
        evec_fac = f["fac"]["evec_lor"][:]
        evec_lmn = f["lmn"]["evec_lor"][:]
        fac_mat = f["fac"]['rot_mat'][...]
        lmn_mat = f["lmn"]['rot_mat'][...]

    # Computing current density in nA/m^2 in gse, fac and lmn
    jvec_gse = den_e.reshape(-1, 1) * q_e * (bulk_vi_gse - bulk_ve_gse)*1e18
    # cm^-3 * C * km/s = 10^18 nA/m^2 
    _, jvec_fac = coord.rotate_to_fac(jvec_gse, fac_mat)
    _, jvec_lmn = coord.rotate_to_lmn(jvec_gse, lmn_mat)

    # Computing jdotE in gse, fac and lmn
    _, jdotE_gse, jdotE_gse0, jdotE_gse1, jdotE_gse2 = je.compute_jdotE(jvec_gse, evec_gse)
    jdotE_gse = np.column_stack([jdotE_gse, jdotE_gse0, jdotE_gse1, jdotE_gse2])
    
    _, jdotE_fac, jdotE_fac0, jdotE_fac1, jdotE_fac2 = je.compute_jdotE(jvec_fac, evec_fac)
    jdotE_fac = np.column_stack([jdotE_fac, jdotE_fac0, jdotE_fac1, jdotE_fac2])

    _, jdotE_lmn, jdotE_lmn0, jdotE_lmn1, jdotE_lmn2 = je.compute_jdotE(jvec_lmn, evec_lmn)
    jdotE_lmn = np.column_stack([jdotE_lmn, jdotE_lmn0, jdotE_lmn1, jdotE_lmn2])

    # Rotating bulk flow to fac and lmn
    _, bulk_ve_fac = coord.rotate_to_fac(bulk_ve_gse, fac_mat)
    _, bulk_vi_fac = coord.rotate_to_fac(bulk_vi_gse, fac_mat)

    _, bulk_ve_lmn = coord.rotate_to_lmn(bulk_ve_gse, lmn_mat)
    _, bulk_vi_lmn = coord.rotate_to_lmn(bulk_vi_gse, lmn_mat)

    # Computing plasma beta profiles
    beta_e_para = hutil.compute_beta_par(den_e, te_para, bmag)
    beta_e_perp = hutil.compute_beta_perp(den_e, te_perp, bmag)
    beta_e_scalar = hutil.compute_beta_scalar(den_e, te_para, te_perp, bmag)

    beta_i_para = hutil.compute_beta_par(den_i, ti_para, bmag)
    beta_i_perp = hutil.compute_beta_perp(den_i, ti_perp, bmag)
    beta_i_scalar = hutil.compute_beta_scalar(den_i, ti_para, ti_perp, bmag)

    # --- Grouping Data for HDF5 ---
    dat_grps = {
        "meta":{
            "time": ntime_e,
            "bg_ratio": bg_ratio,
            "den_e": den_e,
            "den_i": den_i,
            "te_para": te_para,
            "ti_para": ti_para,
            "te_perp": te_perp,
            "ti_perp": ti_perp,
            "bmag": bmag,
            "beta_e_para":   beta_e_para,
            "beta_e_perp":   beta_e_perp,
            "beta_e_scalar": beta_e_scalar,
            "beta_i_para":   beta_i_para,
            "beta_i_perp":   beta_i_perp,
            "beta_i_scalar": beta_i_scalar,
        },
        "gse": {
            "bulk_ve": bulk_ve_gse,
            "bulk_vi": bulk_vi_gse,
            "jvec": jvec_gse,
            "jdotE": jdotE_gse
        },
        "fac": {
            "bulk_ve": bulk_ve_fac,
            "bulk_vi": bulk_vi_fac,
            "jvec": jvec_fac,
            "jdotE": jdotE_fac
        },
        "lmn": {
            "bulk_ve": bulk_ve_lmn,
            "bulk_vi": bulk_vi_lmn,
            "jvec": jvec_lmn,
            "jdotE": jdotE_lmn
        }
    }

    # --- Saving to a .h5 file ---
    # Note: Ensure the variable name passed here matches your dictionary (dat_grps)
    iout.h5sav(hfile, dat_grps)

    print(f"Data saved to file: {hfile}")

    # Create a new tplot file to save the energy data.
    tplot_file = hfile.replace('.h5', '.pyspd')
    tplot_save(['energy_e', 'energy_i'], filename=tplot_file)
    print(f'Energy spectra data saved to file: {tplot_file}')
    
    return hfile, tplot_file