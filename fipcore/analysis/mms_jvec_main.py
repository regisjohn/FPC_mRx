"""
================================================================================
Master function to compute MMS current density in various coordinates and 
save to an existing .h5 file.

Functions:
- mms_jvec_main: Master function to compute J vector in various coordinates.

Author: Regis John
Created: 2026-06-02
================================================================================
"""

import numpy as np
from pyspedas.projects import mms
from pyspedas import tplot_rename, fac_matrix_make
import os

from fipcore.analysis.jvec_proc import compute_jvec, jvec_to_fac, jvec_to_lmn
from fipcore.utils.io_utils import h5sav, mms_name_make
from fipcore.utils.coord_utils import lmn_matrix_make
from fipcore.utils.helper_utils import downsample_cad


def mms_jvec_main(trange, species='e', bin_width_frac=0.25, probe='1', 
    data_rate='brst', level='l2'):
    """
    Master function to compute MMS current density in various coordinates and 
    save to an existing .h5 file.

    Parameters:
    - trange (list of str): [start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - species (str): Species of particle to analyze. Default is 'e'.
    - bin_width_frac (float): Fraction of the thermal velocity used to 
        determine the bin width. Default is 0.25.
    - probe (str): Probe number to analyze. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - level (str): Level of the data. Default is 'l2'.

    Returns:
    - hfile (str): Full path of the .h5 file containing the data.
    """

    # Computing current density vector in GSE coordinates
    jvec_gse = compute_jvec(trange, probe=probe, data_rate=data_rate, level=level)

    # Read in magnetic field and electron bulf flow data
    mms.fpi(trange= trange, probe=probe, data_rate=data_rate, level=level,
    datatype=['des-moms'], time_clip=True, varnames='mms1_des_bulkv_gse_brst', 
    get_support_data=True, no_update=True)
    mms.fgm(trange=trange, probe=probe, data_rate=data_rate, level=level,
        varnames='mms1_fgm_b_gse_brst_l2', time_clip=True, 
        get_support_data=True, no_update=True)

    tplot_rename('mms1_des_bulkv_gse_brst', 'bulk_ve_gse')
    tplot_rename('mms1_fgm_b_gse_brst_l2_bvec', 'bvec_gse')

    # Downsample magnetic field to electron bulk flow cadence
    downsample_cad('bvec_gse', 'bulk_ve_gse', trange, newname='bvec_gse_dwn')

    # Rotate to FAC coordinates
    fac_mat_name = 'fac_mat_var'
    fac_matrix_make(mag_var_name='bvec_gse_dwn', other_dim='Xgse', 
                             newname=fac_mat_name) # Creating FAC matrix
    jvec_fac = jvec_to_fac(jvec_gse, fac_mat_name)

    # Rotate to LMN coordinates
    lmn_mat_name = 'lmn_mat_var'
    lmn_matrix_make('bvec_gse_dwn', trange, newname=lmn_mat_name)
    jvec_lmn = jvec_to_lmn(jvec_gse, lmn_mat_name)

    # Save to .h5 file
    dat_grps = {
        "gse": {
            "jvec_gse": jvec_gse
        },
        "fac": {
            "jvec_fac": jvec_fac
        },
        "lmn": {
            "jvec_lmn": jvec_lmn
        }
    }

    # Data path setup
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    hfile_pref = f'mms_fpc_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, mms_name_make(hfile_pref, trange[0], trange[1]))
    h5sav(hfile, dat_grps)

    return hfile