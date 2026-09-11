"""
================================================================================
Functions for computing J vector from MMS data and rotating into FAC & LMN.

Functions:
- compute_jvec: Compute J vector from MMS data.
- jvec_to_fac: Rotate J vector into FAC coordinates.
- jvec_to_lmn: Rotate J vector into LMN coordinates.

Author: Regis John
Date: 2026-03-31
================================================================================
"""

import numpy as np
from pyspedas.projects import mms
from pyspedas import tplot_rename, get_data, store_data, set_coords, set_units
from scipy.constants import elementary_charge as q_e
from pyspedas.projects import mms

from fipcore.utils.helper_utils import upsample_cad
from fipcore.utils.coord_utils import rotate_to_fac, rotate_to_lmn


def compute_jvec(trange, probe='1', data_rate='brst', level='l2', newname=None, 
        no_update=True):
    """
    Compute J vector from MMS data.

    Parameters:
    - trange (list of str): start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - probe (str): Probe number to analyze. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - level (str): Level of the data. Default is 'l2'.
    - newname (str): The name of the output tplot variable. If None, the default
         name is "jvec_gse"
    - no_update (bool): if True, do not download new data. Default is True. Turn 
        it to False to download new data if it is not available locally.

    Returns:
    - out_name (str): The name of the output tplot variable.
    """
    # Loading data
    dat = mms.fpi(trange=trange, probe=probe, data_rate=data_rate, level=level, 
                            datatype=['des-moms', 'dis-moms'], time_clip=True,
                            varnames=[f'mms{probe}_des_numberdensity_brst',
                                f'mms{probe}_dis_bulkv_gse_brst',
                                f'mms{probe}_des_bulkv_gse_brst'], no_update=no_update)
    
    # Renaming the tplot vars
    tplot_rename(f'mms{probe}_dis_bulkv_gse_brst', 'bulk_vi_gse')
    tplot_rename(f'mms{probe}_des_bulkv_gse_brst', 'bulk_ve_gse')
    tplot_rename(f'mms{probe}_des_numberdensity_brst', 'den_e')

    # Setting everything to DES cadence
    upsample_cad('bulk_vi_gse', 'bulk_ve_gse', newname='bulk_vi_gse_up')

    # Extracting Data
    _, den_e = get_data('den_e')
    times, bulk_ve = get_data('bulk_ve_gse')
    _, bulk_vi = get_data('bulk_vi_gse_up')

    print(den_e.shape, bulk_ve.shape, bulk_vi.shape)
    
    # Computing jvec
    jvec_gse = den_e.reshape(-1, 1) * q_e * (bulk_vi - bulk_ve)*1e18 
    # cm^-3 * C * km/s = 10^18 nA/m^2

    # Storing as a tplot variable
    out_name = newname if newname else "jvec_gse"
    store_data(out_name, data={'x': times, 'y': jvec_gse}, 
               attr_dict={'y': {'units': 'nA/m^2', 'label': "Current Density (nA/m^2) GSE."}})
    set_coords(out_name, 'GSE')
    set_units(out_name, 'nA/m^2')

    return out_name


def jvec_to_fac(jvec_gse, fac_mat_var, newname=None):
    """
    Rotate the current density from GSE to FAC coordinates.

    Parameters:
    jvec_gse (str): Name of the tplot variable containing the current density data.
    fac_mat_var (str): Name of the tplot variable containing the FAC rotation matrix.
    newname (str, optional): Name of the output tplot variable, defaults to "jvec_fac".

    Returns:
    out_name (str): The name of the output tplot variable.
    """
    times, jvec_fac = rotate_to_fac(jvec_gse, fac_mat_var)
    
    # Store the result back as a tplot variable
    out_name = newname if newname else "jvec_fac"
    store_data(out_name, data={'x': times, 'y': jvec_fac}, 
               attr_dict={'y': {'units': 'nA/m^2', 'label': "Current Density (nA/m^2) FAC."}})
    set_coords(out_name, 'FAC')
    set_units(out_name, 'nA/m^2')
    
    return out_name


def jvec_to_lmn(jvec_gse, lmn_mat_var, newname=None):
    """
    Rotate the current density from GSE to LMN coordinates.

    Parameters:
    jvec_gse (str): Name of the tplot variable containing the current density data.
    lmn_mat_var (str): Name of the tplot variable containing the LMN rotation matrix.
    newname (str, optional): Name of the output tplot variable, defaults to "jvec_lmn".

    Returns:
    out_name (str): The name of the output tplot variable.
    """
    times, jvec_lmn = rotate_to_lmn(jvec_gse, lmn_mat_var)
    
    # Store the result back as a tplot variable
    out_name = newname if newname else "jvec_lmn"
    store_data(out_name, data={'x': times, 'y': jvec_lmn}, 
               attr_dict={'y': {'units': 'nA/m^2', 'label': "Current Density (nA/m^2) LMN."}})
    set_coords(out_name, 'LMN')
    set_units(out_name, 'nA/m^2')
    
    return out_name