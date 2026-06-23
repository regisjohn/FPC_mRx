"""
================================================================================
Functions to compute plasma parameters from MMS data.

Functions:
- compute_beta_profiles: Compute the plasma beta profiles.

Author: Regis John
Created: 2026-06-23
================================================================================
"""

from pyspedas.projects import mms
from pyspedas import tplot_rename, get_data
from fipcore.utils.helper_utils import downsample_cad


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