"""
================================================================================
Functions for processing MMS VDF data.

Functions:
- process_vdf: Process the raw velocity distribution function (VDF) data.
- bin_vdf_3d: Bin VDF data into 3D velocity space.
- bin_vdf_vol: Bin VDF data into 3D velocity space with volume weighting.
- bin_vdf_avg: Bin VDF data into 3D velocity space with averaging.

Author: Regis John
Created: 2026-03-26
================================================================================
"""

import numpy as np
from pyspedas import get_data, tplot_rename
from scipy.interpolate import interp1d
from scipy.constants import c, physical_constants
from pyspedas.projects import mms


def equilibrium_vdf(vv_fac, den, t_para, t_perp, species='e', v_flow=None):
    """
    Calculate the equilibrium gyrotropic VDF for a given species and probe in FAC.

    Parameters:
    - vv_fac (ndarray): Velocity vectors in the FAC frame of shape (3, 16384, ntime).
    - den (str): Name of the tplot variable containing the density.
    - t_para (str): Name of tplot variable containing the parallel temperature.
    - t_perp (str): Name of the tplot variable containing the perpendicular temperature.
    - species (str): Species of particle to analyze. Default is 'e'.
    - v_flow (ndarray or list): Mean flow velocity in m/s. Default is None.

    Returns:
    - vdf_eq (ndarray): Equilibrium gyrotropic VDF of shape (bins, ntime) in s^3/m^6.
    """
    # Extract arrays from tplot variables
    _, den = get_data(den)    # Shape: (ntime,)
    _, t_para = get_data(t_para)    # Shape: (ntime,)
    _, t_perp = get_data(t_perp)    # Shape: (ntime,)

    # Mass constants in MeV converted to eV
    if species.lower() == 'ion':
        mc2_ev = physical_constants['proton mass energy equivalent in MeV'][0] * 1e6
    elif species.lower() == 'e':
        mc2_ev = physical_constants['electron mass energy equivalent in MeV'][0] * 1e6
    else:
        raise ValuoeError("Species must be 'ion' or 'e'")

    c_m_s = c   # Speed of light in m/s
    den_si = den * 1e6 # Convert density to m^-3
    
    # Calculate separate parallel and perpendicular thermal velocities (m/s)
    w_para = c_m_s * np.sqrt(2.0 * t_para / mc2_ev)  # Shape: (ntime,)
    w_perp = c_m_s * np.sqrt(2.0 * t_perp / mc2_ev)  # Shape: (ntime,)

    # Extract velocity components from the vv_fac(3, 16384, ntime) 
    v_x = vv_fac[0, :, :]*1e3  # Shape: (16384, ntime)
    v_y = vv_fac[1, :, :]*1e3  # Shape: (16384, ntime)
    v_z = vv_fac[2, :, :]*1e3  # Shape: (16384, ntime)

    if v_flow is not None:
        v_x = v_x - v_flow[0]
        v_y = v_y - v_flow[1]
        v_z = v_z - v_flow[2]

    v_perp_sq = v_x**2 + v_y**2  # Shape: (16384, ntime)
    v_para_sq = v_z**2           # Shape: (16384, ntime)

    # Setting up Tri-Maxwellian
    amplitude = den_si / ((np.pi**1.5) * (w_perp**2) * w_para) # Shape: (ntime,)
    exponent = -(v_perp_sq / (w_perp**2)) - (v_para_sq / (w_para**2))
    vdf_eq = amplitude * np.exp(exponent)
    
    return vdf_eq


def process_vdf(vdf_raw_var, vdf_err_var, dq_flags_var, vvol):
    """
    Process the raw velocity distribution function (VDF) data.

    Parameters
    ----------
    vdf_raw_var : str
        tplot variable of raw VDF data (time, phi, theta, energy)
    vdf_err_var : str
        tplot variable of error in raw VDF data (time, phi, theta, energy)
    dq_flags_var : str
        tplot variable of data quality flags (time, phi, theta, energy)
    vvol : numpy.ndarray
        Volume element (16384, n_sweeps) for weighting

    Returns
    -------
    vdf_raw : numpy.ndarray
        Processed VDF data without volume element weighting in s^3/m^6 (bins, time)
    vdf_vol : numpy.ndarray
        Processed VDF data with volume weighting in m^-3 (bins, time)

    """
    # Get Data
    vdf_rawi = get_data(vdf_raw_var).y # order (time, phi, theta, energy)
    time_x = get_data(vdf_raw_var).times
    vdf_err = get_data(vdf_err_var).y
    dq_flags = get_data(dq_flags_var).y
    
    # Flatten to (ntime, 16384)
    nbin = 32*16*32
    ntime = len(time_x)
    vdf_rawi = vdf_rawi.reshape(ntime, nbin)
    vdf_err = vdf_err.reshape(ntime, nbin)
    
    # Quality Check & Temporal Interpolation
    good_idx = np.where(dq_flags == 0)[0]
    bad_idx = np.where(dq_flags != 0)[0]
    
    if len(bad_idx) > 0 and len(good_idx) > 1:
        # Interpolate every bin across the time dimension
        itp = interp1d(time_x[good_idx], vdf_rawi[good_idx, :], axis=0, 
                       kind='linear', fill_value="extrapolate")
        vdf_rawi[bad_idx, :] = itp(time_x[bad_idx])
        
        # Also interpolate error for the noise calculation
        itp_err = interp1d(time_x[good_idx], vdf_err[good_idx, :], axis=0, 
                           kind='linear', fill_value="extrapolate")
        vdf_err[bad_idx, :] = itp_err(time_x[bad_idx])

    # Counts = round((vdf/sigma)^2)
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore divide by 0, keep console clean
        counts = np.round(np.square(vdf_rawi / vdf_err))

        # Set 0-counts to 0
        both_zero = (vdf_rawi == 0) & (vdf_err == 0)
        counts[both_zero] = 0

        # Replace 0-count or negative vdf with NaN
        mask = (counts <= 0) | (vdf_rawi <= 0) | np.isnan(vdf_rawi)
        vdf_rawi[mask] = np.nan


    # vdf_rawi is (ntime, 16384), vvol is (16384, n_sweeps)
    vdf_vol = vdf_rawi.T * vvol * 1e12 # Convert to SI units (s^3/m^6)
    vdf_raw = vdf_rawi.T * 1e12

    return vdf_raw, vdf_vol


def bin_vdf_3d(vdf_dat, vmap, n_vx, n_vy, n_vz):
    """
    Bin VDF data into a 3D grid.

    Parameters
    ----------
    vdf_dat : numpy.ndarray
        Processed VDF data(bins, time)
    vmap : numpy.ndarray
        Mapping of velocity bins in normalized velocity space (3, n_vx*n_vy*n_vz, n_sweeps)
    n_vx : int
        Number of bins along 1st direction
    n_vy : int
        Number of bins along 2nd direction
    n_vz : int
        Number of bins along 3rd direction

    Returns
    -------
    vdf_binned : numpy.ndarray
        Binned VDF data with shape: (n_vx, n_vy, n_vz, ntime)
    bin_npts : numpy.ndarray
        Number of points in each bin (n_vx, n_vy, n_vz, ntime)
    """
    # Extracting time array
    _, ntime = vdf_dat.shape

    # Output arrays
    vdf_binned = np.zeros((n_vx, n_vy, n_vz, ntime), dtype=float)
    bin_npts  = np.zeros((n_vx, n_vy, n_vz, ntime), dtype=int)

    # Loop only over time
    for t in range(ntime):

        # Extract bin indices for this time
        ix = vmap[0, :, t]   # v⊥1
        iy = vmap[1, :, t]   # v⊥2
        iz = vmap[2, :, t]   # v∥

        # Mask out invalid bins
        valid = (ix >= 0) & (iy >= 0) & (iz >= 0)

        ixv = ix[valid]
        iyv = iy[valid]
        izv = iz[valid]

        # Values to accumulate
        vals = vdf_dat[valid, t]

        # Scatter-add into 3D grid
        np.add.at(vdf_binned[..., t], (ixv, iyv, izv), vals)
        np.add.at(bin_npts[..., t],  (ixv, iyv, izv), 1)

    return vdf_binned, bin_npts

def bin_vdf_vol(*args, **kwargs):
    """
    This is a wrapper around the `bin_vdf_3d` function to bin vdf data that is 
    volume-weighted.

    Parameters:
    vdf_dat: numpy.ndarray of shape (npts, ntime); VDF data
    vmap: numpy.ndarray of shape (3, npts); Mapping of velocity bins in 
        normalized velocity space
    n_vx, n_vy, n_vz: int; number of bins in each velocity dimension

    Returns:
    vdf_vol_binned: numpy.ndarray of shape (n_vx, n_vy, n_vz, ntime)
        binned VDF volume weighted data
    bin_npts: numpy.ndarray of shape (n_vx, n_vy, n_vz, ntime)
        number of points in each bin
    """
    vdf_vol_binned, bin_npts = bin_vdf_3d(*args, **kwargs)
    return vdf_vol_binned, bin_npts


def bin_vdf_avg(*args, **kwargs):
    """
    This is a wrapper around the `bin_vdf_3d` function to bin vdf data that is 
    not volume-weighted and hence weight it by averaging.

    Parameters
    ----------
    vdf_dat: numpy.ndarray of shape (npts, ntime); VDF data
    vmap: numpy.ndarray of shape (3, npts); Mapping of velocity bins in 
        normalized velocity space
    n_vx, n_vy, n_vz: int; number of bins in each velocity dimension

    Returns
    -------
    vdf_avg_binned: numpy.ndarray of shape (n_vx, n_vy, n_vz, ntime)
        binned VDF averaged data
    bin_npts: numpy.ndarray of shape (n_vx, n_vy, n_vz, ntime)
        number of points in each bin
    """
    vdf_raw_binned, bin_npts = bin_vdf_3d(*args, **kwargs)
    with np.errstate(divide='ignore', invalid='ignore'):
        return vdf_raw_binned / bin_npts, bin_npts