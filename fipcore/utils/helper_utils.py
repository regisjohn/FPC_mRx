"""
================================================================================
General utility functions for MMS data processing.

Functions:
- downsample_cad: Downsample a tplot variable to the cadence of a reference variable.
- upsample_cad: Upsample a tplot variable to the cadence of a reference variable.
- is_interleaved: Check if a tplot variable is interleaved.
- bl_recx_idx: Find the time index of a reconnection event.
- compute_vth: Compute the thermal velocity of a species.
- slice3d_to_2d: Slice a 3D or 4D array into 2D arrays.
- slice3d_to_2d_fac: Wrapper of slice3d_to_2d for the FAC case.
- global_vmin_vmax_vdf: Compute the global minimum and maximum of a set of vdfs.
- global_vmin_vmax_fpc: Compute the global minimum and maximum of a set of fpc data.
- print_fac_lmn_proj: Print the FAC to LMN and LMN to FAC projections.
- compute_beta_par: Compute the parallel plasma beta profile.
- compute_beta_perp: Compute the perpendicular plasma beta profile.
- compute_beta_scalar: Compute the scalar plasma beta profile.

Author: Regis John
Created: 2026-03-25
================================================================================
"""

import numpy as np
from pyspedas import avg_data, tinterpol, get_data, time_string
from scipy.constants import c, physical_constants, e, mu_0
import h5py
import os

from .io_utils import mms_name_make


def downsample_cad(tplot_var, ref_tplot_var, trange, newname=None, **kwargs):
    """
    Downsample a tplot variable to match the cadence of a reference tplot 
    variable using width-based boxcar averaging.

    Parameters
    ----------
    - tplot_var: str
        Name of the tplot variable to downsample.
    - ref_tplot_var: str
        Name of the reference tplot variable to match the cadence of.
    - trange: int/float (optional, but recommended)
        Time range over which to downsample the tplot variable.
    - newname: str
        Name of the downsampled tplot variable. If not set, the default is
        to add a '_dwnsmple' to the input name.
    - **kwargs: dict
        Additional keyword arguments to be passed to the avg_data() function.

    Returns
    -------
    - outname: str
        Name of the downsampled tplot variable.
    """
    # Get target cadence
    t_ref, _ = get_data(ref_tplot_var)
    des_cadence = len(t_ref)

    # Input variable cadence
    t_in, _ = get_data(tplot_var)
    in_cadence = len(t_in)

    # Number of values for the averaging window.
    n_avg = in_cadence/des_cadence

    # Name of downsampled tplot variable
    outname = newname if newname else f'{tplot_var}_dwnsmple'

    # Downsample tplot variable
    return avg_data(tplot_var, trange=trange, width=n_avg,
                             newname=outname, **kwargs)


def upsample_cad(var_in, ref_var, method='linear', newname=None, **kwargs):

    """
    Upsamples a tplot variable to match the cadence of a reference tplot variable.

    Parameters
    ----------
    var_in : str
        Name of the tplot variable to upsample.
    ref_var : str
        Name of the reference tplot variable to match the cadence of.
    method : str, optional
        Interpolation method. Default is 'linear'.
    newname : str, optional
        Name of the upsampled tplot variable. If not set, the default is
        to add a '_interp' to the input name.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the tinterpol() function.

    Returns
    -------
    outname : str
        Name of the upsampled tplot variable.
    """
    outname = newname if newname else f'{var_in}_interp'

    tinterpol(names=var_in, interp_to=ref_var, newname=outname, method=method, 
              **kwargs)

    return outname


def is_interleaved(arr, atol=1e-6):
    """
    Checks if an energy table is interleaved by comparing it with a replicated 
    version of its first row.

    Parameters
    ----------
    arr : numpy array
        The energy array to check for interleaving.
    atol : float, optional
        The absolute tolerance for the comparison. Default is 1e-6.

    Returns
    -------
    bool
        True if the energy array is interleaved, False otherwise.
    """
    arr0 = arr[0, :]  # Takes the first row (time t=0) with shape (32,) across all energy bins.
    test_arr = np.tile(arr0, (arr.shape[0], 1))  # Replicates arr0 vertically to arr.shape[0] creating (N, 32)
    return not np.allclose(arr, test_arr, atol)  # 1 if interleaved, otherwise 0


def bl_recx_idx(bvec_lmn, t=None):
    """
    Finds the time index of a reconnection event by comparing the midpoint or 
    max gradient of the reconnecting magnetic field B_L.

    Parameters
    ----------
    bvec_lmn (str or numpy array): Magnetic field data.
    t (numpy array): Time array required when bvec_lmn is provided as a 
        NumPy array. Optional & ignored when a tplot variable name is supplied.


    Returns
    -------
    idx_grad : int
        The time index of the maximum gradient of the magnetic field.
    """
    if isinstance(bvec_lmn, str):
        # Load magnetic field data
        t, blmn_arr = get_data(bvec_lmn)
    else:
        # If inputs are NumPy arrays, use them directly
        if t is None:
            raise ValueError("Time array 't' must be provided when using NumPy arrays.")
        blmn_arr = bvec_lmn

    bl = blmn_arr[:, 0]

    # Midpoint method
    B_start = bl[0]
    B_end   = bl[-1]
    B_mid   = 0.5 * (B_start + B_end)
    idx_mid = np.argmin(np.abs(bl - B_mid))
    t_mid = t[idx_mid]

    # Max-gradient method
    dBL_dt = np.gradient(bl, t)
    idx_grad = np.argmax(np.abs(dBL_dt))
    t_grad = t[idx_grad]

    # --- diagnostics ---
    dt_diff = abs(t_grad - t_mid)

    print(f"Midpoint estimate: idx={idx_mid},  t={time_string(t_mid)}")
    print(f"Max-grad estimate: idx={idx_grad}, t={time_string(t_grad)}")
    print(f"Difference: {dt_diff:.3f} seconds")

    if dt_diff < 0.05:
        print("Clean event: midpoint and gradient agree.")
    else:
        print("Messy event: indices differ significantly. Inspect manually.")

    return idx_grad


def compute_vth(t_para, t_perp, species='e'):
    """
    Compute the thermal speed of a species given the parallel and 
    perpendicular temperatures.

    Parameters
    ----------
    t_para : str
        Name of tplot variable containing the parallel temperature.
    t_perp : str
        Name of tplot variable containing the perpendicular temperature.
    species : str, optional
        Species for which to compute the thermal velocity. Defaults to 'electron'.

    Returns
    -------
    vth : array-like
        The thermal velocity of the species in km/s.
    vth_mean : float
        The mean thermal velocity of the species in km/s.
    """
    # c is in m/s. We multiply by 1e-3 to get km/s.
    c_km_s = c * 1e-3 
    
    if species.lower() == 'ion':
        # Convert proton mass energy equivalent in MeV to eV
        mc2_ev = physical_constants['proton mass energy equivalent in MeV'][0] * 1e6
    elif species.lower() == 'e':
        # Convert electron mass energy equivalent in MeV to eV
        mc2_ev = physical_constants['electron mass energy equivalent in MeV'][0] * 1e6
    else:
        raise ValueError("Species must be 'ion' or 'electron'")

    # Get Data
    _, t_para = get_data(t_para)
    _, t_perp = get_data(t_perp)

    # Compute Scalar Temperature: T_tot = (T_para + 2*T_perp) / 3
    t_tot = (t_para + 2.0 * t_perp) / 3.0

    # Calculate v_th = c * sqrt(2 * T_ev / mc2_ev)
    vth = c_km_s * np.sqrt(2.0 * t_tot / mc2_ev)
    vth_mean = np.mean(vth)

    return vth, vth_mean


def slice3d_to_2d(dat_3d, time_slice=None, mode="integrate", slice_index=None):    
    """
    Slice a 3D or 4D array into 2D arrays in either the "integrate" or "slice" mode.

    Parameters
    ----------
    dat_3d : array-like
        The 3D or 4D input array to slice or integrate.
    time_slice : int, optional
        The time index to slice the array. Required if input is 4D.
    mode : str, optional
        The mode of the slicing operation. Defaults to "integrate".
    slice_index : int, optional
        The index of the slice if mode is "slice". Defaults to Nx // 2.

    Returns
    -------
    dat_xy : array-like
        The 1st 2D array resulting from the integration or slicing operation.
    dat_yz : array-like
        The 2nd 2D array resulting from the integration or slicing operation.
    dat_xz : array-like
        The 3rd 2D array resulting from the integration or slicing operation.
    """
    # If dat_3d is 4‑D, slice time
    if dat_3d.ndim == 4:
        if time_slice is None:
            raise ValueError("time_slice must be provided for 4D input")
        dat_3d_t = dat_3d[:, :, :, time_slice]
    else:
        # Already time‑sliced 3‑D input
        dat_3d_t = dat_3d
    
    Nx = dat_3d_t.shape[0]

    if mode == "integrate":
        dat_xy = np.nansum(dat_3d_t, axis=2)   # integrate over v∥
        dat_yz = np.nansum(dat_3d_t, axis=0)   # integrate over v⊥1
        dat_xz = np.nansum(dat_3d_t, axis=1)   # integrate over v⊥2

    else:  # mode == "slice"
        if slice_index is None:
            slice_index = Nx // 2

        dat_xy = dat_3d_t[:, :, slice_index]
        dat_yz = dat_3d_t[slice_index, :, :]
        dat_xz = dat_3d_t[:, slice_index, :]

    return dat_xy, dat_yz, dat_xz


def slice3d_to_2d_fac(*args, **kwargs):
    """
    Wrapper function of slice3d_to_2d for the FAC case where the arrays are 
    transposed to have parallel components as the x-axis.

    Parameters
    ----------
    *args, **kwargs
        Passed to slice3d_to_2d

    Returns
    -------
    vdf_xy : array-like
        2D array of perp2 vs perp1
    vdf_yz : array-like
        2D array of perp2 vs par, parallel is horizontal
    vdf_xz : array-like
        2D array of perp1 vs par, parallel is horizontal
    """
    vdf_xy, vdf_yz, vdf_xz = slice3d_to_2d(*args, **kwargs)
    return (
        vdf_xy,          # perp2 vs perp1
        vdf_yz.T,        # parallel is horizontal, perp2 vs par
        vdf_xz.T         # parallel is horizontal, perp1 vs par
    )


def global_vmin_vmax_vdf(*vdfs, lower=None, upper=None):
    """
    Compute the global minimum and maximum of a set of velocity distributions.

    Parameters
    ----------
    *vdfs : 2D array-like
        A set of 2D vdf array to compute the global minimum and maximum.
    lower : float, optional
        The lower bound of the global minimum. Defaults to None.
    upper : float, optional
        The upper bound of the global maximum. Defaults to None.

    Returns
    -------
    vmin : float
        The global minimum of the velocity distributions.
    vmax : float
        The global maximum of the velocity distributions.
    """
    # If both limits are provided by the user, skip calculation entirely
    if lower is not None and upper is not None:
        return lower, upper
    
    # Only calculate if necessary
    C_list = [np.log10(np.where(v > 0, v, np.nan)) for v in vdfs]
    
    vmin = lower if lower is not None else np.nanmin(C_list)
    vmax = upper if upper is not None else np.nanmax(C_list)
    
    return vmin, vmax

def global_vmin_vmax_fpc(*arrays, upper=None):

    """
    Compute the global minimum and maximum of a set of fpc data.

    Parameters
    ----------
    *arrays : array-like
        A set of arrays to compute the global minimum and maximum.
    upper : float, optional
        The upper bound of the global maximum. Defaults to None.

    Returns
    -------
    vmin : float
        The global minimum of the arrays.
    vmax : float
        The global maximum of the arrays.
    """
    if upper is not None:
        return -upper, +upper
    
    # Only compute if upper is not provided
    max_abs = np.nanmax([np.nanmax(np.abs(a)) for a in arrays])
    return -max_abs, +max_abs


def print_fac_lmn_proj(trange, key="fac2lmn_proj", t_start=58, t_end=64, 
        species='e', probe='1', data_rate='brst', bin_width_frac=0.25, 
        subtract_f0=False):
    """
    Function to print the FAC to LMN and LMN to FAC projections.

    Parameters:
    - trange (list of str): start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss'].
    - key (str): Key of the projection matrix dataset in the .h5 file. 
        Default is 'fac2lmn_proj'.
    - t_start (int): Start time index of the data segment. Default is 58.
    - t_end (int): End time index of the data segment. Default is 64.
    - species (str): Species of particle to analyze. Default is 'e'.
    - probe (str): Probe number to analyze. Default is '1'.
    - data_rate (str): Data rate of the data. Default is 'brst'.
    - bin_width_frac (float): Bin width fraction. Default is 0.25.
    - subtract_f0 (bool): if True, use the df suffix in the filename. Default is False.

    Returns:
    - None
    """
    data_dir = "data"
    type_tag = 'df' if subtract_f0 else 'f'
    hfile_pref = f'mms{probe}_{data_rate}_{type_tag}_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, mms_name_make(hfile_pref, trange[0], trange[1]))
    
    with h5py.File(hfile, "r") as f:
        fac2lmn = f["meta"]["fac2lmn"][...]
        lmn2fac = f["meta"]["lmn2fac"][...]

    print("\n=== FAC to LMN Projections ===")
    for t in range(t_start, t_end + 1):
        e1, e2, epar = fac2lmn[t]
        print(f"\nTime index {t}:")
        print(f"  e_perp1 = {e1[0]: .3f} L  {e1[1]: .3f} M  {e1[2]: .3f} N")
        print(f"  e_perp2 = {e2[0]: .3f} L  {e2[1]: .3f} M  {e2[2]: .3f} N")
        print(f"  e_par   = {epar[0]: .3f} L  {epar[1]: .3f} M  {epar[2]: .3f} N")

    print("\n=== LMN to FAC Projections ===")
    for t in range(t_start, t_end + 1):
        eL, eM, eN = lmn2fac[t]
        print(f"\nTime index {t}:")
        print(f"  e_L = {eL[0]: .3f} e_perp1  {eL[1]: .3f} e_perp2  {eL[2]: .3f} e_par")
        print(f"  e_M = {eM[0]: .3f} e_perp1  {eM[1]: .3f} e_perp2  {eM[2]: .3f} e_par")
        print(f"  e_N = {eN[0]: .3f} e_perp1  {eN[1]: .3f} e_perp2  {eN[2]: .3f} e_par")


def compute_beta_par(den, t_para, b_mag):
    """
    Function to compute the plasma beta profile using parallel temperature. 

    Parameters:
    - den (ndarray): Density array in cm^-3.
    - t_para (ndarray): Temperature array in eV.
    - b_mag (ndarray): Magnetic field array in nT.
    
    Returns:
    - beta_par (ndarray): Parallel plasma beta profile.
    """
    
    p_para = (den * 1e6) * e * t_para # density in m^3, temperature in K
    p_mag = (b_mag * 1e-9)**2 / (2.0 * mu_0) # magnetic field from nT to T
    return p_para / p_mag


def compute_beta_perp(den, t_perp, b_mag):
    """
    Function to compute the plasma beta profile using perpendicular temperature. 

    Parameters:
    - den (ndarray): Density array in cm^-3.
    - t_perp (ndarray): Temperature array in eV.
    - b_mag (ndarray): Magnetic field array in nT.
    
    Returns:
    - beta_perp (ndarray): Perpendicular plasma beta profile.
    """
    
    
    p_perp = (den * 1e6) * e * t_perp # density in m^3, temperature in K
    p_mag = (b_mag * 1e-9)**2 / (2.0 * mu_0) # magnetic field from nT to T
    return p_perp / p_mag

def compute_beta_scalar(den, t_para, t_perp, b_mag):
    """
    Function to compute the plasma beta profile using scalar temperature. 

    Parameters:
    - den (ndarray): Density array in cm^-3.
    - t_para (ndarray): Parallel temperature array in eV.
    - t_perp (ndarray): Perpendicular temperature array in eV.
    - b_mag (ndarray): Magnetic field array in nT.
    
    Returns:
    - beta_scalar (ndarray): Scalar plasma beta profile.
    """
    
    t_avg = (t_para + 2.0 * t_perp) / 3.0
    p_total = (den * 1e6) * e * t_avg # density in m^3, temperature in K
    p_mag = (b_mag * 1e-9)**2 / (2.0 * mu_0) # magnetic field from nT to T
    return p_total / p_mag