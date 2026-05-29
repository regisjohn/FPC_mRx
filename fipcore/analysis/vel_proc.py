"""
================================================================================
Functions for creating MMS velocity bin and map in both FAC and LMN frames.

Functions:
- nrgy_to_vbin: Convert an energy grid into velocity bins while accounting for 
    spacecraft potential correction.
- vbin_to_vv: Convert velocity bins into velocity vectors in GSE frame.
- vi_bl_flip: Compute the bulk ion velocity at the B_L reversal point in time.
- vv_to_recx: Convert velocity vectors in GSE frame to recx frame coordinates.
- vv_to_fac: Rotate velocity vectors to the FAC frame.
- vv_to_lmn: Rotate velocity vectors to the LMN frame.
- vv_to_vmap: Map velocity vectors to bin indices in normalized velocity space.
- vv_to_vmap_3d: Map velocity vectors to 3D bin indices in normalized velocity space.
- compute_vbin_vol: Compute velocity bin volumes of the device.

Author: Regis John
Created: 2026-03-26
================================================================================
"""


import numpy as np
from scipy.constants import c, physical_constants
from pyspedas import get_data
from fipcore.utils.helper_utils import is_interleaved


def nrgy_to_vbin(nrgyraw_name, scpot_name, species='e'):
    """
    Convert an energy grid into velocity bins while accounting for spacecraft 
    potential correction.

    Parameters
    ----------
    nrgyraw_name : str
        Name of the energy grid tplot variable.
    scpot_name : str
        Name of the downsampled spacecraft potential tplot variable.
    species : str, optional
        Species to convert, 'e' for electrons or 'ion' for ions.
        Default is 'e'.

    Returns
    -------
    vbin : ndarray
        Array of velocity bins in km/s.
    grids_corr : ndarray
        Array of energy bins corrected for spacecraft potential in eV.
    msg : string
        String indicating whether the energy grid is interleaved or not. 
    """
    # Extract data
    _, nrgy_raw = get_data(nrgyraw_name)
    _, scpot_data = get_data(scpot_name)
    
    MASS_ENERGY = {'ion': physical_constants['proton mass energy equivalent in MeV'][0] * 1e6,
                   'e': physical_constants['electron mass energy equivalent in MeV'][0] * 1e6}
    CHARGES = {'ion': 1, 'e': -1}

    messages = {0: "The energy bins are interleaved!",
                1: "ERROR: odd slices inconsistent",
                2: "ERROR: even slices inconsistent",
                3: "The energy bins are instantaneous!",
                4: "The energy bins are non-interleaved!"}

    vbinfactor = MASS_ENERGY[species]
    q = CHARGES[species]

    interleave_check = is_interleaved(nrgy_raw)

    if interleave_check:
        # Check odd-even consistency if grid is interleaved
        oddcheck  = is_interleaved(nrgy_raw[1::2, :]) # odd-indexed rows (1,3,5,…)
        evencheck = is_interleaved(nrgy_raw[0::2, :])  # even-indexed rows (0,2,4,…)
        
        # if oddcheck and evencheck are False or 0 then the odd-even grid is consistent
        check_key = (2 * (evencheck)) + (oddcheck)

    else:
        check_key = 4
    
    msg = messages[check_key]
    print(msg)

    # Accounting for spacecraft potential correction
    nrgy_corr = nrgy_raw + (q*scpot_data[:, None])

    # Convert to velocity bins (km/s)
    vbin = np.sqrt(2.0 * nrgy_corr / vbinfactor) * c * 1e-3  # shape (1, 32) or (2, 32)
    return vbin, nrgy_corr, msg


def vbin_to_vv(vbin, phi_var, mean_phi=False):
    """
    Computes velocity vectors in the instrument frame.
    
    Parameters:
    - vbin: np.array of shape (ntime, 32)
    - phi_var: str name of phi tplot variable
    - mean_phi: bool, whether to use mean phi values or instantaneous phi values (default=False)
    
    Returns:
    - vv: np.array of shape (3, 16384, ntime)
    """
    # Constants
    N_PHI, N_THETA, N_ENERGY = 32, 16, 32
    DEG2RAD = np.pi / 180.0
    ntime = vbin.shape[0]

    # Compute Theta (16,) and reshape to (1, 16, 1)
    theta = (5.625 + 11.25 * np.arange(N_THETA)) * DEG2RAD
    sin_th = np.sin(theta)[None, :, None]
    cos_th = np.cos(theta)[None, :, None]

    # Extract Phi data and convert to radians
    _, phi_data = get_data(phi_var)
    phi_data *= DEG2RAD

    # Reshape vbin is (ntime, energy) -> (ntime, 1, 1, 32)
    v_mag = vbin[:, None, None, :] 

    if mean_phi:
        # Reshape phi to mean phi: (ntime,) -> (1, 32, 1, 1)
        phi_vec = np.mean(phi_data, axis=0)[None, :, None, None]   
    else:# Instantaneous mode
        # Reshape phi: (ntime, 32) -> (ntime, 32, 1, 1)
        phi_vec = phi_data[:, :, None, None]

    # Calculate Trig for Phi (ntime, 32, 1, 1)
    sin_phi = np.sin(phi_vec)
    cos_phi = np.cos(phi_vec)

    # Broadcast Multiplication: results in (ntime, 32, 16, 32)
    # Order: [time, phi, theta, energy]
    vx = v_mag * sin_th * cos_phi
    vy = v_mag * sin_th * sin_phi
    vz = v_mag * cos_th * np.ones((1, N_PHI, 1, 1)) # vz only depends on theta and energy mag

    # Flatten to (3, 16384, ntime)
    vv = np.stack([
        vx.reshape(ntime, -1).T,
        vy.reshape(ntime, -1).T,
        vz.reshape(ntime, -1).T
    ], axis=0)

    return -vv # Instrument looks 'outward', so velocity of plasma is negative of bin vector


def vi_bl_flip(idx, bulk_vi_gse):
    """
    Compute the bulk ion velocity at the B_L reversal time or reconnection time.

    Parameters:
    - idx: int; index of the B_L reversal point
    - bulk_vi_gse: str; name of the bulk ion velocity tplot variable

    Returns:
    - bulk_vi_rev: float bulk ion velocity at the B_L reversal point
    """
    _, bulk_vi_gse = get_data(bulk_vi_gse)
    # Find bulk ion velocity at B_L reversal:
    bulk_vi_rev = bulk_vi_gse[idx]
    print(f'Bulk ion velocity at B_L reversal: {bulk_vi_rev}')

    return bulk_vi_rev


def vv_to_recx(vv, vi):
    """
    Convert velocity vectors in gse frame to reconnection frame coordinates.

    Parameters:
    - vv: np.array of shape (3, nbin, npts); velocity vectors in instrument frame
    - vi: np.array of shape (3,); bulk ion velocity vector in reconnection frame

    Returns:
    - vv_recx: np.array of shape (3, nbin, npts); velocity vectors in reconnection frame
    """
    # Reshape vi so each component can broadcast across all bins and times
    vi_reshaped = vi[:, None, None]   # (3,1,1)

    # Subtract bulk ion flow from every velocity vector
    vv_recx = vv - vi_reshaped      # (3, nbin, npts)

    return vv_recx


def vv_to_fac(vv, fac_matrix):
    """
    Rotate velocity vectors to the FAC frame.

    Parameters:
    - vv : np.array(3, nbin, nsweeps). Velocity vectors (GSE/Reconnection frame). 
        nsweeps is 2 (interleaved), 1 (non-interleaved), or ntime (instantaneous).
    - fac_matrix: np.array of shape (ntime, 3, 3); FAC rotation matrix for each time

    Returns:
    - vv_fac: np.array of shape (3, nbin, npts); velocity vectors in FAC frame
    """

    ntime = fac_matrix.shape[0] # length of time window
    nsweeps = vv.shape[2]

    # Case 1: instantaneous (already time-aligned)
    if nsweeps == ntime:
        vv_time = vv

    # Case 2: interleaved (2 sweeps -> map by parity)
    elif nsweeps == 2:
        ip = np.arange(ntime) % 2
        vv_time = vv[:, :, ip]

    # Case 3: non-interleaved (1 sweep -> repeat for all times)
    elif nsweeps == 1:
        vv_time = np.repeat(vv, ntime, axis=2)

    else:
        raise ValueError("vv has invalid dimensions")
    
    # Rotate each time slice with its own FAC matrix
    vv_fac = np.einsum('tij, jbt -> ibt', fac_matrix, vv_time)

    return vv_fac


def vv_to_lmn(vv, lmn_matrix):
    """
    Rotate velocity vectors to the LMN frame.

    Parameters:
    - vv: np.array of shape (3, nbin, npts); velocity vectors in gse or reconnection frame
    - lmn_matrix: np.array of shape (1,3,3) or (3,3); LMN rotation matrix

    Returns:
    - vv_lmn: np.array of shape (3, nbin, npts); velocity vectors in LMN frame
    """
    # Accept either shape (1,3,3) or (3,3)
    if lmn_matrix.ndim == 3:
        R = lmn_matrix[0]      # shape (3,3)
    else:
        R = lmn_matrix         # shape (3,3)

    # Apply the same rotation to all sweeps
    vv_lmn = np.einsum('ij, jbt -> ibt', R, vv)

    return vv_lmn


def vv_to_vmap(vv_coord, vth_mean, vth_lim=3.5, bin_width_frac=0.1):
    """
    Map velocity vectors to bin indices in normalized velocity space.

    Parameters:
    - vv_coord: np.array of shape (3, nbin, npts); velocity vectors in a frame
    - vth_mean: float; mean thermal velocity for normalization
    - vth_lim: float, optional; limit of normalized velocity space (default: 3.5)
    - bin_width_frac: float, optional; fraction of v_range for bin width (default: 0.1)

    Returns:
    - vmap: np.array of shape (3, nbin, npts); bin indices of velocity vectors in normalized space
    - bin_centers: np.array of shape (n_vbins); physical bin centers in the original velocity space
    - bin_centers_norm: np.array of shape (n_vbins); normalized bin centers in the original velocity space
    - edges: np.array of shape (n_vbins+1); physical bin edges in the original velocity space
    - edges_norm: np.array of shape (n_vbins+1); normalized bin edges in the original velocity space
    - n_vbins: int; number of bins in the original velocity space
    """
    # Normalize velocities
    vv_norm = vv_coord / vth_mean

    # Normalized limits and bin width
    v_min = -vth_lim
    v_max =  vth_lim
    v_range = v_max - v_min

    dv = bin_width_frac
    # n_vbins = int(round(v_range / dv))
    n_vbins = int(np.ceil(v_range / dv))

    # Normalized edges and centers
    edges_norm = np.linspace(v_min, v_max, n_vbins + 1)
    bin_centers_norm = 0.5 * (edges_norm[:-1] + edges_norm[1:])

    # Physical edges/centers (for reference)
    edges = edges_norm * vth_mean
    bin_centers = bin_centers_norm * vth_mean

    # Map measurements to bins in normalized space
    vmap = np.full(vv_norm.shape, -999, dtype=int)
    mask = (vv_norm >= v_min) & (vv_norm < v_max)
    vmap[mask] = np.floor((vv_norm[mask] - v_min) / dv).astype(int)

    return vmap, bin_centers, bin_centers_norm, edges, edges_norm, n_vbins


def vv_to_vmap_3d(vv_coord, vth_mean, vth_lim=(3.5, 3.5, 3.5), 
                  bin_width_frac=(0.1, 0.1, 0.1)):
    """
    This is a wrapper function for `vv_to_vmap` that accepts bin limits and 
    widths in 3D. If the 3D accepts the same bin limits and widths, then one 
    should use 'vv_to_vmap' instead.

    Parameters:
    - vv_coord: np.array of shape (3, nbin, npts); velocity vectors in a frame
    - vth_mean: float; mean thermal velocity for normalization
    - vth_lim: tuple of 3 floats; limits of normalized velocity space for each 
        component (default: 3.5)
    - bin_width_frac: tuple of 3 floats; fraction of v_range for bin width for 
        each component (default : 0.1)

    Returns:
    - dict containing:
        "vmap": np.array of shape (3, nbin, npts); bin indices of velocity vectors in normalized space
        "centers": tuple of 3 np.arrays; physical bin centers in the original velocity space
        "centers_norm": tuple of 3 np.arrays; normalized bin centers in the original velocity space
        "edges": tuple of 3 np.arrays; physical bin edges in the original velocity space
        "edges_norm": tuple of 3 np.arrays; normalized bin edges in the original velocity space
        "nbins": tuple of 3 ints; number of bins in the original velocity space
    """
    # Unpack per-dimension parameters
    lim_x, lim_y, lim_z = vth_lim
    frac_x, frac_y, frac_z = bin_width_frac

    # Run original function for each component
    vmap_x, cx, cx_norm, ex, ex_norm, nx = vv_to_vmap(
        vv_coord[0], vth_mean, vth_lim=lim_x, bin_width_frac=frac_x
    )
    vmap_y, cy, cy_norm, ey, ey_norm, ny = vv_to_vmap(
        vv_coord[1], vth_mean, vth_lim=lim_y, bin_width_frac=frac_y
    )
    vmap_z, cz, cz_norm, ez, ez_norm, nz = vv_to_vmap(
        vv_coord[2], vth_mean, vth_lim=lim_z, bin_width_frac=frac_z
    )

    # Combine into a single array of shape (3, N, T)
    vmap_3d = np.stack([vmap_x, vmap_y, vmap_z], axis=0)

    return {
        "vmap": vmap_3d,
        "centers": (cx, cy, cz),
        "centers_norm": (cx_norm, cy_norm, cz_norm),
        "edges": (ex, ey, ez),
        "edges_norm": (ex_norm, ey_norm, ez_norm),
        "nbins": (nx, ny, nz)
    }


def compute_vbin_vol(vbin):
    """
    Compute velocity bin volumes of the device.

    This function takes velocity bin data from instrument as input, creates the 
    instrument bins and computes their solid volume. The computed volumes are 
    then reshaped to match the MMS energy index (16384 bins).

    Parameters:
    - vbin: np.array of shape (n_sweeps, n_vbin); velocity bin centers

    Returns:
    - vvol: np.array of shape (16384, n_sweeps); velocity bin volumes
    """
    # --- Compute velocity edges from centers ---   
    n_interleave = vbin.shape[0]   # 1 or 2
    n_vbin = vbin.shape[1] # usually 32
    v_centers = vbin * 1e3 # convert km/s to m/s

    v_edges = np.empty((n_interleave, n_vbin + 1)) # allocating array
    # interior edges
    v_edges[:, 1:-1] = 0.5 * (v_centers[:, :-1] + v_centers[:, 1:]) 
    # symmetric endpoints
    v_edges[:, 0]  = v_centers[:, 0]  - (v_edges[:, 1]  - v_centers[:, 0])
    v_edges[:, -1] = v_centers[:, -1] + (v_edges[:, -2] - v_centers[:, -1])

    # --- Compute theta edges (fixed instrument geometry) ---
    N_THETA = 16
    theta_edges = np.linspace(0, np.pi, N_THETA + 1)
    dcos = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])


    # --- Compute phi edges from phi centers ---
    N_PHI = 32
    phi_edges = np.linspace(0, 2*np.pi, N_PHI + 1)
    dphi = phi_edges[1:] - phi_edges[:-1]   # uniform = 2π/32

    # --- Compute volume term ---
    dv3 = (1.0/3.0) * (v_edges[:, 1:]**3 - v_edges[:, :-1]**3)  # (n_interleave, 32)

    # --- Broadcast to full 3D grid (energy × theta × phi) ---
    dv3_4d  = dv3[:, None, None, :]          # (n_interleave, 1, 1, 32)
    dcos_4d = dcos[None, None, :, None]      # (1, 1, 16, 1)
    dphi_4d = dphi[None, :, None, None]      # (1, 32, 1, 1)

    vvol_4d = dv3_4d * dcos_4d * dphi_4d     # (n_interleave, 32, 16, 32)

    # --- Reshape to MMS linear index (16384 bins) ---
    vvol = vvol_4d.reshape(n_interleave, 32*16*32).T  # (16384, n_interleave)\
    
    # --- Verification ---
    rmin = v_edges[0, 0]
    rmax = v_edges[0, -1]
    sphere_vol = (4.0/3.0)*np.pi*(rmax**3 - rmin**3)
    sum_vvol = vvol[:, 0].sum()
    print(f"Analytical shell volume = {sphere_vol:.7e}, sum of bin volumes = {sum_vvol:.7e}")

    return vvol