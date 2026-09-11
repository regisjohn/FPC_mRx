"""
================================================================================
MMS coordinate system transformations and frame rotation utilities.

Funcions:
- lmn_matrix_make: Compute the LMN matrix using Minimum Variance Analysis (MVA) 
    and fix the ambiguity in signs.
- rotate_to_fac: Rotate a tplot variable by a given FAC matrix.
- rotate_to_lmn: Rotate a tplot variable by a given LMN matrix.
- fac_lmn_proj: Compute projection matrices from FAC to LMN and LMN to FAC.

Author: Regis John
Created: 2026-03-26
================================================================================
"""

import numpy as np
from pyspedas import minvar_matrix_make, get_data, store_data, rotmat_set_coords


def lmn_matrix_make(tplot_var, t_range, t_slide=0, newname=None, **kwargs):
    """
    Compute the LMN matrix using Minimum Variance Analysis (MVA) and fix the 
    ambiguity in signs.

    Parameters
    ----------
    - `tplot_var`: str
        Name of the magnetic field tplot variable.
    - `t_range`: list of str
        Time range in which to compute the MVA matrix. Format is [t_start, t_stop].
    - `t_slide`: float, optional
        Time (in seconds) between each successive window start time. Defaults to 0 (single matrix).
    - `newname`: str, optional
        Name of the output tplot variable containing the LMN matrix. Defaults to 'lmn_matrix'.
    - `**kwargs`: dict, optional
        Additional keyword arguments to pass to the function `minvar_matrix_make`.

    Returns
    -------
    - `outname`: str
        Name of the output tplot variable containing the LMN matrix.
    """
    if not isinstance(tplot_var, str):
        raise TypeError(f"tplot_var must be a string (name of the variable), not {type(tplot_var)}")
    
    #--- Compute LMN matrix ----
    t_start = t_range[0]
    t_stop = t_range[1]

    minvar_matrix_make(
        in_var_name=tplot_var,
        tstart=t_start,
        tstop=t_stop,
        tslide=t_slide,
        newname='lmn_matrix_raw',
        **kwargs
    )

    #--- Fix signs ---
    # Retrieve the stored tplot variable 
    _, lmn_matrix_raw = get_data('lmn_matrix_raw')
    print(lmn_matrix_raw.shape)

    # Extract eigenvectors from 1×3×3 matrix
    L = lmn_matrix_raw[0][0]   # max variance
    M = lmn_matrix_raw[0][1]   # intermediate
    N = lmn_matrix_raw[0][2]   # min variance (normal)

    flip_N = 0
    flip_L = 0

    # 1. Fix x-component of N to be in +GSE X
    if N[0] < 0:
        N = -N
        flip_N = 1

    # 2. Fix z-component of L to be in +GSE Z
    if L[2] < 0:
        L = -L
        flip_L = 1
    
    # 3. To ensure right-handedness, we use XOR so M is flipped if either N or L is
    if flip_N ^ flip_L:
        M = -M
    
    # 4. Reassemble corrected LMN matrix
    lmn_matrix = np.vstack([L, M, N]).reshape(1, 3, 3)

    # 5. Name of MVA/LMN matrix tplot variable
    outname = newname if newname else 'lmn_matrix'

    # Storing data into a tplot variable and add metadata information.
    store_data(outname, data={'x': _, 'y': lmn_matrix})
    rotmat_set_coords(varname=outname, in_coords='GSE', out_coords='LMN')

    return outname


def rotate_to_fac(tplot_var, fac_mat_var):
    """
    Rotate a tplot variable using a given FAC matrix.

    Parameters:
    - tplot_var (str or numpy array): Name of the tplot variable or numpy array to be rotated.
    - fac_mat_var (str or numpy array): Name of the tplot variable or numpy array containing the FAC matrix.

    Returns:
    - times (np.ndarray): Time array of the input data or None if NumPy arrays provided.
    - tdata_rot (np.ndarray): Array of rotated data.
    """
    if isinstance(tplot_var, str):
        # Extract Data from tplot variables
        times, tdata = get_data(tplot_var)
        _, fac_matrix = get_data(fac_mat_var)
    else:
        # If inputs are NumPy arrays, use them directly
        times = None
        print('Input data are Numpy Arrays, no time array will be returned!')
        tdata = tplot_var
        fac_matrix = fac_mat_var 

    tdata_rot = np.einsum('tij, tj -> ti', fac_matrix, tdata)

    return times, tdata_rot


def rotate_to_lmn(tplot_var, lmn_mat_var):
    """
    Rotate a tplot variable using a given LMN matrix.

    Parameters:
    - tplot_var (str or numpy array): Name of the tplot variable or numpy array to be rotated.
    - lmn_mat_var (str or numpy array): Name of the tplot variable or numpy array containing the LMN matrix.

    Returns:
    - times (np.ndarray): Time array of the input data or None if NumPy arrays provided.
    - tdata_rot (np.ndarray): Array of rotated data.
    """
    if isinstance(tplot_var, str):
        # Extract Data from tplot variables
        times, tdata = get_data(tplot_var)
        _, lmn_matrix = get_data(lmn_mat_var)
    else:
        # If inputs are NumPy arrays, use them directly
        times = None
        print('Input data are Numpy Arrays, no time array will be returned!')
        tdata = tplot_var
        lmn_matrix = lmn_mat_var 

    # Handle (1,3,3) vs (3,3)
    if lmn_matrix.ndim == 3:
        R = lmn_matrix[0]
    else:
        R = lmn_matrix

    # Static rotation
    tdata_rot = np.einsum('ij, tj -> ti', R, tdata)
    
    return times, tdata_rot


def fac_lmn_proj(fac_matrix_var, lmn_matrix_var):
    """
    Compute projection matrices from FAC to LMN and LMN to FAC.

    Parameters:
    - fac_matrix_var (str): Name of the tplot variable containing the FAC matrix.
    - lmn_matrix_var (str): Name of the tplot variable containing the LMN matrix.

    Returns:
    - fac2lmn_proj (np.ndarray): Projection matrix from FAC to LMN.
    - lmn2fac_proj (np.ndarray): Projection matrix from LMN to FAC.
    """
    # Load data
    _, fac_matrix = get_data(fac_matrix_var)   # shape (N,3,3)
    _, lmn_matrix = get_data(lmn_matrix_var)   # shape (1,3,3)

    # LMN is constant, extract the single 3×3 matrix
    R_lmn = lmn_matrix[0] if lmn_matrix.ndim == 3 else lmn_matrix
    
    # Precompute transpose
    R_lmn_T = R_lmn.T
    fac_T = fac_matrix.transpose(0, 2, 1)  # (N,3,3)

    # FAC to LMN
    fac2lmn_proj = fac_matrix @ R_lmn_T

    # LMN to FAC
    lmn2fac_proj = R_lmn @ fac_T

    return fac2lmn_proj, lmn2fac_proj