"""
================================================================================
Master function to compute the folded FPC Analysis of MMS.

Functions:
- fpc_mrx_fold: Function to compute the folded FPC and save to h5 file.

Author: Regis John
Created: 2026-04-29
================================================================================
"""

import numpy as np
import h5py
import os

from fipcore.utils.io_utils import mms_name_make, h5sav
from fipcore.utils.helper_utils import slice3d_to_2d


def fpc_mrx_fold(trange, species='e', bin_width_frac=0.25, coord_type="fac"):
    """
    Master function to compute folded FPC signature for a given time range
    and species using the existing fpc data.

    Parameters:
    - trange (list of str): [start time, end time] in the format:
        ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
    - species (str): Species of particle to analyze. Default is 'e'. 
    - bin_width_frac (float): Fraction of the thermal velocity used to 
        determine the bin width. Default is 0.25.   
    - coord_type (str): The type of coordinate system to use. Default is "fac".

    Returns:
    - hfile (str): Full path of the .h5 file containing the folded FPC data.
    """


    # Data path setup
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    hfile_pref = f'mms_fpc_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, mms_name_make(hfile_pref, trange[0], trange[1]))


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
        cx_xy, cx_yz, cx_xz = slice3d_to_2d(c_vol[0,:,:,:,t])
        cy_xy, cy_yz, cy_xz = slice3d_to_2d(c_vol[1,:,:,:,t])
        cz_xy, cz_yz, cz_xz = slice3d_to_2d(c_vol[2,:,:,:,t])
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
    
    # --- Saving to a .h5 file ---
    dat_grps = {
        "fac": {
            "cx_folds": cx_folds,
            "cy_folds": cy_folds,
            "cz_folds": cz_folds,
            }
        }
    h5sav(hfile, dat_grps)

    print(f"Data saved to file: {hfile}")
    
    return hfile