"""
================================================================================
Functions for computing J.E dot product from FPC data and rotating into FAC & LMN.

Functions:
- compute_jdotE: Compute J.E dot product from FPC data.
- jdotE_to_fac: Rotate J.E dot product into FAC coordinates.
- jdotE_to_lmn: Rotate J.E dot product into LMN coordinates.

Author: Regis John
Date: 2026-03-31
================================================================================
"""

import numpy as np
from pyspedas import get_data, store_data, set_coords, set_units

def compute_jdotE(jvec_var, evec_var):
    """
    Computes the J.E dot product of two vectors jvec and evec.

    Parameters:
    - jvec_var (str or numpy array): name of the jvec tplot variable or numpy array
    - evec_var (str or numpy array): name of the evec tplot variable or numpy array

    Returns:
    - times (numpy array): time array or None if inputs are NumPy arrays
    - jdotE (numpy array): total J·E
    - jdotE_0 (numpy array): J·E first component
    - jdotE_1 (numpy array): J·E second component
    - jdotE_2 (numpy array): J·E third component
    """
    if isinstance(jvec_var, str):
        # Extract data
        times, jvec_dat = get_data(jvec_var)
        _, evec_dat = get_data(evec_var)
    else:
        # If inputs are NumPy arrays, use them directly
        times = None
        jvec_dat = jvec_var
        evec_dat = evec_var 

    evec_dat = evec_dat*1e-3

    # Total J·E'
    jdotE = np.einsum('ij,ij->i', jvec_dat, evec_dat)

    # Split the dot product into components
    jdotE_0 = jvec_dat[:, 0] * evec_dat[:, 0]
    jdotE_1 = jvec_dat[:, 1] * evec_dat[:, 1]
    jdotE_2 = jvec_dat[:, 2] * evec_dat[:, 2]


    return times, jdotE, jdotE_0, jdotE_1, jdotE_2


def jdotE_fac(jvec_var, evec_var, 
              newnames=['jdotE', 'jdotE_perp1', 'jdotE_perp2', 'jdotE_par']):
    """
    Rotate the J.E dot product into FAC coordinates.

    Parameters:
    jvec_var (str): name of the jvec variable
    evec_var (str): name of the evec variable
    newnames (list, optional): list of names for the output variables, defaults to ['jdotE', 'jdotE_perp1', 'jdotE_perp2', 'jdotE_par']

    Returns:
    out_list (list): list of names for the output variables
    """
    times, jdotE, jdotE_0, jdotE_1, jdotE_2 = compute_jdotE(jvec_var, evec_var)

    # Total (coordinate‑invariant)
    store_data(newnames[0], data={'x': times, 'y': jdotE},
               attr_dict={'y': {'units': 'nW/m^3', 'label': "J·E' (total)"}})
    set_units(newnames[0], 'nW/m^3')

    # FAC components
    comp_data = [jdotE_0, jdotE_1, jdotE_2]
    comp_labels = ["Perp1", "Perp2", "Par"]

    for outname, arr, label in zip(newnames[1:], comp_data, comp_labels):
        store_data(outname, data={'x': times, 'y': arr},
            attr_dict={'y': {'units': 'nW/m^3', 'label': f"J·E' {label} (FAC)"}})
        set_coords(outname, 'FAC')
        set_units(outname, 'nW/m^3')
    
    # Combined tplot variables: total + three FAC components
    combined_matrix = np.column_stack([jdotE, jdotE_0, jdotE_1, jdotE_2])

    # Store it as a single numerical variable
    store_data('jdotE_fac_combined', data={'x': times, 'y': combined_matrix},
        attr_dict={'y': {'units': 'nW/m^3', 'label': ["J·E' (total)", 
                            comp_labels[0], comp_labels[1], comp_labels[2]]}})
    out_list = newnames + ['jdotE_fac_combined']

    return out_list


def jdotE_lmn(jvec_var, evec_var,
              newnames=['jdotE', 'jdotE_L', 'jdotE_M', 'jdotE_N']):
    """
    Rotate the J.E dot product into LMN coordinates.

    Parameters:
    - jvec_var (str): name of the jvec variable
    - evec_var (str): name of the evec variable
    - newnames (list, optional): list of names for the output variables, defaults to ['jdotE', 'jdotE_L', 'jdotE_M', 'jdotE_N']

    Returns:
    - out_list (list): list of names for the output variables
    """
    times, jdotE, jdotE_L, jdotE_M, jdotE_N = compute_jdotE(jvec_var, evec_var)

    # Total (coordinate‑invariant)
    store_data(newnames[0], data={'x': times, 'y': jdotE},
               attr_dict={'y': {'units': 'nW/m^3', 'label': "J·E' (total)"}})
    set_units(newnames[0], 'nW/m^3')

    # LMN components
    comp_data   = [jdotE_L, jdotE_M, jdotE_N]
    comp_labels = [r"J\cdot E'_{L}", r"J\cdot E'_{M}", r"J\cdot E'_{N}"]

    for outname, arr, label in zip(newnames[1:], comp_data, comp_labels):
        store_data(outname, data={'x': times, 'y': arr},
            attr_dict={'y': {'units': 'nW/m^3',
                             'label': f'{label} (LMN)'}})
        set_coords(outname, 'LMN')
        set_units(outname, 'nW/m^3')
    
     # Combined tplot variables: total + three FAC components
    combined_matrix = np.column_stack([jdotE, jdotE_L, jdotE_M, jdotE_N])

    # Store it as a single numerical variable
    store_data('jdotE_lmn_combined', data={'x': times, 'y': combined_matrix},
            attr_dict={'y': {'units': 'nW/m^3', 'legend_names': ["J·E' (total)", 
                            comp_labels[0], comp_labels[1], comp_labels[2]]}})
    out_list = newnames + ['jdotE_lmn_combined']

    return out_list