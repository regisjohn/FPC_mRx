"""
================================================================================
Input/Output utilities for MMS data and results.

Functions:
- mms_name_make: Generate a standardized filename based on the given parameters.
- h5sav: Save data to an .h5 file and overwrites existing datasets.

Author: Regis John
Created: 2026-03-26
================================================================================
"""
import h5py

def mms_name_make(prefix, tstart, tend, ext="h5"):
    """
    Generate a standardized filename based on the given parameters.

    Parameters
    ----------
    prefix : str
        Prefix of the filename.
    tstart : str
        Start time of the data segment in the format 'YYYY-MM-DD/hh:mm:ss'.
    tend : str
        End time of the data segment in the format 'YYYY-MM-DD/hh:mm:ss'.
    ext : str, optional
        File extension of the generated filename. Default is 'h5'.

    Returns
    -------
    str
        The standardized filename.

    """
    date = tstart.split('/')[0].replace('-', '')
    s_time = tstart.split('/')[1].replace(':', '')
    e_time = tend.split('/')[1].replace(':', '')
    return f"{prefix}_{date}_{s_time}_{e_time}.{ext}"


# Standard MMS default units map
UNITS_MAP = {
    "jvec": "nA/m^2", 
    "jdotE": "nW/m^3", 
    "bvec": "nT", 
    "bmag": "nT",
    "evec_lor": "mV/m", 
    "den_e": "cm^-3", 
    "den_i": "cm^-3",
    "bulk_ve": "km/s", 
    "bulk_vi": "km/s",
    "te_para": "eV", "te_perp": "eV", 
    "ti_para": "eV", "ti_perp": "eV",
    "energy_e": "eV/(cm^2 s sr eV)", "energy_i": "eV/(cm^2 s sr eV)"
}

def h5sav(hfile, groups_dict, units_override=None):
    """
    Saves data to an .h5 file and overwrites existing datasets.

    Parameters:
    href (str): Path to the HDF5 file to be written.
    groups_dict (dict): Dictionary where the keys are group names and the values 
        are inner dictionary containing the data to be written.
    units_override (dict, optional): A dictionary specifying variable names as 
        keys and their desired override unit formats as values. Defaults to 
        None, using standard MMS units if no overrides specified.

    Returns:
    None
    The function prints out the current file structure after writing is complete.

    Example use: `h5sav(hfile, dat_grps, units_override={"den_i": "m^-3"})`
    """
    # Automatically merge default units with any provided overrides
    active_units = {**UNITS_MAP, **(units_override or {})}

    with h5py.File(hfile, "a") as f:
        for group_name, data_dict in groups_dict.items():

            grp = f[group_name] if group_name in f else f.create_group(group_name)

            for key, data in data_dict.items():
                if key in grp:
                    del grp[key]      # overwrite cleanly

                # Create dataset and assign unit attribute if key matches map
                dset = grp.create_dataset(key, data=data)
                if key in active_units:
                    dset.attrs['units'] = active_units[key]

    # --- Verification printout ---
    with h5py.File(hfile, "r") as f: 
        print("\nHDF5 write complete. Current file structure:\n") 
        for group_name in f.keys(): 
            print(f"[{group_name}] -> {list(f[group_name].keys())}")