"""
================================================================================
Input/Output utilities for MMS data and results.

Functions:
- mms_name_make: Generate a standardized filename based on the given parameters.
- h5sav: Save data to an .h5 file and overwrites existing datasets.
- load_plt_mms_data: Load C_E and delta f for several probes, integrated over two
  velocity axes, for the 1D plots in plotting/plt_vdf_fpc.

Author: Regis John
Created: 2026-03-26
================================================================================
"""
import h5py
import numpy as np

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
    "eigvals": "nT^2",
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
    hfile (str): Path to the HDF5 file to be written.
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
                dset.attrs['units'] = active_units.get(key, "N/A")

    # --- Verification printout ---
    with h5py.File(hfile, "r") as f: 
        print("\nHDF5 write complete. Current file structure:\n") 
        for group_name in f.keys(): 
            print(f"[{group_name}] -> {list(f[group_name].keys())}")


def load_plt_mms_data(hfiles, time_indices):
    """
    Load FPC data for all probes and return integrated perp1, perp2, and parallel
    components for both C_E and dvdf, at a given time_index.
    """
    # If time_indices is just one int, convert to a list
    if isinstance(time_indices, int):
        time_indices = [time_indices] * len(hfiles)

    probes = []

    for hfile, time_index in zip(hfiles, time_indices):
        with h5py.File(hfile, "r") as f:
            c_vol = f["fac"]["c_vol"][...]          # (3,28,28,28,133)
            vpar   = f["fac"]["binc_n"][...]        # (28,)
            dvdf   = f["fac"]["bvdf_vol"][...]      # (28,28,28,133)
            tarr   = f["meta"]["time"][...]         # (133,)
            # Metadata used by the plot functions to name saved figures
            meta = {"trange": list(f["meta"]["trange"].asstr()[...]),
                    "data_rate": f["meta"]["data_rate"].asstr()[()],
                    "species": f["meta"]["species"].asstr()[()],
                    "bin_width_frac": float(f["meta"]["bin_width_frac"][()])}

        # Slice the 3 components of fpc and delta f at time_index
        c_perp1 = c_vol[0, :, :, :, time_index]
        c_perp2 = c_vol[1, :, :, :, time_index]
        c_par   = c_vol[2, :, :, :, time_index]

        dvdf_t = dvdf[:, :, :, time_index]

        # Integrate over two directions
        # C_perp1
        c_perp1_vperp1 = np.nansum(c_perp1, axis=(1, 2))
        c_perp1_vperp2 = np.nansum(c_perp1, axis=(0, 2))
        c_perp1_vpar   = np.nansum(c_perp1, axis=(0, 1))

        # C_perp2
        c_perp2_vperp1 = np.nansum(c_perp2, axis=(1, 2))
        c_perp2_vperp2 = np.nansum(c_perp2, axis=(0, 2))
        c_perp2_vpar   = np.nansum(c_perp2, axis=(0, 1))

        # C_parallel
        c_par_vperp1 = np.nansum(c_par, axis=(1, 2))
        c_par_vperp2 = np.nansum(c_par, axis=(0, 2))
        c_par_vpar   = np.nansum(c_par, axis=(0, 1))

        # delta vdf integration
        dvdf_vperp1 = np.nansum(dvdf_t, axis=(1, 2))
        dvdf_vperp2 = np.nansum(dvdf_t, axis=(0, 2))
        dvdf_vpar   = np.nansum(dvdf_t, axis=(0, 1))

        probes.append({

            "vpar": vpar,

            "c_e": {
                "c_perp1": {
                    "vperp1": c_perp1_vperp1,
                    "vperp2": c_perp1_vperp2,
                    "vpar":   c_perp1_vpar,
                },

                "c_perp2": {
                    "vperp1": c_perp2_vperp1,
                    "vperp2": c_perp2_vperp2,
                    "vpar":   c_perp2_vpar,
                },

                "c_par": {
                    "vperp1": c_par_vperp1,
                    "vperp2": c_par_vperp2,
                    "vpar":   c_par_vpar,
                },
            },
             "df": {
                "vperp1": dvdf_vperp1,
                "vperp2": dvdf_vperp2,
                "vpar":   dvdf_vpar,
            },

            "time": tarr[time_index],

            "meta": meta
        })

    return probes
