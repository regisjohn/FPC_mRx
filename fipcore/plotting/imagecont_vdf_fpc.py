"""
================================================================================
12 Panel Plotting functions for MMS VDF and FPC data.

Functions:
- imagecont_vdf_fpc: Generate 12 panel subplots of VDF and FPC components.
- imagecont_vdf_fpc_panel: Helper function for imagecont_vdf_fpc.

Author: Regis John
Created: 2026-03-31
================================================================================
"""

import h5py
from pyspedas import time_string
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
import os

from fipcore.utils.helper_utils import (mms_name_make, slice3d_to_2d, 
                                    global_vmin_vmax_vdf, global_vmin_vmax_fpc)
from fipcore.plotting.imagecont_vdf import imagecont_vdf
from fipcore.plotting.imagecont_fpc import imagecont_fpc


def imagecont_vdf_fpc(trange, species, bin_width_frac, time_index=0,
                          coord_type='both', suptitle_sfx=None, dpi_val=100, 
                          axis_fntsz=16, cbar_ht=0.050, save_fig=False, 
                          outdir_sfx='', show_tind=True, subtract_f0=False, **kwargs):
    """
    Master function to generate 12 panel subplots of VDF and FPC components for 
    a given time index with the VDFs on the top row and the FPCs on the bottom 
    three rows.

    Parameters:
    - trange (list of str): start time, end time] in the format:
            ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss'].
    - species (str): species name ('e' or 'i').
    - bin_width_frac (float): bin width as a fraction of thermal velocity.
    - time_index (int): time index for the plot.
    - coord_type (str): coordinate system to use ('fac', 'lmn', or 'both').
    - suptitle_sfx (str or None): suffix for the suptitle.
    - dpi_val (int): DPI value for the plot.
    - axis_fntsz (int): font size for the axis labels.
    - cbar_ht (float): height of the colorbar.
    - save_fig (bool): if True, save the figure.
    - outdir_sfx (str): suffix for the output directory.
    - show_tind (bool): if True, show the time index in the plot at the top right corner.
    - subtract_f0 (bool): if True, use the df suffix in the filename. Default is False.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object.
    """
    if suptitle_sfx is None:
        suptitle_sfx = f"with {bin_width_frac} binwidth"

    # Data path setup
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    type_tag = 'df' if subtract_f0 else 'f'
    hfile_pref = f'fpc_mrx_{type_tag}_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, mms_name_make(hfile_pref, trange[0], trange[1]))

    with h5py.File(hfile, "r") as f:
        # Global Metadata
        time = f["meta"]["time"][:]
        species = f["meta"]["species"].asstr()[()]
        probe = f["meta"]["probe"].asstr()[()]
        title_time = time_string(time[time_index])[:-3]
        species_label = 'Electron' if species == 'e' else 'Ion'

        # Determine which coordinates to process
        c_types = ["fac", "lmn"] if coord_type == "both" else [coord_type]

        for ct in c_types:
            grp = f[ct]

            # Dynamic Labels based on Coordinate System
            if ct == "fac":
                ax_lbls = [r"$v_{\perp 1}/v_{te}$", r"$v_{\perp 2}/v_{te}$", 
                           r"$v_{\parallel}/v_{te}$"]
                subs = [r"{\perp 1}", r"{\perp 2}", r"{\parallel}"]
                row_lbls = [rf"$\mathrm{{log}}(f_{{{species}}})$", 
                    r"$C_{\mathrm{E}\perp 1}$", r"$C_{\mathrm{E}\perp 2}$", 
                    r"$C_{\mathrm{E}\parallel}$"]
            else:
                ax_lbls = [r"$v_L$", r"$v_M$", r"$v_N$"]
                subs = ["L", "M", "N"]
                row_lbls = [rf"$\mathrm{{log}}(f_{{{species}}})$", 
                    r"$C_{\mathrm{E}_L}$", r"$C_{\mathrm{E}_M}$", 
                    r"$C_{\mathrm{E}_N}$"]

            # Data Loading
            vdf_vol = grp["bvdf_vol"][:, :, :, time_index]
            c_vol = grp["c_vol"][:, :, :, :, time_index]
            je = grp["JE_tot"][:, time_index]
            xc = grp["binc_n"][:]
            yc = grp["binc_n"][:]
            zc = grp["binc_n"][:]

            # Slicing and Prep
            vdf_xy, vdf_yz, vdf_xz = slice3d_to_2d(vdf_vol)
            vdf_2d_list = (vdf_xy, vdf_xz, vdf_yz.T) # The order is xy, xz and zy
            cx_xy, cx_yz, cx_xz = slice3d_to_2d(c_vol[0])
            cy_xy, cy_yz, cy_xz = slice3d_to_2d(c_vol[1])
            cz_xy, cz_yz, cz_xz = slice3d_to_2d(c_vol[2])
            cx_2d_list = (cx_xy, cx_xz, cx_yz.T) # The order is xy, xz and zy
            cy_2d_list = (cy_xy, cy_xz, cy_yz.T) # The order is xy, xz and zy
            cz_2d_list = (cz_xy, cz_xz, cz_yz.T) # The order is xy, xz and zy   

            # Axes labels and limits
            ax_pairs = ((xc, yc), (xc, zc), (zc, yc))
            axlabel_pairs = ((ax_lbls[0], ax_lbls[1]), (ax_lbls[0], ax_lbls[2]), 
                    (ax_lbls[2], ax_lbls[1])) # xy, xz, zy
            diag_lbls = [rf"$J_{subs[i]}E_{subs[i]} = {je[i]:.2e}$" for i in range(3)]
            
            # Call rendering function
            fig, axes = imagecont_vdf_fpc_panel(vdf_2d_list, cx_2d_list, cy_2d_list, cz_2d_list, 
                        ax_pairs, axlabel_pairs, row_lbls, diag_lbls, dpi_val, 
                        axis_fntsz, cbar_ht)
            
            # Title and Save
            suptitle = (f"{species_label} VDF and FPC components ({ct.upper()}) " 
                            f"{suptitle_sfx}\n MMS{probe} — {title_time}")
            fig.suptitle(suptitle, fontsize = 24)
            if show_tind:
                fig.text(0.98, 0.98, f"t: {time_index}", ha="right", va="top", fontsize=18)

            if save_fig:
                plot_name = f"fpc_mrx_{species}_{ct}_{bin_width_frac:.2f}_{time_index}.png"
                sfx_str = f"_{outdir_sfx}" if outdir_sfx else ""
                plot_dir = f'plots_fac{sfx_str}'
                output_dir = os.path.join(project_root, plot_dir)
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(os.path.join(output_dir, plot_name), dpi=dpi_val, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

def imagecont_vdf_fpc_panel(vdf_2d, cx_2d, cy_2d, cz_2d, ax_pairs, axlabels_pairs, 
                            row_labels, diag_labels, dpi_val=100, axis_fntsz=16, 
                            cbar_ht=0.047):
    """
    Render helper function to the above imagecont_vdf_fpc_panel function which plots 
    a 4x3 panel of VDF and FPC components.

    Parameters:
    - vdf_2d (tuple of length 3 of 2D arrays): The VDF data.
    - cx_2d (tuple of length 3 of 2D arrays): The Cx FPC component data.
    - cy_2d (tuple of length 3 of 2D arrays): The Cy FPC component data.
    - cz_2d (tuple of length 3 of 2D arrays): The Cz FPC component data.
    - ax_pairs (tuple of tuples of length 3 of 2-tuples of 1D arrays): The pairs of axes for the plots.
    - axlabels_pairs (tuple of tuples of length 3 of 2-tuples of strings): The labels for the axes.
    - row_labels (tuple of strings): The labels for the rows.
    - diag_labels (tuple of strings): The labels for the diagonal plots.
    - dpi_val (int): The DPI for the figure.
    - axis_fntsz (int): The fontsize for the axis labels.
    - cbar_ht (float): The height of the colorbars.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 18), dpi=dpi_val, constrained_layout=True)

    # --- Row 1: VDF top row ---
    for i in range(3):
        vmin, vmax = global_vmin_vmax_vdf(vdf_2d[i])
        ax1, ax2 = ax_pairs[i]
        _, ylabel = axlabels_pairs[i]
        im1, _ = imagecont_vdf(ax1, ax2, vdf_2d[i], xlabel=None, ylabel=ylabel, 
                           ax=axes[0,i], vmin=vmin, vmax=vmax, axis_fntsz=axis_fntsz)
        height, width = len(ax2), len(ax1)
        cbar = fig.colorbar(im1, ax=axes[0,i], fraction=cbar_ht*height/width, pad=0.08)
        cbar.set_label(r"[m$^{-3}$]")
        axes[0,0].text(-0.4, 0.85, row_labels[0], transform=axes[0,0].transAxes, 
                va='center', ha='left', rotation='horizontal', fontsize=20)


    # --- Row 2: Cx ---
    for j in range(3):
        vmin, vmax = global_vmin_vmax_fpc(cx_2d[j])
        ax1, ax2 = ax_pairs[j]
        _, ylabel = axlabels_pairs[j]
        im2, _ = imagecont_fpc(ax1, ax2, cx_2d[j], xlabel=None, ylabel=ylabel, 
                           ax=axes[1,j], vmin=vmin, vmax=vmax, axis_fntsz=axis_fntsz)
        height, width = len(ax2), len(ax1)
        cbar = fig.colorbar(im2, ax=axes[1,j], fraction=cbar_ht*height/width, pad=0.005)
        cbar.set_label(r"[W m$^{-3}$]")
        axes[1,0].text(-0.4, 0.85, row_labels[1], transform=axes[1,0].transAxes, 
                va='center', ha='left', rotation='horizontal', fontsize=20)
        axes[1,2].text(0.05, 0.95, diag_labels[0], transform=axes[1,2].transAxes, 
                       ha="left", va="top", fontsize=14, color="black")
        

    # --- Row 3: Cy ---
    for k in range(3):
        vmin, vmax = global_vmin_vmax_fpc(cy_2d[k])
        ax1, ax2 = ax_pairs[k]
        _, ylabel = axlabels_pairs[k]
        im3, _ = imagecont_fpc(ax1, ax2, cy_2d[k], xlabel=None, ylabel=ylabel, 
                           ax=axes[2,k], vmin=vmin, vmax=vmax, axis_fntsz=axis_fntsz)
        height, width = len(ax2), len(ax1)
        cbar = fig.colorbar(im3, ax=axes[2,k], fraction=cbar_ht*height/width, pad=0.005)
        cbar.set_label(r"[W m$^{-3}$]")
        axes[2,0].text(-0.4, 0.85, row_labels[2], transform=axes[2,0].transAxes, 
                va='center', ha='left', rotation='horizontal', fontsize=20)
        axes[2,1].text(0.05, 0.95, diag_labels[1], transform=axes[2,1].transAxes, 
                       ha="left", va="top", fontsize=14, color="black")

    # --- Row 4: Cz ---
    for l in range(3):
        vmin, vmax = global_vmin_vmax_fpc(cz_2d[l])
        ax1, ax2 = ax_pairs[l]
        xlabel, ylabel = axlabels_pairs[l]
        im4, _ = imagecont_fpc(ax1, ax2, cz_2d[l], xlabel=xlabel, ylabel=ylabel, 
                           ax=axes[3,l], vmin=vmin, vmax=vmax, axis_fntsz=axis_fntsz)
        height, width = len(ax2), len(ax1)
        cbar = fig.colorbar(im4, ax=axes[3,l], fraction=cbar_ht*height/width, pad=0.005)
        cbar.set_label(r"[W m$^{-3}$]")
        axes[3,0].text(-0.4, 0.85, row_labels[3], transform=axes[3,0].transAxes, 
                va='center', ha='left', rotation='horizontal', fontsize=20)
        axes[3,0].text(0.05, 0.95, diag_labels[2], transform=axes[3,0].transAxes, 
                       ha="left", va="top", fontsize=14, color="black")
        
    return fig, axes


def imagecont_vdf_fpcfold(trange, species, bin_width_frac, time_index=0,
                          coord_type='fac', suptitle_sfx=None, dpi_val=100, 
                          axis_fntsz=16, cbar_ht=0.050, save_fig=False, 
                          outdir_sfx='', show_tind=True, subtract_f0=False, **kwargs):
    """
    Master function to generate 12 panel subplots of VDF and FPC components for 
    a given time index with the VDFs on the top row and the FPCs on the bottom 
    three rows.

    Parameters:
    - trange (list of str): start time, end time] in the format:
            ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss'].
    - species (str): species name ('e' or 'i').
    - bin_width_frac (float): bin width as a fraction of thermal velocity.
    - time_index (int): time index for the plot.
    - coord_type (str): coordinate system to use ('fac', 'lmn', or 'both').
    - suptitle_sfx (str or None): suffix for the suptitle.
    - dpi_val (int): DPI value for the plot.
    - axis_fntsz (int): font size for the axis labels.
    - cbar_ht (float): height of the colorbar.
    - save_fig (bool): if True, save the figure.
    - outdir_sfx (str): suffix for the output directory.
    - show_tind (bool): if True, show the time index in the plot at the top right corner.
    - subtract_f0 (bool): if True, use the df suffix in the filename. Default is False.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object.
    """
    if suptitle_sfx is None:
        suptitle_sfx = f"with {bin_width_frac} binwidth"

    # Data path setup
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    type_tag = 'df' if subtract_f0 else 'f'
    hfile_pref = f'fpc_mrx_{type_tag}_{species}_{bin_width_frac:.2f}'
    hfile = os.path.join(data_dir, mms_name_make(hfile_pref, trange[0], trange[1]))

    with h5py.File(hfile, "r") as f:
        # Global Metadata
        time = f["meta"]["time"][:]
        species = f["meta"]["species"].asstr()[()]
        probe = f["meta"]["probe"].asstr()[()]
        title_time = time_string(time[time_index])[:-3]
        species_label = 'Electron' if species == 'e' else 'Ion'

        # Accessing the fac data group
        c_types = "fac"
        grp = f[c_types]

        # FAC Labels of Coordinate System
        ax_lbls = [r"$v_{\perp 1}/v_{te}$", r"$v_{\perp 2}/v_{te}$", 
                    r"$v_{\parallel}/v_{te}$"]
        subs = [r"{\perp 1}", r"{\perp 2}", r"{\parallel}"]
        row_lbls = [rf"$\mathrm{{log}}(f_{{{species}}})$", 
            r"$C_{\mathrm{E}\perp 1}$", r"$C_{\mathrm{E}\perp 2}$", 
            r"$C_{\mathrm{E}\parallel}$"]

        # Data Loading
        vdf_vol = grp["bvdf_vol"][:, :, :, time_index]
        c_vol = grp["c_vol"][:, :, :, :, time_index]
        cx_folds = grp["cx_folds"][:, :, :, time_index]
        cy_folds = grp["cy_folds"][:, :, :, time_index]
        cz_folds = grp["cz_folds"][:, :, :, time_index]
        je = grp["JE_tot"][:, time_index]
        xc = grp["binc_n"][:]
        yc = grp["binc_n"][:]
        zc = grp["binc_n"][:]

        # Slicing and Prep
        vdf_xy, vdf_yz, vdf_xz = slice3d_to_2d(vdf_vol)
        vdf_2d_list = (vdf_xy, vdf_xz, vdf_yz.T) # The order is xy, xz and zy
        cx_xy, cx_yz, cx_xz = slice3d_to_2d(c_vol[0])
        cy_xy, cy_yz, cy_xz = slice3d_to_2d(c_vol[1])
        cz_xy, cz_yz, cz_xz = slice3d_to_2d(c_vol[2])
        cx_2d_list = (cx_folds[0, :, :], cx_folds[1, :, :], cx_yz.T) # The order is xy, xz and zy
        cy_2d_list = (cy_folds[0, :, :], cy_xz, cy_folds[1, :, :]) # The order is xy, xz and zy
        cz_2d_list = (cz_xy, cz_folds[0, :, :], cz_folds[1, :, :]) # The order is xy, xz and zy   

        # Axes labels and limits
        ax_pairs = ((xc, yc), (xc, zc), (zc, yc))
        axlabel_pairs = ((ax_lbls[0], ax_lbls[1]), (ax_lbls[0], ax_lbls[2]), 
                (ax_lbls[2], ax_lbls[1])) # xy, xz, zy
        diag_lbls = [rf"$J_{subs[i]}E_{subs[i]} = {je[i]:.2e}$" for i in range(3)]
        
        # Call rendering function
        fig, axes = imagecont_vdf_fpc_panel(vdf_2d_list, cx_2d_list, cy_2d_list, cz_2d_list, 
                    ax_pairs, axlabel_pairs, row_lbls, diag_lbls, dpi_val, 
                    axis_fntsz, cbar_ht)
        
        # Title and Save
        suptitle = (f"{species_label} VDF and FPC components ({c_types.upper()}) " 
                        f"{suptitle_sfx}\n MMS{probe} — {title_time}")
        fig.suptitle(suptitle, fontsize = 24)
        if show_tind:
            fig.text(0.98, 0.98, f"t: {time_index}", ha="right", va="top", fontsize=18)

        if save_fig:
            plot_name = f"fpc_mrx_{species}_{c_types}_{bin_width_frac:.2f}_{time_index}.png"
            sfx_str = f"_{outdir_sfx}" if outdir_sfx else ""
            plot_dir = f'plots_fac{sfx_str}'
            output_dir = os.path.join(project_root, plot_dir)
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, plot_name), dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()