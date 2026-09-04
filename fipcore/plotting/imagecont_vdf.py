"""
================================================================================
Plotting functions for MMS VDF data.

Functions:
- imagecont_vdf: Image and Contour Plots of VDF.
- imagecont_vdf_fac: Image and Contour Subplots (1x3) of VDFs in FAC.
- imagecont_vdf_lmn: Image and Contour Subplots (1x3) of VDFs in LMN.

Author: Regis John
Created: 2026-03-26
================================================================================
"""

import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
import numpy as np
from pyspedas import time_string
from fipcore.utils.helper_utils import slice3d_to_2d_fac, global_vmin_vmax_vdf, slice3d_to_2d

def imagecont_vdf(xc, yc, vdf_2d, ax = None, vmin=None, vmax=None, title="", xlabel='X',
                    ylabel='Y', xlim=3, ylim=3, cmap=None, contour_levels=20, 
                    title_fntsz = 16, axis_fntsz = 12, tick_fntsz = 12, 
                    cbar_fntsz = 10, alpha_val = 1.0, dpi_val=100, 
                    subtract_f0=False):
    """
    Image and Contour Plots of VDF.

    Parameters:
    - xc (np.ndarray): The x-coordinates of the data points.
    - yc (np.ndarray): The y-coordinates of the data points.
    - vdf_2d (np.ndarray): The 2D VDF data.
    - ax (matplotlib.axes.Axes, optional): The axis object. If not provided, 
        a new figure and axes will be created.
    - vmin (float, optional): The minimum value for the colormap. Default is None.
    - vmax (float, optional): The maximum value for the colormap. Default is None.
    - title (str, optional): The title of the plot. Default is "".
    - xlabel (str, optional): The label of the x-axis. Default is "X".
    - ylabel (str, optional): The label of the y-axis. Default is "Y".
    - xlim (float, optional): The x-axis limit. Default is 3.
    - ylim (float, optional): The y-axis limit. Default is 3.
    - cmap (str, optional): The colormap to use, default is None. If none is 
        provided it will use 'jet' for full vdf and 'seismic' for subtracted vdf.
    - contour_levels (int, optional): The number of contour levels. Default is 20.
    - title_fntsz (int, optional): The font size of the title. Default is 16.
    - axis_fntsz (int, optional): The font size of the axis labels. Default is 12.
    - tick_fntsz (int, optional): The font size of the axis ticks. Default is 12.
    - cbar_fntsz (int, optional): The font size of the colorbar ticks. Default is 10.
    - alpha_val (float, optional): The alpha value of the contours. Default is 1.0.
    - dpi_val (int, optional): The dots per inch of the figure, default is 100.
    - subtract_f0 (bool, optional): Whether the vdf is subtracted one or full one, 
        default is False.

    Returns:
    - im (matplotlib.image.AxesImage): The image object.
    - ax (matplotlib.axes.Axes): The axis object.
    """
    # Create axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5), dpi=dpi_val)
        standalone = True
    else:
        standalone = False

    if subtract_f0:
        # Linear scale for delta vdf
        C = vdf_2d
        if cmap is None:
            cmap = "seismic"
    else:
        # Log scale for full vdf
        C = np.log10(np.where(vdf_2d > 0, vdf_2d, np.nan))
        if cmap is None:
            cmap = "jet"

    im = ax.imshow(
        C.T,
        origin="lower",
        extent=[xc[0], xc[-1], yc[0], yc[-1]],
        aspect="equal",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    
    # Contours
    ax.contour(
        xc, yc, C.T,
        levels=contour_levels,
        colors="black",
        linewidths=0.6,
        alpha=alpha_val,
    )

    ax.set_xlabel(xlabel, fontsize = axis_fntsz )
    ax.set_ylabel(ylabel, fontsize = axis_fntsz)
    ax.set_title(title, fontsize = title_fntsz)
    ax.tick_params(axis='both', which='major', labelsize=tick_fntsz)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    if standalone:
        ax.cbar = plt.colorbar(im, ax=ax)
        ax.cbar.set_label("log10(f)", fontsize = cbar_fntsz)
        plt.tight_layout()
        plt.show()

    return im, ax


def imagecont_vdf_fac(vdfbindat, xc, yc, zc, time_index, time, 
                      title_prefix = "Weighted VDF (FAC)", 
                      sub_titles=(None, None, None), coord_labels=None,
                      mode="integrate", slice_index=None, 
                      cbar_label=r"$\log_{10}(f\,\Delta v^3)\;[1/\mathrm{m}^3]$", 
                      *args, **kwargs):
    """
    Image and Contour Subplots (1x3) of VDFs in FAC. This is a wrapper for 
    imagecont_vdf function.

    Parameters:
    - vdfbindat (np.ndarray): 3D VDF data.
    - xc (np.ndarray): The x-coordinates of the data points.
    - yc (np.ndarray): The y-coordinates of the data points.
    - zc (np.ndarray): The z-coordinates of the data points.
    - time_index (int): The index of the time array to be used for slicing.
    - time (np.ndarray): The time array.
    - title_prefix (str, optional): The prefix of the title. Default is "Weighted VDF (FAC)".
    - sub_titles (tuple, optional): The individual titles of the subplots. 
        Default is (None, None, None).
    - coord_labels (tuple, optional): The labels of the coordinates. Default 
        coordinate labels correspond to v_perp1, v_perp2, and v_parallel normalized by v_te.
    - mode (str, optional): Options are "integrate" and "slice". Default is "integrate".
    - slice_index (int, optional): The index of the slice to be used for 
        plotting if mode is "slice". Default is None.
    - cbar_label (str, optional): The label of the colorbar. Default is 
        log_{10}(f*Delta v^3)[m^-3]".
    - *args: Additional arguments to be passed to imagecont_vdf.
    - **kwargs: Additional keyword arguments to be passed to imagecont_vdf.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - axes (list): The list of axes objects.
    """
    # Default coordinate labels:
    if coord_labels is None:
        coord_labels = (r"$v_{\perp 1}/v_{te}$", r"$v_{\perp 2}/v_{te}$", r"$v_{\parallel}/v_{te}$")

    # Extract slices on the fly
    vdf_xy, vdf_zy, vdf_zx = slice3d_to_2d_fac(vdfbindat, time_index, 
                                            mode=mode, slice_index=slice_index)

    # Compute global vmin/vmax
    vmin, vmax = global_vmin_vmax_vdf(vdf_xy, vdf_zy, vdf_zx)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

    im1, _ = imagecont_vdf(xc, yc, vdf_xy, title=sub_titles[0], 
                           xlabel=coord_labels[0], ylabel=coord_labels[1], 
                           ax=axes[0], vmin=vmin, vmax=vmax, *args, **kwargs)

    im2, _ = imagecont_vdf(yc, zc, vdf_zy, title=sub_titles[1], 
                           xlabel=coord_labels[2], ylabel=coord_labels[1], 
                           ax=axes[1], vmin=vmin, vmax=vmax, *args, **kwargs)

    im3, _ = imagecont_vdf(xc, zc, vdf_zx, title=sub_titles[2], 
                           xlabel=coord_labels[2], ylabel=coord_labels[0], 
                           ax=axes[2], vmin=vmin, vmax=vmax, *args, **kwargs)
    # Colorbar for all subplots
    factor = 0.046/(3 + 0.15)
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=factor, pad=0.04)
    cbar.set_label(cbar_label, fontsize=kwargs.get("cbar_fntsz", 10))
    # Title
    title_time = time_string(time[time_index])[:-3]
    fig.suptitle(f"{title_prefix} at {title_time}", fontsize = 16)
    
    plt.show()
    return fig, axes


def imagecont_vdf_lmn(vdfbindat, xc, yc, zc, time_index, time, 
                      title_prefix = "Raw VDF (LMN)", 
                      sub_titles=(None, None, None), coord_labels=None,
                      mode="integrate", slice_index=None, 
                      cbar_label=r"$\log_{10}(f\,\Delta v^3)\;[1/\mathrm{m}^3]$", 
                      *args, **kwargs):
    """
    Image and Contour Subplots (1x3) of VDFs in LMN. This is a wrapper for 
    imagecont_vdf.

    Parameters:
    - vdfbindat (np.ndarray): 3D VDF data.
    - xc (np.ndarray): The x-coordinates of the data points.
    - yc (np.ndarray): The y-coordinates of the data points.
    - zc (np.ndarray): The z-coordinates of the data points.
    - time_index (int): The index of the time array to be used for slicing.
    - time (np.ndarray): The time array.
    - title_prefix (str, optional): The prefix of the title. Default is "Raw VDF (LMN)".
    - sub_titles (tuple, optional): The individual titles of the subplots. 
        Default is (None, None, None).
    - coord_labels (tuple, optional): The labels of the coordinates. Default is 
        ("v_L", "v_M", "v_N").
    - mode (str, optional): Options are "integrate" and "slice". Default is "integrate".
    - slice_index (int, optional): The index of the slice to be used for 
        plotting if mode is "slice". Default is None.
    - cbar_label (str, optional): The label of the colorbar. Default is 
        "log_{10}(f*Delta v^3)[m^-3]".
    - *args: Additional arguments to be passed to imagecont_vdf.
    - **kwargs: Additional keyword arguments to be passed to imagecont_vdf.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - axes (list): The list of axes objects.
    """
    # Default coordinate labels:
    if coord_labels is None:
        coord_labels = (r"$v_L$", r"$v_M$", r"$v_N$")

    # Extract slices on the fly
    vdf_xy, vdf_yz, vdf_xz = slice3d_to_2d(vdfbindat, time_index, mode=mode, 
                                    slice_index=slice_index)

    # Compute global vmin/vmax
    vmin, vmax = global_vmin_vmax_vdf(vdf_xy, vdf_yz, vdf_xz)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

    im1, _ = imagecont_vdf(xc, yc, vdf_xy, title=sub_titles[0], 
                           xlabel=coord_labels[0], ylabel=coord_labels[1], 
                           ax=axes[0], vmin=vmin, vmax=vmax, *args, **kwargs)

    im2, _ = imagecont_vdf(yc, zc, vdf_yz, title=sub_titles[1], 
                           xlabel=coord_labels[2], ylabel=coord_labels[1], 
                           ax=axes[1], vmin=vmin, vmax=vmax)

    im3, _ = imagecont_vdf(xc, zc, vdf_xz, title=sub_titles[2], 
                           xlabel=coord_labels[2], ylabel=coord_labels[0], 
                           ax=axes[2], vmin=vmin, vmax=vmax)
    # Colorbar for all subplots
    factor = 0.046/(3 + 0.15)
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=factor, pad=0.04)
    cbar.set_label(cbar_label, fontsize=kwargs.get("cbar_fntsz", 10))
    # Title
    title_time = time_string(time[time_index])[:-3]
    fig.suptitle(f"{title_prefix} at {title_time}", fontsize = 16)
    
    plt.show()
    return fig, axes