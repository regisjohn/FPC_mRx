"""
================================================================================
Plots of the MMS1-4 trajectories (position and velocity) in the mean LMN frame.

The input is the .h5 file written by fipcore.analysis.fpc_mrx_main.fpc_mrx_traj.

Functions:
- load_traj: Read the trajectory .h5 file for a given trange.
- plt_traj_lmn_2d: Scatter of each probe in the L-M, L-N and M-N planes (matplotlib).
- plt_traj_lmn_3d: Interactive 3D scatter of each probe in L, M, N (pyvista).

Author: Regis John
Created: 2026-10-30
================================================================================
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fipcore.utils.io_utils import mms_name_make

# Style for MMS1-4 (same as plt_vdf_fpc)
COLORS = {'1': 'red', '2': 'green', '3': 'blue', '4': 'black'}
LABELS = ['L', 'M', 'N']
QTY_INFO = {'r': ('r', 'km', 'positions'), 'v': ('v', 'km/s', 'velocities')}
FRAME_INFO = {'recx': ('recx', 'reconnection frame'), 'sc': ('lmn', 'spacecraft frame')}


def load_traj(trange, data_rate='brst'):
    """
    Read the trajectory .h5 file written by fpc_mrx_traj.

    Parameters:
    - trange (list of str): [start time, end time], same as passed to fpc_mrx_traj.
    - data_rate (str): Data rate used in fpc_mrx_traj. Default is 'brst'.

    Returns:
    - traj (dict): {'meta': {...}, '1': {...}, '2': {...}, ...}, one entry per probe
        with time, recx_idx, bulk_vi_rev, r_gse, v_gse, r_lmn, v_lmn, r_recx, v_recx.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    hfile = os.path.join(project_root, "data",
                         mms_name_make(f'mms_traj_{data_rate}', trange[0], trange[1]))
    traj = {}
    with h5py.File(hfile, 'r') as f:
        probes = list(f['meta']['probes'].asstr()[...])
        traj['meta'] = {'trange': list(f['meta']['trange'].asstr()[...]),
                        'probes': probes,
                        'data_rate': f['meta']['data_rate'].asstr()[()],
                        'rot_mat': f['meta']['rot_mat'][...]}
        for p in probes:
            g = f[f'mms{p}']
            traj[p] = {k: g[k][...] for k in g.keys()}
            traj[p]['recx_idx'] = int(traj[p]['recx_idx'])
    return traj


def _mark_list(mark_idx, probes):
    # One time index per probe (int -> same index for all probes, None -> no marks)
    if mark_idx is None:
        return [None] * len(probes)
    if isinstance(mark_idx, (int, np.integer)):
        return [int(mark_idx)] * len(probes)
    return list(mark_idx)


def _star_center(traj, probes, key):
    # Mean of the four B_L reversal points
    return np.mean([traj[p][key][traj[p]['recx_idx']] for p in probes], axis=0)


def _pos_offset(traj, probes, key):
    # Axis offset for positions: mean of the B_L reversal points, rounded to 10 km
    return np.round(_star_center(traj, probes, key), -1)


def _arrow_pts(vec, bounds=None):
    # Start and tip of the 3D direction arrow: the last ~1/8 of the points inside
    # bounds [xmin, xmax, ymin, ymax, zmin, zmax] (all points if None), tip on the last
    if bounds is not None:
        lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])
        inside = vec[np.all((vec >= lo) & (vec <= hi), axis=1)]
        vec = inside if len(inside) >= 3 else vec
    n = len(vec)
    return vec[n - 1 - max(n // 8, 2)], vec[n - 1]


def _axis_label(sym, lab, o, unit, math=True):
    # Axis label with the offset written in it, e.g. r_L + 25750 (km): tick = value - o
    name = f'{sym}$_{lab}$' if math else f'{sym}_{lab}'
    if o == 0:
        return f'{name} ({unit})'
    sign = '+' if o < 0 else '-'
    return f'{name} {sign} {abs(o):.0f} ({unit})'


def _plt_file(prefix, trange, sav_ext):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    plot_dir = os.path.join(project_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return os.path.join(plot_dir, mms_name_make(prefix, trange[0], trange[1], ext=sav_ext))


def plt_traj_lmn_2d(trange, quantity='r', frame='recx', data_rate='brst', mark_idx=None,
                    mark_label='marked time', lims=None, figsize=None, point_size=None,
                    marker_size=None, save_fig=False, sav_ext='png', use_style=False):
    """
    Scatter of each probe's position or velocity in the L-M, L-N and M-N planes.

    Markers: star = B_L reversal (recx_idx), gold diamond = mark_idx, open circle =
    start, square = end (positions only). Position axes show values relative to an offset (the
    mean B_L reversal point, rounded to 10 km) that is written in the axis labels.

    Parameters:
    - trange (list of str): [start time, end time], same as passed to fpc_mrx_traj.
    - quantity (str): 'r' for position or 'v' for velocity. Default is 'r'.
    - frame (str): 'recx' for each probe's reconnection frame or 'sc' for the
        spacecraft frame. Default is 'recx'.
    - data_rate (str): Data rate used in fpc_mrx_traj. Default is 'brst'.
    - mark_idx (int, list of int or None): Time index to mark, one per probe
        (e.g. FPC peak indices [62, 58, 60, 62]) or one for all. Default is None.
    - mark_label (str): Legend label of the mark_idx marker, e.g. 'FPC peak'.
        Default is 'marked time'.
    - lims (dict or None): Axis limits per coordinate, e.g. {'L': (-60, 40),
        'N': (-50, 20)}, in the units of the tick labels (relative to the offset
        for positions). A coordinate gets the same limits in every panel it
        appears in; coordinates left out are autoscaled. All of trange is still
        plotted. A panel keeps 1:1 aspect unless only one of its two axes has
        limits. Default is None (autoscale everything).
    - figsize (tuple or None): Figure size in inches. Default is None: (16, 5), or
        with use_style twice the style's column width (one full page width).
    - point_size (float or None): Scatter marker size (points^2). Default is None: 4,
        or with use_style (0.15 * font.size)^2.
    - marker_size (float or None): Size of the start, end and mark_idx markers; the
        star is 1.75 times larger. Default is None: 8, or with use_style 0.5 * font.size.
    - save_fig (bool): if True, save the figure to <project root>/plots; otherwise show it.
    - sav_ext (str): file extension of the saved figure. Default is 'png'.
    - use_style (bool): If True, the figure size, marker sizes, font sizes and dpi
        that are not given explicitly come from the active matplotlib style
        (e.g. plt.style.use(['science'])); the title is dropped and the panels
        get (a)/(b)/(c) labels. Default is False.
    """
    # Hardcoded defaults unless use_style; unset values then come from rcParams
    legend_fntsz, dpi_val = None, None
    if use_style:
        # Marker sizes scale with the style's font size (lines.markersize is
        # usually left at matplotlib's default, too big for column-width panels)
        fntsz = plt.rcParams['font.size']
        fig_w, fig_h = plt.rcParams['figure.figsize']
        figsize = (2 * fig_w, fig_h) if figsize is None else figsize
        point_size = (0.15 * fntsz) ** 2 if point_size is None else point_size
        marker_size = 0.5 * fntsz if marker_size is None else marker_size
    else:
        legend_fntsz, dpi_val = 9, 300
        figsize = (16, 5) if figsize is None else figsize
        point_size = 4 if point_size is None else point_size
        marker_size = 8 if marker_size is None else marker_size

    traj = load_traj(trange, data_rate=data_rate)
    probes = traj['meta']['probes']
    sym, unit, qty_name = QTY_INFO[quantity]
    key = f'{quantity}_{FRAME_INFO[frame][0]}'
    marks = _mark_list(mark_idx, probes)
    off = _pos_offset(traj, probes, key) if quantity == 'r' else np.zeros(3)
    vecs = {p: np.asarray(traj[p][key], dtype=np.float64) - off for p in probes}
    lims = {} if lims is None else lims
    ms, ms_star = marker_size, 1.75 * marker_size

    planes = [(0, 1), (0, 2), (1, 2)] # (x, y): L-M, L-N, M-N; L always on x when present
    fig, axs = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    for ax, (i, j) in zip(axs, planes):
        for p, m in zip(probes, marks):
            vec = vecs[p]
            c = COLORS[p]
            k = traj[p]['recx_idx']
            ax.scatter(vec[:, i], vec[:, j], s=point_size, color=c, lw=0)
            if quantity == 'r':
                ax.plot(vec[0, i], vec[0, j], 'o', mfc='white', mec=c, ms=ms)    # start
                ax.plot(vec[-1, i], vec[-1, j], 's', color=c, mec='k', ms=ms)    # end
            ax.plot(vec[k, i], vec[k, j], '*', color=c, mec='k', ms=ms_star) # B_L reversal
            if m is not None:
                ax.plot(vec[m, i], vec[m, j], 'D', color='gold', mec=c, ms=ms)
        ax.set_xlabel(_axis_label(sym, LABELS[i], off[i], unit))
        ax.set_ylabel(_axis_label(sym, LABELS[j], off[j], unit))
        xlim, ylim = lims.get(LABELS[i]), lims.get(LABELS[j])
        if xlim is None and ylim is None:
            ax.set_aspect('equal', adjustable='datalim')
        elif xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
        else:
            # Only one axis fixed: the other is fitted to the points inside it
            pts = np.concatenate([vecs[p][:, [i, j]] for p in probes])
            fix, free = (0, 1) if xlim is not None else (1, 0)
            lo, hi = xlim if xlim is not None else ylim
            sel = pts[(pts[:, fix] >= lo) & (pts[:, fix] <= hi), free]
            sel = pts[:, free] if sel.size == 0 else sel
            pad = 0.05 * max(np.ptp(sel), 1e-3)
            free_lim = (sel.min() - pad, sel.max() + pad)
            ax.set_xlim(xlim if xlim is not None else free_lim)
            ax.set_ylim(ylim if ylim is not None else free_lim)
        ax.grid(alpha=0.3)

    # Legend: probe colours, then the marker meanings (in grey); above the panels
    # with use_style, otherwise in the first panel below the title.
    # Plain mathtext only, so it also works with the LaTeX of the 'science' style.
    handles = [Line2D([], [], ls='', marker='o', color=COLORS[p], label=f'MMS{p}')
               for p in probes]
    handles.append(Line2D([], [], ls='', marker='*', color='gray', mec='k', ms=ms_star,
                          label='B$_L$ reversal'))
    if any(m is not None for m in marks):
        handles.append(Line2D([], [], ls='', marker='D', color='gold', mec='gray', ms=ms,
                              label=mark_label))
    if quantity == 'r':
        handles += [Line2D([], [], ls='', marker='o', mfc='white', mec='gray', ms=ms,
                           label='start'),
                    Line2D([], [], ls='', marker='s', color='gray', mec='k', ms=ms,
                           label='end')]
    if use_style:
        fig.legend(handles=handles, loc='outside upper center', ncols=len(probes))
        for ax, pl in zip(axs, ['(a)', '(b)', '(c)']):
            ax.set_title(rf'$\mathbf{{{pl}}}$', loc='left')
    else:
        axs[0].legend(handles=handles, loc='lower left', fontsize=legend_fntsz)
        fig.suptitle(f'MMS {qty_name} in mean LMN, {FRAME_INFO[frame][1]}  '
                     f'{trange[0]} – {trange[1][-8:]}')

    if save_fig:
        plt_pref = ('PRL_' if use_style else '') + f'traj_{quantity}_{frame}_2d_{data_rate}'
        pltfile = _plt_file(plt_pref, trange, sav_ext)
        fig.savefig(pltfile, bbox_inches='tight', dpi=dpi_val)
        print(f'File saved to {pltfile}!')
        plt.close(fig)
    else:
        plt.show()


def plt_traj_lmn_3d(trange, quantity='r', frame='recx', data_rate='brst', mark_idx=None,
                    lims=None, azimuth=0., elevation=0., roll=0., zoom=1., camera_position=None,
                    point_size=6, window_size=(900, 750), jupyter_backend=None,
                    save_fig=False, sav_ext='png', save_html=False, show=True):
    """
    Interactive 3D scatter of each probe's position or velocity in L, M, N.

    The view starts from the isometric view and is then rotated by azimuth,
    elevation and roll (degrees) and zoomed. A full camera_position overrides these.
    The final camera_position is printed so a view found with the mouse can be
    reused. Big spheres mark the B_L reversal (recx_idx), gold spheres mark mark_idx, 
    arrows show the direction of travel (positions only). 
    Positions are drawn relative to an offset (the mean B_L reversal point, rounded 
    to 10 km), which is written in the axis titles.

    Parameters:
    - trange (list of str): [start time, end time], same as passed to fpc_mrx_traj.
    - quantity (str): 'r' for position or 'v' for velocity. Default is 'r'.
    - frame (str): 'recx' for each probe's reconnection frame or 'sc' for the
        spacecraft frame. Default is 'recx'.
    - data_rate (str): Data rate used in fpc_mrx_traj. Default is 'brst'.
    - mark_idx (int, list of int or None): Time index to mark, one per probe or one
        for all. Default is None.
    - lims (dict or None): Limits of the grid box per coordinate, e.g.
        {'L': (-60, 40), 'N': (-50, 20)}, in the units of the axis labels (relative
        to the offset for positions); coordinates left out span all the data. The
        camera is fitted to the box and the arrows sit on the last points inside
        it. All of trange is still plotted. Default is None (whole trange).
    - azimuth, elevation, roll (float): Camera rotations in degrees applied to the
        isometric view. Default is 0.
    - zoom (float): Camera zoom factor. Default is 1.
    - camera_position (list or None): [position, focal_point, view_up]. Overrides
        azimuth, elevation, roll and zoom. Default is None.
    - point_size (float): Point size in pixels. Default is 6.
    - window_size (tuple): Window size in pixels. Default is (900, 750).
    - jupyter_backend (str or None): pyvista notebook backend ('trame', 'static', ...).
        None uses pyvista's global default. Ignored outside notebooks.
    - save_fig (bool): if True, save a screenshot to <project root>/plots.
    - sav_ext (str): file extension of the screenshot. Default is 'png'.
    - save_html (bool): if True, save an interactive .html to <project root>/plots.
    - show (bool): if True, show the plot. Default is True.

    Returns:
    - pl (pyvista.Plotter): the plotter, e.g. to read pl.camera_position.
    """
    import pyvista as pv # imported here so `import fipcore` doesn't load VTK

    traj = load_traj(trange, data_rate=data_rate)
    probes = traj['meta']['probes']
    sym, unit, qty_name = QTY_INFO[quantity]
    key = f'{quantity}_{FRAME_INFO[frame][0]}'
    marks = _mark_list(mark_idx, probes)
    off = _pos_offset(traj, probes, key) if quantity == 'r' else np.zeros(3)
    vecs = {p: np.asarray(traj[p][key], dtype=np.float64) - off for p in probes}

    # Grid box: lims where given, otherwise the data range
    bounds = None
    if lims is not None:
        allv = np.concatenate([vecs[p] for p in probes])
        bounds = [b for n, lab in enumerate(LABELS)
                  for b in (lims[lab] if lims.get(lab) is not None
                            else (allv[:, n].min(), allv[:, n].max()))]

    pl = pv.Plotter(window_size=window_size, off_screen=not show)
    pl.set_background('white')
    for p, m in zip(probes, marks):
        vec = vecs[p]
        c = COLORS[p]
        k = traj[p]['recx_idx']
        pl.add_points(vec, color=c, point_size=point_size,
                      render_points_as_spheres=True, label=f'MMS{p}')
        pl.add_points(vec[k:k + 1], color=c, point_size=4 * point_size,
                      render_points_as_spheres=True)                    # B_L reversal
        if m is not None:
            pl.add_points(vec[m:m + 1], color='gold', point_size=3 * point_size,
                          render_points_as_spheres=True)
        if quantity == 'r':                                                 # direction
            a, b = _arrow_pts(vec, bounds)
            pl.add_mesh(pv.Arrow(start=a, direction=b - a, scale=np.linalg.norm(b - a)),
                        color=c)

    titles = [_axis_label(sym, lab, o, unit, math=False) for lab, o in zip(LABELS, off)]
    pl.show_grid(xtitle=titles[0], ytitle=titles[1], ztitle=titles[2], bounds=bounds, 
                 color='black', font_size=10,
                 n_xlabels=4, n_ylabels=4, n_zlabels=4,
                 fmt='%.0f' if quantity == 'r' else '%.1f')
    pl.add_axes(xlabel='L', ylabel='M', zlabel='N', color='black')
    pl.add_legend(bcolor='white', border=True, size=(0.12, 0.15), loc='upper right')
    pl.add_text(f'MMS {qty_name} in mean LMN, {FRAME_INFO[frame][1]}\n'
                f'{trange[0]} - {trange[1][-8:]}\nbig sphere: B_L reversal, gold: mark_idx, '
                f'arrow: direction',
                position='lower_right', font_size=8, color='black')

    # --- Camera ---
    if camera_position is not None:
        pl.camera_position = camera_position
    else:
        pl.view_isometric()
        if bounds is not None:
            # Aim at the grid box and move in until it fills the view
            cam = pl.camera
            box = np.array(bounds).reshape(3, 2)
            focal = box.mean(axis=1)
            view_dir = np.array(cam.position) - np.array(cam.focal_point)
            view_dir /= np.linalg.norm(view_dir)
            half_diag = 0.5 * np.linalg.norm(np.diff(box, axis=1))
            dist = half_diag / np.sin(np.radians(cam.view_angle / 2))
            cam.focal_point = focal
            cam.position = focal + dist * view_dir
        pl.camera.Azimuth(azimuth)
        pl.camera.Elevation(elevation)
        pl.camera.Roll(roll)
        pl.camera.OrthogonalizeViewUp()
        pl.camera.zoom(zoom)
    pl.camera_set = True # keep this view when the plot is first rendered
    print(f'camera_position = {[tuple(np.round(c, 3).tolist()) for c in pl.camera_position]}')

    if save_html:
        htmlfile = _plt_file(f'traj_{quantity}_{frame}_3d_{data_rate}', trange, 'html')
        pl.export_html(htmlfile)
        print(f'File saved to {htmlfile}!')
    if save_fig:
        pltfile = _plt_file(f'traj_{quantity}_{frame}_3d_{data_rate}', trange, sav_ext)
        pl.screenshot(pltfile)
        print(f'File saved to {pltfile}!')
    if show:
        pl.show(jupyter_backend=jupyter_backend)

    return pl
