"""
================================================================================
1D plots of integrated delta f and C_E versus velocity for all MMS probes.

The input probes_data comes from fipcore.utils.io_utils.load_plt_mms_data.

Functions:
- plt_vdf_fpc: 4x3 plot of delta f and C_E components at one time index.
- plt_vdf_fpc_peak: 4x3 plot with each probe at its own (peak) time index.
- plt_vdf_cepar_vpar_peak: 2x1 plot of delta f and C_E_parallel vs v_parallel.

Author: Regis John
Created: 2026-10-14
================================================================================
"""

import matplotlib.pyplot as plt
from pyspedas import time_string


def plt_vdf_fpc(probes_data, time_index, prb=0, figsize=(18, 14), save_path=None):
    """
    4×3 subplot layout:

        Row 0: delta f
        Row 1: C_E perpendicular 1
        Row 2: C_E perpendicular 2
        Row 3: C_E parallel

    Parameters:
    - probes_data (list of dict): output of load_plt_mms_data, one dict per probe.
    - time_index (int): time index of the data, shown in the top right corner.
    - prb (int): probe whose timestamp is used in the title. Default is 0 (MMS1).
    - figsize (tuple): figure size in inches. Default is (18, 14).
    - save_path (str or None): if given, save the figure there; otherwise show it.
    """
    # Velocity coordinate
    vpar = probes_data[0]["vpar"]

    fig, axs = plt.subplots(4, 3, figsize=figsize, constrained_layout=True)

    # Style for MMS1–4
    colors = ['red', 'green', 'blue', 'black']
    styles = ['-', '--', ':', '-.']
    zorders = [5, 4, 3, 2]

    # Velocity keys for the three columns
    v_keys = ['vperp1', 'vperp2', 'vpar']

    # Plot each probe
    for i, pdata in enumerate(probes_data):

        lbl = f"MMS{i+1}"
        color = colors[i]
        ls = styles[i]
        zo = zorders[i]

        for col, v_key in enumerate(v_keys):
            # Row 0: delta f
            axs[0, col].plot(vpar, pdata["df"][v_key], color=color,
                    linestyle=ls, linewidth=2, zorder=zo, label=lbl)

            # Row 1: C_E perpendicular 1
            axs[1, col].plot(vpar, pdata["c_e"]["c_perp1"][v_key], color=color,
                    linestyle=ls, linewidth=2, zorder=zo, label=lbl)

            # Row 2: C_E perpendicular 2
            axs[2, col].plot(vpar, pdata["c_e"]["c_perp2"][v_key], color=color,
                    linestyle=ls, linewidth=2, zorder=zo, label=lbl)

            # Row 3: C_E parallel
            axs[3, col].plot(vpar, pdata["c_e"]["c_par"][v_key], color=color,
                    linestyle=ls, linewidth=2, zorder=zo, label=lbl)

    # Axis formatting
    for ax in axs.flat:
        ax.set_xlim([-3, 3])
        ax.tick_params(labelsize=16)
        ax.axhline(0, color='black', linewidth=1, linestyle='dashed')
        ax.axvline(0, color='black', linewidth=1, linestyle='dashed')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(14)

    # X-axis labels
    x_labels = [r'$v_{\perp 1}/v_{te}$', r'$v_{\perp 2}/v_{te}$',
                r'$v_{\parallel}/v_{te}$']

    for i in range(3):
        axs[3, i].set_xlabel(x_labels[i], fontsize=20)

    # Y-axis labels
    axs[0, 0].set_ylabel(r'$\delta f\ (\mathrm{m^{-3}})$', fontsize=20)

    axs[1, 0].set_ylabel(r'$C_{\mathrm{E}\perp 1}\ (\mathrm{W\cdot m^{-3}})$',
        fontsize=20)

    axs[2, 0].set_ylabel(r'$C_{\mathrm{E}\perp 2}\ (\mathrm{W\cdot m^{-3}})$',
        fontsize=20)

    axs[3, 0].set_ylabel(r'$C_{\mathrm{E}\parallel}\ (\mathrm{W\cdot m^{-3}})$',
        fontsize=20)

    # Legend
    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center',
        bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=16, frameon=True)

    # Time annotation
    fig.text(0.98, 0.98, f"t: {time_index}", ha="right", va="top", fontsize=18)

    fig.suptitle(f"MMS: {time_string(probes_data[prb]['time'])[:-3]}",
        fontsize=18, fontweight='bold')

    # Save / show
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'File saved to {save_path}!')
        plt.close(fig)
    else:
        plt.show()


def plt_vdf_fpc_peak(probes_data, time_indices, figsize=(18, 14), save_path=None):
    """
    4×3 subplot layout:
        Row 0: delta f
        Row 1: C_E perpendicular 1
        Row 2: C_E perpendicular 2
        Row 3: C_E parallel
    Each probe is plotted at its respective peak time.

    Parameters:
    - probes_data (list of dict): output of load_plt_mms_data, one dict per probe.
    - time_indices (list of int): time index of each probe, shown in the legend.
    - figsize (tuple): figure size in inches. Default is (18, 14).
    - save_path (str or None): if given, save the figure there; otherwise show it.
    """
    # Velocity coordinate
    vpar = probes_data[0]["vpar"]

    # Figure
    fig, axs = plt.subplots(4, 3, figsize=figsize, constrained_layout=True)

    # Style for MMS1–4
    colors = ['red', 'green', 'blue', 'black']
    styles = ['-', '--', ':', '-.']
    zorders = [5, 4, 3, 2]

    # Velocity keys for the three columns
    v_keys = ['vperp1', 'vperp2', 'vpar']

    # Plot each probe
    for i, pdata in enumerate(probes_data):

        lbl = f"MMS{i+1} (t={time_indices[i]})"
        color = colors[i]
        ls = styles[i]
        zo = zorders[i]

        for col, v_key in enumerate(v_keys):

            # Row 0: delta f
            axs[0, col].plot(vpar, pdata["df"][v_key], color=color,
                linestyle=ls, linewidth=2, zorder=zo, label=lbl)

            # Row 1: C_E perpendicular 1
            axs[1, col].plot(vpar, pdata["c_e"]["c_perp1"][v_key], color=color,
                linestyle=ls, linewidth=2, zorder=zo, label=lbl)

            # Row 2: C_E perpendicular 2
            axs[2, col].plot(vpar, pdata["c_e"]["c_perp2"][v_key], color=color,
                linestyle=ls, linewidth=2, zorder=zo, label=lbl)

            # Row 3: C_E parallel
            axs[3, col].plot(vpar, pdata["c_e"]["c_par"][v_key], color=color,
                linestyle=ls, linewidth=2, zorder=zo, label=lbl)

    # Axis formatting
    for ax in axs.flat:
        ax.set_xlim([-3, 3])
        ax.tick_params(labelsize=16)
        ax.axhline(0, color='black', linewidth=1, linestyle='dashed')
        ax.axvline(0, color='black', linewidth=1, linestyle='dashed')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(14)

    # X-axis labels
    x_labels = [r'$v_{\perp 1}/v_{te}$', r'$v_{\perp 2}/v_{te}$',
        r'$v_{\parallel}/v_{te}$']

    for i in range(3):
        axs[3, i].set_xlabel(x_labels[i], fontsize=20)

    # Y-axis labels
    axs[0, 0].set_ylabel(r'$\delta f\ (\mathrm{m^{-3}})$', fontsize=20)
    axs[1, 0].set_ylabel(r'$C_{\mathrm{E}\perp 1}\ (\mathrm{W\cdot m^{-3}})$',
        fontsize=20)
    axs[2, 0].set_ylabel(r'$C_{\mathrm{E}\perp 2}\ (\mathrm{W\cdot m^{-3}})$',
        fontsize=20)

    axs[3, 0].set_ylabel(r'$C_{\mathrm{E}\parallel}\ (\mathrm{W\cdot m^{-3}})$',
        fontsize=20)

    # Legend
    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01),
        ncol=4, fontsize=16, frameon=True)

    # Figure title
    fig.suptitle(r"Integrated $\delta f$ and Velocity-Space Signatures at " \
    r"Peak $J\cdot E_{\parallel}$ Energization",fontsize=18,fontweight='bold')

    # Save / show
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'File saved to {save_path}!')
        plt.close(fig)
    else:
        plt.show()


def plt_vdf_cepar_vpar_peak(probes_data, figsize=None, save_path=None, use_style=False):
    """
    2×1 subplot layout:

        Row 0: delta f vs v_parallel
        Row 1: C_E_parallel vs v_parallel

    Parameters:
    - probes_data (list of dict): output of load_plt_mms_data, one dict per probe.
    - figsize (tuple or None): figure size in inches. Default is None: (6, 8),
        or with use_style one column of the active style with height for two panels.
    - save_path (str or None): if given, save the figure there; otherwise show it.
    - use_style (bool): If True, font sizes, line widths and the figure size that
        are not given explicitly come from the active matplotlib style
        (e.g. plt.style.use(['science', 'ieee'])), the panels share the x-axis and
        get (a)/(b) labels. Default is False.
    """
    # Velocity coordinate
    vpar = probes_data[0]["vpar"]

    # Hardcoded defaults unless use_style; unset values then come from rcParams
    lw, ref_lw, legend_fntsz = None, None, None
    if not use_style:
        lw, ref_lw, legend_fntsz = 2, 1, 10
    if figsize is None:
        if use_style:
            # One column wide from the active style; height for two stacked panels
            fig_w, fig_h = plt.rcParams['figure.figsize']
            figsize = (fig_w, 2 * fig_h)
        else:
            figsize = (6, 8)

    # Figure
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=use_style,
                            constrained_layout=True)

    # Style for MMS1–4
    colors = ['red', 'green', 'blue', 'black']
    styles = ['-', '--', ':', '-.']
    zorders = [5, 4, 3, 2]

    # Plot each probe
    for i, pdata in enumerate(probes_data):

        lbl = f"MMS{i+1}"
        color = colors[i]
        ls = styles[i]
        zo = zorders[i]

        # Top: delta f vs v_parallel
        axs[0].plot(vpar, pdata["df"]["vpar"], color=color, linestyle=ls,
            linewidth=lw, zorder=zo, label=lbl)

        # Bottom: C_E_parallel vs v_parallel
        axs[1].plot(vpar, pdata["c_e"]["c_par"]["vpar"], color=color,
            linestyle=ls, linewidth=lw, zorder=zo, label=lbl)

    # Axis formatting
    for ax in axs:
        ax.set_xlim([-3, 3])
        ax.axhline(0, color='black', linewidth=ref_lw, linestyle='dashed')
        ax.axvline(0, color='black', linewidth=ref_lw, linestyle='dashed')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        if not use_style:
            ax.tick_params(labelsize=16)
            ax.yaxis.get_offset_text().set_fontsize(14)

    # X-axis labels
    axs[1].set_xlabel(r'$v_{\parallel}/v_{te}$')

    # Y-axis labels
    axs[0].set_ylabel(r'$\delta f\ (\mathrm{m^{-3}})$')
    axs[1].set_ylabel(r'$C_{\mathrm{E}\parallel}\ (\mathrm{W\cdot m^{-3}})$')

    if not use_style:
        for label in (axs[1].xaxis.label, axs[0].yaxis.label, axs[1].yaxis.label):
            label.set_fontsize(20)

    # Panel labels (top right, clear of the y-axis multiplier and the legend)
    if use_style:
        for ax, pl in zip(axs, ['(a)', '(b)']):
            ax.text(0.97, 0.95, rf'$\mathbf{{{pl}}}$', transform=ax.transAxes,
                va='top', ha='right')

    # Legend
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc='upper left', fontsize=legend_fntsz)
    axs[1].legend(handles, labels, loc='upper left', fontsize=legend_fntsz)

    # Figure title
    # fig.suptitle(r"Integrated $\delta f$ and $C_{\mathrm{E}\parallel}$ "
    #     r"at Peak $J\cdot E_{\parallel}$ Energization", fontsize=18, fontweight='bold')

    # Save / show
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'File saved to {save_path}!')
        plt.close(fig)
    else:
        plt.show()
