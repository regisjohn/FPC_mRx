"""
================================================================================
Plotting function that creates a pyspeadas style tplot of Nx1 subplots for 
tplot variables.

Functions:
- rtplot: Nx1 subplot of tplot variables.

Author: Regis John
Created: 2026-03-31
================================================================================
"""

import matplotlib.pyplot as plt
from pyspedas import get_data, time_datetime

def rtplot(varlist, tind_lims=None, figsize=(12, 2.5), yzero_line=True, 
          **styles):
    """
    Plotting function that creates a pyspeadas style tplot of Nx1 subplots for 
    tplot variables.

    Parameters:
    - varlist (list): List of tplot variable names to plot.
    - tind_lims (list or tuple): List or tuple of two time indices to limit the x-axis.
    - figsize (tuple): Figure size in inches (width, height).
    - yzero_line (bool): Whether to plot a horizontal line at y=0.
    - **styles (dict): Dictionary of dictionaries containing plotting styles for each variable.
        - Each dictionary should contain the following:
            - 'colors': List of colors to use for each component of the variable.
            - 'labels': List of labels to use for each component of the variable.
            - 'linestyles': List of linestyles to use for each component of the variable.
            - 'linewidth': List of line widths to use for each component of the variable.
            - 'ytitle': String to use as the y-axis title.

    Returns:
    - fig (matplotlib.figure.Figure): The resulting figure object.
    - axes (list): List of axes objects for the subplots.

    Examples:
    >>> styles = {'b_field_gse': {'colors': ['black', 'red', 'blue'], 
                    'labels': ['Bx', 'By', 'Bz'], 'linestyles': ['-', '--', '-.']}}
    >>> fig, axes = rtplot(['b_field_gse'], tind_lims=(0, 100), figsize=(12, 2.5), 
                               yzero_line=True, **styles)

    """
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black','red', 'blue', 'green', 'orange'])
    n_var = len(varlist)
    fig, axes = plt.subplots(n_var, 1, figsize=(figsize[0], figsize[1]*n_var), 
                    sharex=True, constrained_layout=True)

    if n_var==1:
        axes = [axes]

    for ax, var in zip(axes, varlist):
        times, data = get_data(var)
        t_dt = time_datetime(times)
        
        n_comp = data.shape[1]
        clrs = styles[var].get('colors', [None]*n_comp)
        lbls = styles[var].get('labels', [None]*n_comp)
        ln_styles = styles[var].get('linestyles', [None]*n_comp)
        ln_width = styles[var].get('linewidth', [1.5]*n_comp) 

        for i in range(n_comp):
            ax.plot(t_dt, data[: ,i], color=clrs[i], 
                linestyle=ln_styles[i], 
                linewidth=ln_width[i],
                label=lbls[i])
            ax.legend(loc='upper right')
        
        if yzero_line:
            ax.axhline(0, color='k', linestyle='dotted', linewidth=1.0)

        ax.set_ylabel(styles[var].get('ytitle', 'y'))

    fig.suptitle(styles.get('suptitle'))

    if tind_lims is not None:
        axes[-1].set_xlim(t_dt[tind_lims[0]], t_dt[tind_lims[1]])

    return fig, axes