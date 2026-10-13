"""
================================================================================
Runnable script to plot the paths of the four MMS probes in a common LMN frame.

The LMN frame comes from a single MVA on the four-probe averaged B (FGM brst).
Probes 2-4 are interpolated onto the MMS1 FGM time stamps before averaging.
Positions and velocities come from MEC (srvy only for this date; no brst MEC
files exist) and are interpolated onto the same time stamps.

Figure 1: position paths in the L-M, L-N and M-N planes (km, origin at Earth's
          center since the rotation does not translate).
Figure 2: velocity vectors in the same three planes (km/s).
Figures 3-4: 3D scatter of positions (with tetrahedron at the start) and velocities.

Author: Regis John
Created: 2026-10-09
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pyspedas import del_data, get_data, store_data, tinterpol, time_double
from pyspedas.projects import mms

from fipcore.utils.coord_utils import lmn_matrix_make

trange = ['2015-12-09/05:03:55', '2015-12-09/05:03:59']
probes = ['1', '2', '3', '4']
colors = {'1': 'black', '2': 'red', '3': 'green', '4': 'blue'}   # MMS convention

del_data('*')

#--- Load FGM (brst) and MEC (srvy, 30 s cadence) ---
mms.fgm(trange=trange, probe=probes, data_rate='brst', time_clip=True, no_update=True)
# MEC is not time-clipped so there are ephemeris points on both sides of trange
mec_trange = [time_double(trange[0]) - 300, time_double(trange[1]) + 300]
mms.mec(trange=mec_trange, probe=probes, data_rate='srvy')

#--- Four-probe averaged B on MMS1 time stamps ---
b_ref = 'mms1_fgm_b_gse_brst_l2_bvec'
tinterpol([f'mms{p}_fgm_b_gse_brst_l2_bvec' for p in probes[1:]], b_ref, suffix='_itp')

t_ref, b1 = get_data(b_ref)
b_all = [b1] + [get_data(f'mms{p}_fgm_b_gse_brst_l2_bvec_itp')[1] for p in probes[1:]]
b_avg = np.mean(np.stack(b_all), axis=0)
store_data('bvec_gse_avg', data={'x': t_ref, 'y': b_avg})

#--- Single MVA on averaged B -> mean LMN matrix ---
lmn_matrix_make('bvec_gse_avg', trange, newname='lmn_matrix_avg')
R = get_data('lmn_matrix_avg')[1][0]   # rows: L, M, N in GSE
print('L =', R[0], '\nM =', R[1], '\nN =', R[2])

#--- Interpolate MEC position/velocity onto t_ref and rotate to LMN ---
r_lmn, v_lmn = {}, {}
for p in probes:
    tinterpol([f'mms{p}_mec_r_gse', f'mms{p}_mec_v_gse'], b_ref, method='cubic', suffix='_itp')
    r_gse = get_data(f'mms{p}_mec_r_gse_itp')[1]   # km
    v_gse = get_data(f'mms{p}_mec_v_gse_itp')[1]   # km/s
    r_lmn[p] = np.einsum('ij,tj->ti', R, r_gse)
    v_lmn[p] = np.einsum('ij,tj->ti', R, v_gse)

#--- Scatter of each probe's LMN vectors in the L-M, L-N, M-N planes ---
planes = [(0, 1), (0, 2), (1, 2)]   # (x, y): L-M, L-N, M-N; L always on x when present
labels = ['L', 'M', 'N']

def plt_lmn_planes(vec_lmn, sym, unit, title):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, (i, j) in zip(axs, planes):
        for p in probes:
            ax.scatter(vec_lmn[p][:, i], vec_lmn[p][:, j], s=4, color=colors[p], label=f'MMS{p}')
            ax.plot(vec_lmn[p][0, i], vec_lmn[p][0, j], 'o', mfc='none', mec=colors[p], ms=9)   # start
            ax.plot(vec_lmn[p][-1, i], vec_lmn[p][-1, j], 's', mfc='none', mec=colors[p], ms=9)  # end
        ax.set_xlabel(f'{sym}$_{labels[i]}$ ({unit})')
        ax.set_ylabel(f'{sym}$_{labels[j]}$ ({unit})')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)
    axs[0].legend(title='\u25cb start, \u25a1 end', fontsize=9, markerscale=3)
    fig.suptitle(f'{title}  {trange[0]} \u2013 {trange[1][-8:]}')
    return fig

fig1 = plt_lmn_planes(r_lmn, 'r', 'km', 'MMS positions in mean LMN')
fig2 = plt_lmn_planes(v_lmn, 'v', 'km/s', 'MMS spacecraft velocity in mean LMN')

#--- 3D scatter of each probe's LMN vectors ---
def plt_lmn_3d(vec_lmn, sym, unit, title, tetra=False):
    fig = plt.figure(figsize=(8, 7), layout='constrained')
    ax = fig.add_subplot(projection='3d')
    for p in probes:
        v = vec_lmn[p]
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=4, color=colors[p], label=f'MMS{p}')
        ax.scatter(*v[0], marker='o', s=60, facecolors='none', edgecolors=colors[p])    # start
        ax.scatter(*v[-1], marker='s', s=60, facecolors='none', edgecolors=colors[p])   # end
    if tetra:
        # Tetrahedron edges at the first time step
        for a in range(4):
            for b in range(a + 1, 4):
                seg = np.array([vec_lmn[probes[a]][0], vec_lmn[probes[b]][0]])
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color='gray', lw=0.8, ls='--')
    # Equal scaling on all three axes
    allv = np.concatenate([vec_lmn[p] for p in probes])
    ax.set_box_aspect(np.ptp(allv, axis=0), zoom=0.85)
    ax.set_xlabel(f'{sym}$_L$ ({unit})', labelpad=12)
    ax.set_ylabel(f'{sym}$_M$ ({unit})', labelpad=12)
    ax.set_zlabel(f'{sym}$_N$ ({unit})', labelpad=12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.legend(title='○ start, □ end', fontsize=9, markerscale=3, loc='upper left')
    ax.set_title(f'{title}  {trange[0]} – {trange[1][-8:]}')
    return fig

fig3 = plt_lmn_3d(r_lmn, 'r', 'km', 'MMS positions in mean LMN', tetra=True)
fig4 = plt_lmn_3d(v_lmn, 'v', 'km/s', 'MMS spacecraft velocity in mean LMN')

plt.show()
