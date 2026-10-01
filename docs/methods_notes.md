# fipcore methods notes

Notes for writing the methodology section. They describe what the code in `fipcore/` does as of commit `d895f12`. Nothing here is taken from the literature or from intent. Where the code is ambiguous, or its behaviour may not match what a paper would normally claim, the item is marked **⚠ Flag** and collected in [§11](#11-flags-and-open-questions).

Function references use the form `module.function`. All modules are under [`fipcore/`](../fipcore/).

---

## 0. Parameters used for the published figures

[`scripts/Publication_plots.ipynb`](../scripts/Publication_plots.ipynb) calls `fpc_mrx_main` with only these arguments set; everything else is the default:

| Parameter | Value | Where set |
|---|---|---|
| Event interval `trange` | `2015-12-09/05:03:55` – `05:03:59` (4 s) | notebook |
| Probes | MMS1–4, each processed independently | notebook |
| Species | electrons (`'e'`) | notebook |
| Data rate / level | `brst` / `l2` | default |
| `bin_width_frac` | 0.25 (also run at 0.10, 0.20, 0.30, 0.40, 0.50 for the bin-width checks) | notebook |
| `vth_lim` | 3.5 | default |
| `mean_phi` | `False` (per-time azimuth angles) | default |
| `subtract_f0` | run both `False` (full *f*) and `True` (δ*f*) | notebook |
| `v_flow` | `None` | default |
| `bg_ratio` | 0.5 (saved as metadata only, not used in any calculation) | notebook, `fpc_mrx_plasma_params` |

In the saved MMS1 file this gives 133 time samples at 30 ms cadence, a 28×28×28 velocity grid, and $\langle v_{th,e}\rangle$ = 4441.8 km/s.

---

## 1. Data loaded

From [`analysis/fpc_mrx_main.py`](../fipcore/analysis/fpc_mrx_main.py) → `fpc_mrx_main`, via pyspedas (`mms.fpi`, `mms.edp`, `mms.fgm`, with `time_clip=True`):

| Instrument | Quantity | Variable |
|---|---|---|
| FPI-DES | 3D electron distribution $f$, its error $\sigma_f$, per-distribution error flags | `des_dist_brst`, `des_disterr_brst`, `des_errorflags_brst_dist` |
| FPI-DES | Energy table (per sweep), azimuth angles $\phi$ (per sweep) | `des_energy_brst`, `des_phi_brst` |
| FPI-DES | Moments: $n_e$, $T_{e\parallel}$, $T_{e\perp}$, $\mathbf{V}_e$ (GSE) | `des_numberdensity`, `des_temppara`, `des_tempperp`, `des_bulkv_gse` |
| FPI-DIS | Ion bulk velocity $\mathbf{V}_i$ (GSE) | `dis_bulkv_gse_brst` |
| EDP | Electric field $\mathbf{E}$ (GSE), spacecraft potential $V_{sc}$ | `edp_dce_gse_brst_l2`, `edp_scpot_brst_l2` |
| FGM | Magnetic field $\mathbf{B}$ (GSE) | `fgm_b_gse_brst_l2` |

`edp_dce_par_epar_brst_l2` is also loaded but never used.

## 2. Common time base (30 ms DES cadence)

Every quantity is put on the DES burst time base, which is the timestamps of $\mathbf{V}_e$.

- **Faster fields ($\mathbf{E}$, $\mathbf{B}$, $V_{sc}$) are boxcar-averaged** ([`utils/helper_utils.downsample_cad`](../fipcore/utils/helper_utils.py), which wraps pyspedas `avg_data`). The window width in *samples* is $N_{in}/N_{DES}$, the ratio of the number of points in the input series to the number of DES points over `trange`.
- **Slower data ($\mathbf{V}_i$, 150 ms DIS cadence) is linearly interpolated** onto the DES timestamps (`helper_utils.upsample_cad`, which wraps pyspedas `tinterpol`).

⚠ Flag (F1): `avg_data` puts its output on its own uniform grid, $t_k=t_{start}+(k+\tfrac12)\Delta t$, not on the DES timestamps. Downstream code then pairs the averaged fields with DES samples by array index, not by time. The alignment is presumably within one sample but is not enforced.

## 3. Velocity-space coordinates of each measurement

### 3.1 Energy table and interleaving ([`analysis/vel_proc.nrgy_to_vbin`](../fipcore/analysis/vel_proc.py))
- The energy table can alternate between two tables on consecutive sweeps (interleaving). The code tests for this with `helper_utils.is_interleaved` and saves the result as `meta/interleave_info`. The event file says "interleaved".
- Either way, the energy table of each sweep is used for that sweep: the velocity grid is built per time sample. Saved `vv` has shape `(3, 16384, 133)`.

### 3.2 Spacecraft-potential correction (`vel_proc.nrgy_to_vbin`)
$$
\mathcal{E}_{corr} = \mathcal{E}_{table} + q\,V_{sc}, \qquad q=-1\ \text{(e)},\ +1\ \text{(ion)},\ \text{energies in eV, } V_{sc} \text{ in V}
$$
For electrons this is $\mathcal{E}_{table}-eV_{sc}$, applied with the 30 ms-averaged $V_{sc}$ at each time.

The speed of each energy channel is the non-relativistic
$$
v = c\sqrt{2\,\mathcal{E}_{corr}/(mc^2)} \quad \text{(km/s)}.
$$
⚠ Flag (F2): if $\mathcal{E}_{corr}<0$ (a channel below $eV_{sc}$), the square root gives NaN and that channel's measurements are silently dropped later. This didn't happen for the event (0 NaN velocities in the MMS1 file), but the code doesn't guard against it.

### 3.3 Velocity vectors (`vel_proc.vbin_to_vv`)
Each of the 32 (φ) × 16 (θ) × 32 (energy) = 16384 bins becomes a velocity vector in the instrument frame (DBCS). The code treats that frame as GSE with no further rotation, because FPI L2 moments and pyspedas conventions do the same:
$$
\mathbf{v} = -\,v\,(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta).
$$
- The minus sign converts the detector look direction into the particle's direction of travel.
- **Polar angles are hard-coded**: $\theta_k = 5.625^\circ + 11.25^\circ k$, for $k=0..15$. They are not read from the data file.
- **Azimuth angles** $\phi$ are the per-sweep values from `des_phi_brst` (because `mean_phi=False`). With `mean_phi=True`, their time average would be used instead.

⚠ Flag (F3): the code assumes DBCS ≈ GSE for the particle velocities; there is no explicit DBCS→GSE rotation. Confirm this approximation is acceptable and state it in the paper.

### 3.4 Phase-space volume of each instrument bin (`vel_proc.compute_vbin_vol`)
$$
\Delta^3 v = \tfrac13\left(v_{hi}^3 - v_{lo}^3\right)\left(\cos\theta_{lo}-\cos\theta_{hi}\right)\Delta\phi
$$
- $v_{lo}$, $v_{hi}$ are the midpoints between adjacent channel speeds; the outermost edges are mirrored symmetrically.
- θ edges are 16 uniform steps over $[0,\pi]$; φ edges are 32 uniform steps over $[0,2\pi]$ ($\Delta\phi = 2\pi/32$).
- Volumes are computed per time (in (m/s)³) and saved as `meta/vvol`.
- As a check, the code prints the sum of all bin volumes next to the analytic spherical-shell volume.

## 4. Distribution-function processing ([`analysis/vdf_proc.process_vdf`](../fipcore/analysis/vdf_proc.py))

1. **Bad distributions.** Any sample whose error flag is non-zero is replaced by linear interpolation in time between good samples, bin by bin, with extrapolation at the ends (`scipy.interp1d`). $\sigma_f$ is interpolated the same way.
   - ⚠ Flag (F4): *any* non-zero flag bit triggers replacement, so the paper needs to say which flag bits count as "bad".
2. **Counts reconstruction.** $N = \mathrm{round}\big[(f/\sigma_f)^2\big]$, i.e. Poisson statistics are assumed. Bins with $N\le 0$, $f\le 0$ or NaN are set to NaN. One-count bins are kept.
3. **Units.** $f$ is converted from s³ cm⁻⁶ to s³ m⁻⁶ (×10¹²).
4. **Two products.** Both are carried through the rest of the pipeline:
   - `vdf_raw` $= f$ (s³ m⁻⁶)
   - `vdf_vol` $= f\,\Delta^3 v$ (m⁻³), the number density contained in each instrument bin

The event has 48% of raw bin values set to NaN by step 2. These are mostly empty (zero-count) bins.

## 5. Frames of reference and coordinate systems

### 5.1 X-line time and reconnection-frame velocity
- **B in LMN.** B is rotated into LMN using the matrix from §5.3 (`pyspedas.tvector_rotate`).
- **X-line index.** [`utils/helper_utils.bl_recx_idx`](../fipcore/utils/helper_utils.py) takes the index of maximum $|dB_L/dt|$ over the whole `trange` (computed with `np.gradient`). It also prints the index where $B_L$ crosses the midpoint of its start and end values, as a diagnostic, but the max-gradient index is the one used.
- **Frame velocity.** $\mathbf{U} = \mathbf{V}_i(t_X)$, the interpolated ion bulk velocity at that one sample (`vel_proc.vi_bl_flip`). It is **one constant vector for the whole interval**.
- **Galilean shift of the measurements** (`vel_proc.vv_to_recx`): $\mathbf{v}' = \mathbf{v} - \mathbf{U}$. The value of $f$ is unchanged, because $f$ is invariant under the shift.
- **Lorentz transform of the field** ([`analysis/evec_proc.evec_to_recx`](../fipcore/analysis/evec_proc.py)):
$$
\mathbf{E}' = \mathbf{E} + \mathbf{U}\times\mathbf{B}\qquad(\text{km/s}\cdot\text{nT}\times10^{-3}\to\text{mV/m})
$$
  applied at each time with the 30 ms-averaged $\mathbf{B}$.

⚠ Flag (F5): $t_X$, and therefore $\mathbf{U}$ and every frame-dependent output, depends on the choice of `trange`, because the max-gradient search runs over the whole window.

### 5.2 Field-aligned coordinates (FAC), time-dependent
- **Construction.** `pyspedas.fac_matrix_make` with `other_dim='Xgse'` is applied to the 30 ms-averaged $\mathbf{B}$ at each time sample:
$$
\hat{\mathbf{e}}_\parallel=\hat{\mathbf{b}},\qquad
\hat{\mathbf{e}}_{\perp2}=\frac{\hat{\mathbf{b}}\times\hat{\mathbf{x}}_{GSE}}{|\hat{\mathbf{b}}\times\hat{\mathbf{x}}_{GSE}|},\qquad
\hat{\mathbf{e}}_{\perp1}=\hat{\mathbf{e}}_{\perp2}\times\hat{\mathbf{b}}
$$
- **Ordering.** Matrix rows are $(\perp1,\perp2,\parallel)$, so component index 0, 1, 2 means ⊥1, ⊥2, ∥ throughout.
- **What gets rotated.** Reconnection-frame velocities are rotated sample by sample (`vel_proc.vv_to_fac`), as is $\mathbf{E}'$ (`evec_proc.evec_to_fac`, which uses [`utils/coord_utils.rotate_to_fac`](../fipcore/utils/coord_utils.py)).

### 5.3 LMN, one fixed matrix
- **Construction.** [`utils/coord_utils.lmn_matrix_make`](../fipcore/utils/coord_utils.py) runs minimum variance analysis (`pyspedas.minvar_matrix_make`, with `tslide=0`) on the 30 ms-averaged $\mathbf{B}$ over the **entire `trange`**, producing one matrix for the whole interval.
- **Axes.** Eigenvectors are ordered by descending eigenvalue: L = maximum, M = intermediate, N = minimum variance.
- **Sign convention.** After pyspedas's own sign handling, the code enforces:
  - $N_x>0$ (N points toward +X$_{GSE}$)
  - $L_z>0$ (L points toward +Z$_{GSE}$)
  - M flipped if exactly one of N or L was flipped, to keep the system right-handed
- **Event result (MMS1).** Rows of `lmn/rot_mat`:
  - L = (−0.283, −0.127, 0.951)
  - M = (0.285, −0.957, −0.043)
  - N = (0.916, 0.259, 0.307)
- **What gets rotated.** Velocities (`vel_proc.vv_to_lmn`), $\mathbf{E}'$ (`evec_proc.evec_to_lmn`) and $\mathbf{B}$.

⚠ Flag (F6): the MVA eigenvalues (and their ratios, λ_max/λ_int and λ_int/λ_min) are neither saved nor checked, so the quality of the LMN determination isn't recorded. The LMN also depends on `trange`.

### 5.4 FAC↔LMN projections
`coord_utils.fac_lmn_proj` computes, per time sample, $R_{FAC}R_{LMN}^{T}$ and its inverse, and saves them as `meta/fac2lmn` and `meta/lmn2fac`. These are diagnostics only and are not used in any calculation.

## 6. Velocity-space binning

### 6.1 Normalization speed ([`utils/helper_utils.compute_vth`](../fipcore/utils/helper_utils.py))
$$
T = \tfrac13\left(T_\parallel + 2T_\perp\right),\qquad v_{th}=\sqrt{2T/m} = c\sqrt{2T/(mc^2)},\qquad \langle v_{th}\rangle = \text{mean of } v_{th}(t) \text{ over } \texttt{trange}.
$$
A **single** $\langle v_{th}\rangle$ normalizes the grid for all times; it is computed from DES moments. ⚠ Flag (F7): the grid scale therefore depends on `trange` too.

### 6.2 Grid (`vel_proc.vv_to_vmap`)
- **Extent.** A Cartesian cube in normalized velocity $\tilde v_j = v'_j/\langle v_{th}\rangle$ spanning $[-v_{lim}, v_{lim})$, with $v_{lim}$ = `vth_lim` = 3.5. The same grid is used for all three components and for both FAC and LMN.
- **Bins.** Number of bins $n = \lceil 2v_{lim}/\Delta\rceil$, where $\Delta$ = `bin_width_frac`. For $\Delta = 0.25$ that is $n = 28$, with physical bin width $\approx 0.25\langle v_{th}\rangle \approx 1110$ km/s.
- **Index assignment.** Each measurement's index along axis $j$ is $\lfloor(\tilde v_j + v_{lim})/\Delta\rfloor$.
- **Excluded measurements.** A measurement is dropped if **any** component lies outside the cube. For MMS1, 42.6% of instrument bins (all bins, including empty ones) fall outside.
- **Saved quantities.** Bin edges and centres, both physical (km/s) and normalized: `edges`, `edges_n`, `binc`, `binc_n`.

⚠ Flag (F8): the edges and centres come from `linspace(-v_lim, v_lim, n+1)`, but indices are assigned with the nominal Δ. When $2v_{lim}/\Delta$ is not an integer (e.g. Δ = 0.3 or 0.4 with $v_{lim}$ = 3.5), the saved edges and centres don't match the bins that measurements were actually placed in, and the gradient in §7.3 uses the wrong spacing. For Δ = 0.10, 0.20, 0.25 and 0.50 they match. This affects the bin-width validation runs at 0.30 and 0.40.

### 6.3 Binning (`vdf_proc.bin_vdf_3d`, [`analysis/fpc_proc.bin_cprime_3d`](../fipcore/analysis/fpc_proc.py))
Measurements are scatter-added into the 3D grid independently for each time sample, with the count of contributing measurements saved as `npts_*`. There are two variants:

| Variant | Definition | Units | Functions |
|---|---|---|---|
| `_vol` (volume-weighted sum) | $\sum_{i\in\text{cell}} f_i\,\Delta^3 v_i$ | m⁻³ (density in the cell) | `bin_vdf_vol`, `bin_cprime_vol` |
| `_avg` (plain average) | $\frac{1}{N_{cell}}\sum_{i\in\text{cell}} f_i$ | s³ m⁻⁶ | `bin_vdf_avg`, `bin_cprime_avg` |

**The published figures use the `_vol` products** (`bvdf_vol`, `c_vol`).

⚠ Flag (F9): NaN values from §4 step 2 are included in the sums (`np.add.at`), so **one NaN measurement makes the whole cell NaN**. In the MMS1 file, 130,191 of the 571,161 occupied cell-times (23%) are NaN for this reason. Later steps use `nansum`, which treats these cells as zero. If the intent was to ignore zero-count measurements rather than blank the cell, this is a bug.

## 7. Field–particle correlation

### 7.1 Unbinned correlation (`fpc_proc.compute_cprime`)
At each instrument bin and each 30 ms sample, for each component $j$ (no sum over $j$):
$$
C'_{E_j}(\mathbf{v},t) = q\,v_j\,E'_j(t)\,F(\mathbf{v},t),\qquad q=-e\ \text{for electrons}
$$
- $\mathbf{v}$ and $\mathbf{E}'$ are both expressed in the same frame, FAC or LMN (reconnection frame).
- $F$ is either $f$ (giving `Cprime_raw`) or $f\Delta^3v$ (giving `Cprime_vol`).
- Units: km/s × mV/m = 1 m/s·V/m, so `Cprime_vol` comes out directly in **W m⁻³ per instrument bin**.

### 7.2 Binning and total energization
- $C'$ is binned exactly like $f$ (§6.3), giving `bCprime_vol` and `bCprime_avg`.
- **Energy transfer rate per component** (`fpc_proc.jEtot_cprime`):
$$
(\mathbf{J}\cdot\mathbf{E}')_j^{FPC}(t) = \sum_{\text{cells}} \text{bCprime\_vol}_j \quad (\texttt{nansum},\ \text{W m}^{-3}),
$$
  saved as `JE_tot`, shape (3, ntime).

  ⚠ Flag (F10): this sum covers only the ±3.5 ⟨v_th⟩ cube and treats NaN cells (F9) as zero, so it can underestimate the moment-based $\mathbf{J}\cdot\mathbf{E}'$ (§9).

### 7.3 Standard FPC (`fpc_proc.compute_fpc`)
The standard correlation $C_{E_j} = -q\frac{v_j^2}{2}\frac{\partial f}{\partial v_j}E_j$ is obtained from $C'$ using the identity
$$
C_{E_j} = -\frac{v_j}{2}\frac{\partial C'_{E_j}}{\partial v_j} + \frac{1}{2}C'_{E_j}.
$$
Substituting $C'_{E_j}=qv_jfE_j$ recovers $-q\frac{v_j^2}{2}E_j\,\partial f/\partial v_j$.
- The derivative is `np.gradient`: second-order central differences in the interior and first-order one-sided differences at the grid edges. The spacing is the median difference of the physical bin centres (km/s), and $v_j$ is the physical bin centre, so $C$ has the same units as the binned $C'$.
- It is applied to both `bCprime_vol` (giving `c_vol`, the published one) and `bCprime_avg` (giving `c_avg`).
- ⚠ Flag (F11): NaN cells (F9) and empty cells in `_avg` (0/0 = NaN) spread to their neighbours through the finite difference.

### 7.4 Correlation interval and time averaging
**No time averaging is applied anywhere in `fipcore`.** Every output is computed independently at each 30 ms DES sample: there is no correlation interval τ and no running mean of $C'$ or $C$.

⚠ Flag (F12): the effective correlation interval is one DES sample (30 ms). If the paper describes a correlation interval τ, it needs to say this explicitly.

### 7.5 Reductions used for figures
- **2D plots** ([`plotting/imagecont_vdf_fpc.imagecont_vdf_fpc`](../fipcore/plotting/imagecont_vdf_fpc.py), [`utils/helper_utils.slice3d_to_2d`](../fipcore/utils/helper_utils.py) with `mode="integrate"`): `nansum` over the third velocity axis of `bvdf_vol` and `c_vol` at one time index. This is a sum over cells, not multiplied by a bin width. `mode="slice"` (a mid-plane cut) exists but isn't used for the published figures.
- **1D plots** (`load_plt_mms_data` in `Publication_plots.ipynb`): `nansum` of `c_vol` and `bvdf_vol` over two velocity axes, plotted against normalized bin centres (`binc_n`).
- **Peak times.** Figures use per-probe time indices `t_peaks = [62, 58, 60, 62]` (MMS1–4). These are set by hand in the notebook. ⚠ Flag (F13): the criterion for choosing them is not in the code, so the paper needs to state it.
- **Colour scales.** VDF panels show `log10(f)`; values ≤ 0 are masked.

### 7.6 Folded FPC (`fpc_mrx_main.fpc_mrx_fold`, FAC only)
- **Input.** The 2D-integrated planes of `c_vol` from §7.5.
- **Operation.** Each $C_{E_j}$ is folded about $v_j=0$ along its own velocity axis: $C_{E_j}(|v_j|) = C_{E_j}(+v_j) + C_{E_j}(-v_j)$. The $v_j<0$ half of the output is left at zero.
- **Planes folded:**
  - $C_{\perp1}$: in the (⊥1,⊥2) and (⊥1,∥) planes
  - $C_{\perp2}$: in the (⊥1,⊥2) and (∥,⊥2) planes
  - $C_\parallel$: in the (⊥1,∥) and (∥,⊥2) planes
- **Fold index.** The split is at `nbin // 2`, which is symmetric only when `nbin` is even. It is even (28) for Δ = 0.25, but odd (35) for Δ = 0.20. At Δ = 0.20 the fold would pair mismatched bins and the $v_j\approx0$ bin would be counted on only one side.
- **Output.** Saved as `fac/cx_folds`, `cy_folds` and `cz_folds`.

## 8. Background subtraction, δf (`subtract_f0=True`)

`vdf_proc.equilibrium_vdf` evaluates a gyrotropic **bi-Maxwellian** at every measurement velocity and time sample:
$$
f_0 = \frac{n}{\pi^{3/2}\,w_\perp^2\,w_\parallel}\exp\!\left[-\frac{v_{\perp1}^2+v_{\perp2}^2}{w_\perp^2}-\frac{v_\parallel^2}{w_\parallel^2}\right],\qquad w_{\parallel,\perp}=\sqrt{2T_{\parallel,\perp}/m}.
$$
- $n$, $T_\parallel$ and $T_\perp$ are the DES moments at each time, so $f_0$ is local and time-varying.
- Velocities are the FAC, reconnection-frame ones from §5.
- Then $\delta f\,\Delta^3v = f\Delta^3v - f_0\Delta^3v$ replaces `vdf_vol`, and $f_0$ is saved as `fac/vdf_eq`. Output files are tagged `df`.
- The docstrings call $f_0$ "tri-Maxwellian", but the formula has a single perpendicular temperature, which makes it a bi-Maxwellian.

⚠ Flags:
- **F14:** with `v_flow=None` (as in the published runs), $f_0$ is centred at $\mathbf{v}'=0$, i.e. at the ion velocity at the X-line $\mathbf{U}$, **not** at the local electron bulk velocity $\mathbf{V}_e(t)$. Any electron drift relative to $\mathbf{U}$ therefore stays in δf.
- **F15:** the subtraction is applied to the `_vol` chain only (`bvdf_vol`, `Cprime_vol`, `bCprime_vol`, `c_vol`, `JE_tot`). `vdf_raw`, `Cprime_raw`, `bvdf_avg`, `bCprime_avg` and `c_avg` stay full-$f$ even in `df` files.
- **F16:** measurements that are NaN in $f$ (§4) stay NaN in δf; they are not replaced by $-f_0$.

## 9. Ancillary plasma parameters (`fpc_mrx_main.fpc_mrx_plasma_params`)

Computed on the DES time base, with ion moments interpolated to it (§2):
- **Current density from moments.** $\mathbf{J} = n_e\,e\,(\mathbf{V}_i-\mathbf{V}_e)$ in nA m⁻². The electron density is used for both species, which assumes quasi-neutrality. It is rotated into FAC and LMN with the same matrices as §5 (read back from the HDF5 file). The current is not computed with the curlometer.
- **Energy transfer.** $\mathbf{J}\cdot\mathbf{E}'$ and its three components $J_jE'_j$ in GSE, FAC and LMN (nW m⁻³), using $\mathbf{E}'$ from §5.1 ([`analysis/jdotE_proc.compute_jdotE`](../fipcore/analysis/jdotE_proc.py)). Compare with `JE_tot` (§7.2), which is in W m⁻³, a factor of 10⁹ different.
- **Plasma beta** (`helper_utils.compute_beta_par/perp/scalar`): $\beta = n k T/(B^2/2\mu_0)$, computed separately with $T_\parallel$, $T_\perp$ and $T=(T_\parallel+2T_\perp)/3$, for electrons and ions. $|B|$ is the 30 ms-averaged FGM magnitude.
- **Bulk velocities.** $\mathbf{V}_e$ and $\mathbf{V}_i$, rotated into FAC and LMN.
- `bg_ratio` (guide-field ratio) is stored as metadata only.

## 10. Outputs

Results are written to `data/mms{probe}_{rate}_{f|df}_{species}_{Δ:.2f}_{YYYYMMDD}_{hhmmss}_{hhmmss}.h5` by [`utils/io_utils.h5sav`](../fipcore/utils/io_utils.py), which appends to the file and overwrites datasets that already exist. The groups are `meta`, `gse`, `fac` and `lmn`. [CLAUDE.md](../CLAUDE.md) lists the keys. The main published quantities are `fac/bvdf_vol`, `fac/c_vol`, `fac/c*_folds` and `fac/JE_tot`, plus the `lmn/` equivalents.

---

## 11. Flags and open questions

| # | Issue | Affects paper? |
|---|---|---|
| F1 | Averaged E/B/V_sc are on avg_data's uniform grid and paired with DES samples by index, not timestamp | Minor. State "boxcar-averaged to 30 ms" |
| F2 | Channels with $\mathcal{E}_{table}<eV_{sc}$ would become NaN and be dropped (didn't occur for this event) | No, for this event |
| F3 | Instrument (DBCS) velocities treated as GSE; no despin rotation applied | State the assumption |
| F4 | Any non-zero FPI error flag triggers time interpolation of the whole distribution | State the criterion |
| F5 | X-line time (max \|dB_L/dt\| over `trange`) and frame velocity **U** depend on `trange`; one **U** for the whole interval | **Yes**, describe |
| F6 | MVA eigenvalue ratios not saved; LMN quality not reported | Report λ ratios |
| F7 | Grid normalization ⟨v_th⟩ is a mean over `trange` | Describe |
| F8 | Saved edges/centres inconsistent with bin assignment when 7/Δ is not an integer (Δ = 0.3, 0.4) | **Yes**, for the bin-width comparison |
| F9 | Any zero-count/NaN measurement blanks the whole cell in `_vol` and `_avg` sums (23% of occupied cells here) | **Yes**. Likely unintended; review before publishing |
| F10 | `JE_tot` covers only the ±3.5 v_th cube and skips NaN cells | Mention when comparing with moment J·E′ |
| F11 | NaN cells spread through ∂/∂v_j | Follows from F9 |
| F12 | **No correlation-interval (τ) averaging**: every result is a single 30 ms sample | **Yes**, state explicitly |
| F13 | Peak time indices [62, 58, 60, 62] set by hand | State the selection criterion |
| F14 | f₀ centred on the ion X-line velocity U, not the local V_e | **Yes**, describe |
| F15 | δf applies only to the `_vol` products | Describe which products are shown |
| F16 | NaN bins remain NaN in δf (not −f₀) | Minor |

Smaller code issues noticed. These don't change results for the published runs:
- `vdf_proc.equilibrium_vdf` raises the misspelled `ValuoeError` for an invalid species, which would surface as a `NameError`.
- `is_interleaved` passes `atol` into `np.allclose`'s positional `rtol` slot, so the check uses rtol = 10⁻⁶ and the default atol.
- The `fpc_mrx_main` docstring says `mean_phi` defaults to True; the actual default is False.
- Species strings are `'e'`/`'ion'` in analysis code but `'e'`/`'i'` in plotting docstrings.
