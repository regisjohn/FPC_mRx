# fipcore analysis pipeline

`fipcore` (FIeld Particle COrrelation for Reconnection Events) computes the electron field–particle correlation (FPC) in velocity space from MMS burst-mode data. This page describes how the pipeline works, step by step, and which function does each step. Function references use the form `module.function`. All modules are under [`fipcore/`](../fipcore/).

---

## 0. Entry points

All three entry points are in [`analysis/fpc_mrx_main.py`](../fipcore/analysis/fpc_mrx_main.py). Each one processes one spacecraft over one time interval and writes to the HDF5 file described in §10:

| Function | What it does |
|---|---|
| `fpc_mrx_main` | Runs the full chain in §1–8 and writes the results |
| `fpc_mrx_fold` | Reads `c_vol` from that file and adds the folded FPC (§7.6) |
| `fpc_mrx_plasma_params` | Reads E′ and the rotation matrices from that file and adds moment-based parameters (§9) |

Run them in that order. `fpc_mrx_fold` must be rerun whenever `fpc_mrx_main` is, so the folds match the new `c_vol` (see §10).

**Parameters of `fpc_mrx_main`:**

| Parameter | Default | Meaning |
|---|---|---|
| `trange` | — | `['YYYY-MM-DD/hh:mm:ss', 'YYYY-MM-DD/hh:mm:ss']` |
| `probe` | `'1'` | MMS spacecraft |
| `species` | `'e'` | Species (see note below) |
| `data_rate`, `level` | `'brst'`, `'l2'` | MMS data product |
| `bin_width_frac` | 0.25 | Velocity bin width Δ, as a fraction of ⟨v_th⟩ (§6) |
| `vth_lim` | 3.5 | Half-extent of the velocity grid, in units of ⟨v_th⟩ (§6) |
| `mean_phi` | `False` | Use time-averaged instead of per-sweep azimuth angles (§3.3) |
| `subtract_f0` | `False` | Subtract a bi-Maxwellian background, giving δf (§8) |
| `v_flow` | `None` | Drift velocity of f₀ in the reconnection frame, m/s (§8) |
| `no_update` | `True` | Use the local pyspedas data cache and download only missing files |
| `get_support_data` | `True` | Load support variables (energy table, angles) |
| `lmn_mat_name`, `fac_mat_name` | `'lmn_matrix'`, `'fac_matrix'` | tplot names for the rotation matrices |

**Species.** The variable names loaded by `fpc_mrx_main` are the FPI-DES (electron) ones, so the pipeline currently runs for electrons only. The lower-level functions accept `'e'` or `'ion'`.

**Example:**
```python
from fipcore import fpc_mrx_main, fpc_mrx_fold, fpc_mrx_plasma_params
trange = ['2015-12-09/05:03:55', '2015-12-09/05:03:59']
fpc_mrx_main(trange, probe='1', bin_width_frac=0.25, subtract_f0=False)
fpc_mrx_fold(trange, probe='1', bin_width_frac=0.25, subtract_f0=False)
fpc_mrx_plasma_params(trange, probe='1', bin_width_frac=0.25, subtract_f0=False)
```

### Conventions used across modules
- **tplot variables pass data between steps.** Most functions take and return pyspedas tplot variable *names*, not arrays. `fpc_mrx_main` renames the loaded MMS variables to short names (`nrgy`, `scpot`, `vdf_raw`, `bvec_gse`, `evec_gse`, …) that the later steps expect.
- **The tplot namespace is cleared first.** Existing tplot variables are not reliably overwritten, so every entry point begins with `del_data('*')`. Keep this call in any new entry point.
- **Array shapes:**
  - measurement velocities: `(3, 16384, ntime)`
  - distributions: `(16384, ntime)`
  - binned 3D grids: `(n, n, n, ntime)`
  - binned correlations: `(3, n, n, n, ntime)`
- **Component order.** Index 0, 1, 2 means ⊥1, ⊥2, ∥ in FAC and L, M, N in LMN.

## 1. Data loaded

`fpc_mrx_main` loads data with pyspedas (`mms.fpi`, `mms.edp`, `mms.fgm`, with `time_clip=True`):

| Instrument | Quantity | Variable |
|---|---|---|
| FPI-DES | 3D electron distribution $f$, its error $\sigma_f$, per-distribution error flags | `des_dist_brst`, `des_disterr_brst`, `des_errorflags_brst_dist` |
| FPI-DES | Energy table (per sweep), azimuth angles $\phi$ (per sweep) | `des_energy_brst`, `des_phi_brst` |
| FPI-DES | Moments: $n_e$, $T_{e\parallel}$, $T_{e\perp}$, $\mathbf{V}_e$ (GSE) | `des_numberdensity`, `des_temppara`, `des_tempperp`, `des_bulkv_gse` |
| FPI-DIS | Ion bulk velocity $\mathbf{V}_i$ (GSE) | `dis_bulkv_gse_brst` |
| EDP | Electric field $\mathbf{E}$ (GSE), spacecraft potential $V_{sc}$ | `edp_dce_gse_brst_l2`, `edp_scpot_brst_l2` |
| FGM | Magnetic field $\mathbf{B}$ (GSE) | `fgm_b_gse_brst_l2` |

`edp_dce_par_epar_brst_l2` is also loaded but not used.

**Data location.** pyspedas reads and caches CDF files under `SPEDAS_DATA_DIR`. If that variable isn't set, it uses `./pydata` relative to the working directory.

## 2. Common time base (30 ms DES cadence)

Every quantity is put on the DES burst time base, which is the timestamps of $\mathbf{V}_e$.

- **Faster fields ($\mathbf{E}$, $\mathbf{B}$, $V_{sc}$) are boxcar-averaged** by [`utils/helper_utils.downsample_cad`](../fipcore/utils/helper_utils.py), which wraps pyspedas `avg_data`. The window width in samples is $N_{in}/N_{DES}$, the ratio of the number of input points to the number of DES points over `trange`.
  - `avg_data` lays its bins out uniformly from `trange[0]`, at $t_k = t_{start} + (k+\tfrac12)\Delta t$, and the averaged values are paired with DES samples by array index. The resulting timestamp offsets are well under one DES interval.
- **Slower data ($\mathbf{V}_i$, 150 ms DIS cadence) is linearly interpolated** onto the DES timestamps by `helper_utils.upsample_cad`, which wraps pyspedas `tinterpol`.

## 3. Velocity of each measurement

### 3.1 Energy table and interleaving ([`analysis/vel_proc.nrgy_to_vbin`](../fipcore/analysis/vel_proc.py))
- The energy table may alternate between two tables on consecutive sweeps (interleaving). `helper_utils.is_interleaved` detects this, and the result is saved as `meta/interleave_info`.
- Either way, each sweep's own energy table is used, so the velocity grid is built separately for every time sample.

### 3.2 Spacecraft-potential correction (`vel_proc.nrgy_to_vbin`)
$$
\mathcal{E}_{corr} = \mathcal{E}_{table} + q\,V_{sc}, \qquad q=-1\ \text{(e)},\ +1\ \text{(ion)},\ \text{energies in eV, } V_{sc} \text{ in V}
$$
For electrons this is $\mathcal{E}_{table}-eV_{sc}$, using the 30 ms-averaged $V_{sc}$ at each time. The speed of each energy channel is the non-relativistic
$$
v = c\sqrt{2\,\mathcal{E}_{corr}/(mc^2)} \quad \text{(km/s)}.
$$
A channel below $eV_{sc}$ would give $\mathcal{E}_{corr}<0$ and a NaN speed, and those measurements would fall off the grid in §6. Check this when analysing low-density regions where $V_{sc}$ is large.

### 3.3 Velocity vectors (`vel_proc.vbin_to_vv`)
Each of the 32 (φ) × 16 (θ) × 32 (energy) = 16384 bins becomes a velocity vector:
$$
\mathbf{v} = -\,v\,(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta).
$$
- The minus sign converts the detector look direction into the direction of particle travel.
- **Polar angles** are the fixed FPI values $\theta_k = 5.625^\circ + 11.25^\circ k$, $k=0..15$ (the values in `des_theta_brst`).
- **Azimuth angles** are the per-sweep values from `des_phi_brst`, or their time average when `mean_phi=True`.
- **Frame.** The L2 angles are in the despun spacecraft frame (`DBCS` in the CDF metadata). The velocities are used together with GSE quantities ($\mathbf{U}$, $\mathbf{B}$, $\mathbf{E}$) without a further rotation.

### 3.4 Phase-space volume of each instrument bin (`vel_proc.compute_vbin_vol`)
$$
\Delta^3 v = \tfrac13\left(v_{hi}^3 - v_{lo}^3\right)\left(\cos\theta_{lo}-\cos\theta_{hi}\right)\Delta\phi
$$
- $v_{lo}$ and $v_{hi}$ are the midpoints between adjacent channel speeds; the outermost edges are mirrored.
- θ edges are 16 uniform steps over $[0,\pi]$; φ edges are 32 uniform steps over $[0,2\pi]$ ($\Delta\phi = 2\pi/32$).
- Volumes are computed per time sample, in (m/s)³, and saved as `meta/vvol`.
- As a check, the function prints the sum of all bin volumes next to the analytic spherical-shell volume.

## 4. Distribution processing ([`analysis/vdf_proc.process_vdf`](../fipcore/analysis/vdf_proc.py))

1. **Bad distributions.** A sample with any non-zero L2 error flag is replaced by linear interpolation in time between good samples, bin by bin, with extrapolation at the ends (`scipy.interp1d`). $\sigma_f$ is interpolated the same way. The flag bits are: bit 0 = manually flagged interval, bit 1 = overcounting/saturation likely, bit 2 = compression pipeline error.
2. **Counts reconstruction.** $N = \mathrm{round}\big[(f/\sigma_f)^2\big]$, assuming Poisson statistics. Bins with $N\le 0$ or $f\le 0$ are set to $f = 0$: no counts is a measurement of no particles, not missing data. One-count bins are kept.
3. **Units.** $f$ is converted from s³ cm⁻⁶ to s³ m⁻⁶ (×10¹²).
4. **Two products** are carried through the rest of the pipeline:
   - `vdf_raw` $= f$ (s³ m⁻⁶)
   - `vdf_vol` $= f\,\Delta^3 v$ (m⁻³), the number density contained in each instrument bin

## 5. Frames of reference and coordinate systems

### 5.1 X-line time and reconnection-frame velocity
- **B in LMN.** B is rotated into LMN with the matrix from §5.3 (`pyspedas.tvector_rotate`).
- **X-line index.** [`utils/helper_utils.bl_recx_idx`](../fipcore/utils/helper_utils.py) takes the index of maximum $|dB_L/dt|$ over the whole `trange` (computed with `np.gradient`). It also prints the index where $B_L$ crosses the midpoint of its start and end values, as a diagnostic, but the max-gradient index is the one used. Because the search covers the whole window, the choice of `trange` determines $t_X$.
- **Frame velocity.** $\mathbf{U} = \mathbf{V}_i(t_X)$, the interpolated ion bulk velocity at that sample (`vel_proc.vi_bl_flip`). It is one constant vector for the whole interval.
- **Galilean shift of the measurements** (`vel_proc.vv_to_recx`): $\mathbf{v}' = \mathbf{v} - \mathbf{U}$. The value of $f$ is unchanged.
- **Lorentz transform of the field** ([`analysis/evec_proc.evec_to_recx`](../fipcore/analysis/evec_proc.py)), applied at each time with the 30 ms-averaged $\mathbf{B}$:
$$
\mathbf{E}' = \mathbf{E} + \mathbf{U}\times\mathbf{B}\qquad(\text{km/s}\cdot\text{nT}\times10^{-3}\to\text{mV/m})
$$

### 5.2 Field-aligned coordinates (FAC), time-dependent
- **Construction.** `pyspedas.fac_matrix_make` with `other_dim='Xgse'` is applied to the 30 ms-averaged $\mathbf{B}$ at each time sample:
$$
\hat{\mathbf{e}}_\parallel=\hat{\mathbf{b}},\qquad
\hat{\mathbf{e}}_{\perp2}=\frac{\hat{\mathbf{b}}\times\hat{\mathbf{x}}_{GSE}}{|\hat{\mathbf{b}}\times\hat{\mathbf{x}}_{GSE}|},\qquad
\hat{\mathbf{e}}_{\perp1}=\hat{\mathbf{e}}_{\perp2}\times\hat{\mathbf{b}}
$$
- **Ordering.** Matrix rows are $(\perp1,\perp2,\parallel)$.
- **What gets rotated.** Reconnection-frame velocities, sample by sample (`vel_proc.vv_to_fac`), and $\mathbf{E}'$ (`evec_proc.evec_to_fac`, which uses [`utils/coord_utils.rotate_to_fac`](../fipcore/utils/coord_utils.py)).

### 5.3 LMN, one fixed matrix
- **Construction.** [`utils/coord_utils.lmn_matrix_make`](../fipcore/utils/coord_utils.py) runs minimum variance analysis (`pyspedas.minvar_matrix_make`, `tslide=0`) on the 30 ms-averaged $\mathbf{B}$ over the entire `trange`, giving one matrix for the whole interval.
- **Axes.** Eigenvectors are ordered by descending eigenvalue: L = maximum, M = intermediate, N = minimum variance.
- **Sign convention.** After pyspedas's own sign handling, the code enforces:
  - $N_x>0$ (N points toward +X$_{GSE}$)
  - $L_z>0$ (L points toward +Z$_{GSE}$)
  - M is flipped if exactly one of N or L was flipped, to keep the system right-handed
- **What gets rotated.** Velocities (`vel_proc.vv_to_lmn`), $\mathbf{E}'$ (`evec_proc.evec_to_lmn`) and $\mathbf{B}$.
- **MVA eigenvalues.** Saved as `lmn/eigvals` ($\lambda_{max}, \lambda_{int}, \lambda_{min}$, nT²), so the eigenvalue ratios can be checked to judge how well LMN is determined.

### 5.4 FAC↔LMN projections
`coord_utils.fac_lmn_proj` computes $R_{FAC}R_{LMN}^{T}$ and its inverse at each time sample and saves them as `meta/fac2lmn` and `meta/lmn2fac`. These are diagnostics and are not used in any calculation.

## 6. Velocity-space binning

### 6.1 Normalization speed ([`utils/helper_utils.compute_vth`](../fipcore/utils/helper_utils.py))
$$
T = \tfrac13\left(T_\parallel + 2T_\perp\right),\qquad v_{th}=\sqrt{2T/m} = c\sqrt{2T/(mc^2)},\qquad \langle v_{th}\rangle = \text{mean of } v_{th}(t) \text{ over } \texttt{trange}.
$$
A single ⟨v_th⟩, computed from DES moments, normalizes the grid for all times. The physical bin width is therefore Δ·⟨v_th⟩, which depends on the chosen `trange`.

### 6.2 Grid (`vel_proc.vv_to_vmap`)
- **Bins.** $n = \lceil 2v_{lim}/\Delta\rceil$, with $v_{lim}$ = `vth_lim` and Δ = `bin_width_frac`. For the defaults (3.5, 0.25), $n$ = 28.
- **Extent.** A Cartesian cube in normalized velocity $\tilde v_j = v'_j/\langle v_{th}\rangle$ spanning $[-\tfrac{n\Delta}{2}, \tfrac{n\Delta}{2})$. That is exactly ±`vth_lim` when $2v_{lim}/\Delta$ is a whole number, and up to half a bin wider otherwise (e.g. ±3.6 for Δ = 0.30). Every bin is exactly Δ wide and the grid is centred on 0. The same grid is used for all three components and for both FAC and LMN.
- **Index assignment.** Each measurement's index along axis $j$ is $\lfloor(\tilde v_j + n\Delta/2)/\Delta\rfloor$.
- **Excluded measurements.** A measurement is dropped if any of its components lies outside the cube.
- **Saved quantities.** Bin edges and centres, both physical (km/s) and normalized: `edges`, `edges_n`, `binc`, `binc_n`.
- **Where v = 0 falls.** For even $n$, 0 is a bin edge. For odd $n$, 0 is the centre of the middle bin.

`vel_proc.vv_to_vmap_3d` is a variant that accepts a different extent and bin width for each component.

**Checking the binning** ([`plotting/plt_vmap_check.plt_vmap_check`](../fipcore/plotting/plt_vmap_check.py)). This diagnostic plot shows whether measurements land in the right grid cells. It isn't called by the pipeline; use it from a notebook after `vv_to_vmap`.

- **Inputs.** The velocities in one coordinate system (`vv_fac` or `vv_lmn`, km/s), the matching `vmap`, the normalized edges `edges_n`, and ⟨v_th⟩.
- **Selection.** It takes one time sample (`sweep_idx`) and two velocity components: `comp_x` on the horizontal axis and `comp_y` on the vertical.
- **Drawing.**
  - Each measurement is plotted at its normalized velocity, $v/\langle v_{th}\rangle$.
  - For every occupied cell, the outline of that cell is drawn from the saved edges.
  - Points and outlines are coloured by cell index, so a point and the cell it was assigned to share a colour.
  - The colour is `3*(i_y mod 3) + (i_x mod 3)`, using 9 of the 10 `tab10` colours. The pattern repeats every 3 cells along each axis, so every cell differs in colour from all 8 of its neighbours, including diagonal ones.
- **Reading the plot.** Every point should sit inside the outline of its own colour. A point outside its outline means the bin assignment and the saved edges disagree.
- **Covering all three components.** Each bin index is computed separately for each velocity component, so checking planes that include every component at least once verifies the whole assignment. Each plot shows the measurements binned in its two components; some of these are outside the grid in the third component and so aren't in the 3D binning (§6.2).
- **Defaults.** The axes span ±1.5 ⟨v_th⟩, the central part of the grid; `xlim`/`ylim` widen this. The figure is 8 × 8 in; `figsize` changes it.
- **Example.** `scripts/vdf_generation.ipynb` ("Validating the Coordinate Binning") checks the first time sample in four planes: (v∥, v⊥1) and (v∥, v⊥2) in FAC, and (v_N, v_L) and (v_N, v_M) in LMN.

### 6.3 Binning (`vdf_proc.bin_vdf_3d`, [`analysis/fpc_proc.bin_cprime_3d`](../fipcore/analysis/fpc_proc.py))
Measurements are scatter-added (`np.add.at`) into the 3D grid independently for each time sample. The number of contributing measurements is saved as `npts_*`. There are two variants:

| Variant | Definition | Units | Functions |
|---|---|---|---|
| `_vol` (volume-weighted sum) | $\sum_{i\in\text{cell}} f_i\,\Delta^3 v_i$ | m⁻³ (density in the cell) | `bin_vdf_vol`, `bin_cprime_vol` |
| `_avg` (plain average) | $\frac{1}{N_{cell}}\sum_{i\in\text{cell}} f_i$ | s³ m⁻⁶ | `bin_vdf_avg`, `bin_cprime_avg` |

- The plotting functions use the `_vol` products (`bvdf_vol`, `c_vol`).
- Zero-count measurements enter both as $f = 0$ (§4): they add nothing to `_vol` and count toward the `_avg` mean.
- Cells that no measurement reaches are 0 in `_vol` and 0/0 = NaN in `_avg`.

## 7. Field–particle correlation

### 7.1 Unbinned correlation (`fpc_proc.compute_cprime`)
At each instrument bin and each 30 ms sample, for each component $j$ (no sum over $j$):
$$
C'_{E_j}(\mathbf{v},t) = q\,v_j\,E'_j(t)\,F(\mathbf{v},t),\qquad q=-e\ \text{for electrons}
$$
- $\mathbf{v}$ and $\mathbf{E}'$ are in the same frame, FAC or LMN (reconnection frame).
- $F$ is either $f$ (giving `Cprime_raw`) or $f\Delta^3v$ (giving `Cprime_vol`).
- Units: km/s × mV/m = 1 m/s·V/m, so `Cprime_vol` comes out directly in W m⁻³ per instrument bin.

### 7.2 Binning and total energization
- $C'$ is binned exactly like $f$ (§6.3), giving `bCprime_vol` and `bCprime_avg`.
- **Energy transfer rate per component** (`fpc_proc.jEtot_cprime`), saved as `JE_tot` with shape (3, ntime):
$$
(\mathbf{J}\cdot\mathbf{E}')_j^{FPC}(t) = \sum_{\text{cells}} \text{bCprime\_vol}_j \quad (\texttt{nansum},\ \text{W m}^{-3})
$$
- The sum covers only the velocity cube of §6.2; particles outside it don't contribute. Keep this in mind when comparing with the moment-based $\mathbf{J}\cdot\mathbf{E}'$ (§9).

### 7.3 Standard FPC (`fpc_proc.compute_fpc`)
The standard correlation $C_{E_j} = -q\frac{v_j^2}{2}\frac{\partial f}{\partial v_j}E_j$ is obtained from $C'$ using the identity
$$
C_{E_j} = -\frac{v_j}{2}\frac{\partial C'_{E_j}}{\partial v_j} + \frac{1}{2}C'_{E_j}.
$$
Substituting $C'_{E_j}=qv_jfE_j$ recovers $-q\frac{v_j^2}{2}E_j\,\partial f/\partial v_j$.
- **Derivative.** `np.gradient`: second-order central differences in the interior and first-order one-sided differences at the grid edges. The spacing is the median difference of the physical bin centres (km/s), and $v_j$ is the physical bin centre, so $C$ has the same units as the binned $C'$.
- **Outputs.** Applied to `bCprime_vol` (giving `c_vol`) and to `bCprime_avg` (giving `c_avg`).
- **NaNs.** `c_vol` contains no NaNs. In `c_avg`, empty cells (0/0) are NaN, and they spread to their neighbours through the finite difference.

### 7.4 Correlation interval
The correlation is computed independently at each 30 ms DES sample. No averaging over a correlation interval is applied.

### 7.5 Reductions used by the plotting functions
- **2D plots** ([`plotting/imagecont_vdf_fpc.imagecont_vdf_fpc`](../fipcore/plotting/imagecont_vdf_fpc.py), using [`utils/helper_utils.slice3d_to_2d`](../fipcore/utils/helper_utils.py) with `mode="integrate"`): `nansum` of `bvdf_vol` and `c_vol` over the third velocity axis at one time index. This is a sum over cells, not multiplied by a bin width. `mode="slice"` instead takes a single plane (the middle one by default).
- **Colour scales.** VDF panels show `log10(f)`, with values ≤ 0 masked. δf panels use a symmetric linear scale.

### 7.6 Folded FPC (`fpc_mrx_main.fpc_mrx_fold`, FAC only)
- **Input.** The 2D-integrated planes of `c_vol` from §7.5.
- **Operation.** Each $C_{E_j}$ is folded about $v_j=0$ along its own velocity axis: $C_{E_j}(|v_j|) = C_{E_j}(+v_j) + C_{E_j}(-v_j)$. The $v_j<0$ half of the output is left at zero.
- **Planes folded:**
  - $C_{\perp1}$: in the (⊥1,⊥2) and (⊥1,∥) planes
  - $C_{\perp2}$: in the (⊥1,⊥2) and (∥,⊥2) planes
  - $C_\parallel$: in the (⊥1,∥) and (∥,⊥2) planes
- **Fold index.** The positive half starts at `idx = nbin // 2`.
  - Even `nbin`: $v_j = 0$ is the edge between bins `idx−1` and `idx`, and bin `idx+k` receives bin `idx−1−k`.
  - Odd `nbin`: bin `idx` is centred on $v_j = 0$ and keeps its own value, and bin `idx+k` receives bin `idx−k`.
- **Output.** Saved as `fac/cx_folds`, `cy_folds` and `cz_folds`, each with shape (2, n, n, ntime): the two folded planes per component.
- **Plotting.** `plotting/imagecont_vdf_fpc.imagecont_vdf_fpcfold` plots the folds.

## 8. Background subtraction, δf (`subtract_f0=True`)

`vdf_proc.equilibrium_vdf` evaluates a gyrotropic bi-Maxwellian (a 3D Maxwellian with equal perpendicular temperatures, $T_{\perp1} = T_{\perp2} = T_\perp$) at every measurement velocity and time sample:
$$
f_0 = \frac{n}{\pi^{3/2}\,w_\perp^2\,w_\parallel}\exp\!\left[-\frac{v_{\perp1}^2+v_{\perp2}^2}{w_\perp^2}-\frac{v_\parallel^2}{w_\parallel^2}\right],\qquad w_{\parallel,\perp}=\sqrt{2T_{\parallel,\perp}/m}.
$$
- **Inputs.** $n$, $T_\parallel$ and $T_\perp$ are the DES moments at each time, so $f_0$ is local and time-varying. The velocities are the FAC, reconnection-frame ones from §5.
- **Drift.** With `v_flow=None`, $f_0$ is centred at $\mathbf{v}'=0$, the ion velocity at the X-line $\mathbf{U}$, so electron drift relative to $\mathbf{U}$ appears in δf. Passing `v_flow` (m/s, FAC) shifts the centre of $f_0$.
- **Subtraction.** $f_0\Delta^3v$ is subtracted from `vdf_vol` and $f_0$ from `vdf_raw`, so every downstream product (`bvdf_*`, `Cprime_*`, `bCprime_*`, `c_vol`, `c_avg`, `JE_tot`) is δf-based. $f_0$ is saved as `fac/vdf_eq`, and the output file is tagged `df`.
- **Empty bins.** In zero-count bins, where $f = 0$, this gives $\delta f = -f_0$.

## 9. Ancillary plasma parameters (`fpc_mrx_main.fpc_mrx_plasma_params`)

Computed on the DES time base, with ion moments interpolated to it (§2):
- **Current density from moments.** $\mathbf{J} = n_e\,e\,(\mathbf{V}_i-\mathbf{V}_e)$, in nA m⁻². The electron density is used for both species, which assumes quasi-neutrality. $\mathbf{J}$ is rotated into FAC and LMN with the matrices from §5, read back from the HDF5 file.
- **Energy transfer.** $\mathbf{J}\cdot\mathbf{E}'$ and its three components $J_jE'_j$ in GSE, FAC and LMN, in nW m⁻³, using $\mathbf{E}'$ from §5.1 ([`analysis/jdotE_proc.compute_jdotE`](../fipcore/analysis/jdotE_proc.py)). `JE_tot` (§7.2) is in W m⁻³, a factor of 10⁹ different.
- **Plasma beta** (`helper_utils.compute_beta_par/perp/scalar`): $\beta = n k T/(B^2/2\mu_0)$, computed separately with $T_\parallel$, $T_\perp$ and $T=(T_\parallel+2T_\perp)/3$, for electrons and ions. $|B|$ is the 30 ms-averaged FGM magnitude.
- **Bulk velocities.** $\mathbf{V}_e$ and $\mathbf{V}_i$, rotated into FAC and LMN.
- **`bg_ratio`** (guide-field ratio) is stored as metadata only.
- **Energy spectrograms.** The omnidirectional electron and ion energy spectra are saved to a separate `.pyspd` tplot file next to the HDF5 file.

## 10. Outputs

Results are written to `data/mms{probe}_{rate}_{f|df}_{species}_{Δ:.2f}_{YYYYMMDD}_{hhmmss}_{hhmmss}.h5` by [`utils/io_utils.h5sav`](../fipcore/utils/io_utils.py). The filename is built by `io_utils.mms_name_make`, and the `data/` directory is located relative to the package, not the working directory.

**Writing behaviour.** `h5sav` opens an existing file in append mode and replaces only the datasets it writes; any other datasets already in the file are kept. This is why `fpc_mrx_fold` must be rerun after `fpc_mrx_main`. Each dataset gets a `units` attribute from `io_utils.UNITS_MAP`, or `"N/A"` if its name isn't listed there.

**Main datasets:**

| Group | Datasets |
|---|---|
| `meta` | `time`, `trange`, `probe`, `species`, `bin_width_frac`, `vth_lim`, `vthe`, `vth_mean`, `subtract_f0`, `interleave_info`, `vvol`, `vdf_raw`, `vdf_vol`, `fac2lmn`, `lmn2fac` |
| `gse` | `vv`, `bvec`, `evec_lor` |
| `fac` | `rot_mat`, `vv`, `vmap`, `evec_lor`, `bvdf_vol`, `bvdf_avg`, `npts_vol`, `Cprime_vol`, `Cprime_raw`, `bCprime_vol`, `bCprime_avg`, `npts_vol_cp`, `JE_tot`, `c_vol`, `c_avg`, `binc`, `binc_n`, `edges`, `edges_n`; plus `vdf_eq` (δf runs) and `cx_folds`, `cy_folds`, `cz_folds` (after `fpc_mrx_fold`) |
| `lmn` | as `fac` (without `vdf_eq` and the folds), plus `bvec` and `eigvals` |

`fpc_mrx_plasma_params` adds moment-based quantities to the same groups: densities, temperatures, `bmag` and the beta profiles in `meta`, and `bulk_ve`, `bulk_vi`, `jvec` and `jdotE` in `gse`, `fac` and `lmn`.
