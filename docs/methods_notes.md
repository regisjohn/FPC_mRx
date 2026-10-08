# Methods notes: field–particle correlation analysis of MMS electron energization

These notes are source material for writing the methodology section. They describe exactly what the analysis code (`fipcore`, a Python package) does, applied to the MMS magnetosheath reconnection event of 2015-12-09. Equations are the ones the code implements. Numbers quoted for the event come from the processed MMS data. Appendix A maps each step to the code, for traceability only.

---

## 1. Overview

The analysis computes the field–particle correlation (FPC) of electrons with the electric field, in velocity space, at every 30 ms burst-mode sample. Each of the four MMS spacecraft is processed independently. The steps are:

1. Load burst-mode L2 data and put all quantities on the 30 ms electron (FPI-DES) time base.
2. Convert each measured distribution bin into a velocity vector, correcting energies for the spacecraft potential.
3. Clean the distributions and compute the phase-space volume of each instrument bin.
4. Transform into the reconnection (X-line) frame and rotate into field-aligned (FAC) and boundary-normal (LMN) coordinates.
5. Bin the measurements onto a Cartesian velocity grid normalized by the electron thermal speed.
6. Compute the correlation $C'_{E_j} = q v_j E'_j f$, its velocity integral (the energy transfer rate $J_jE'_j$), and the standard FPC $C_{E_j}$.
7. Optionally, repeat with a bi-Maxwellian background subtracted ($\delta f = f - f_0$).

Ancillary moment-based quantities (current density, $\mathbf{J}\cdot\mathbf{E}'$, plasma beta) are computed on the same time base (§10).

## 2. Event and processing parameters

| Item | Value |
|---|---|
| Event interval | 2015-12-09 05:03:55–05:03:59 UT (4 s) |
| Spacecraft | MMS1–4, each processed independently |
| Species | Electrons |
| Data | Burst mode, Level 2 |
| Time base | FPI-DES burst cadence, 30 ms (133–134 samples per spacecraft) |
| Velocity grid extent | ±3.5 ⟨v_th,e⟩ in each component |
| Bin width Δ | 0.25 ⟨v_th,e⟩ (28 × 28 × 28 bins). Also run at 0.10, 0.20, 0.30, 0.40 and 0.50 to test sensitivity to bin width. |
| Azimuth angles | Per-sweep values (not time-averaged) |
| Background subtraction | Analyses run with the full distribution $f$ and with $\delta f = f - f_0$ |
| f₀ drift velocity | Zero in the reconnection frame (§9) |

Per-spacecraft values:

| | MMS1 | MMS2 | MMS3 | MMS4 |
|---|---|---|---|---|
| ⟨v_th,e⟩ (km/s) | 4442 | 4420 | 4438 | 4496 |
| Physical bin width at Δ = 0.25 (km/s) | 1110 | 1105 | 1110 | 1124 |
| Time of peak energization (UT) | 05:03:56.872 | 05:03:56.745 | 05:03:56.817 | 05:03:56.888 |

## 3. Data and common time base

The following L2 burst-mode products are used:
- **FPI-DES:** the 3D electron distribution $f$ and its uncertainty $\sigma_f$, the per-distribution quality flag, the energy table and azimuth angles for each sweep, and the moments $n_e$, $T_{e\parallel}$, $T_{e\perp}$, $\mathbf{V}_e$ (GSE).
- **FPI-DIS:** the ion bulk velocity $\mathbf{V}_i$ (GSE).
- **EDP:** the electric field $\mathbf{E}$ (GSE) and the spacecraft potential $V_{sc}$.
- **FGM:** the magnetic field $\mathbf{B}$ (GSE).

All quantities are placed on the 30 ms DES time base:
- **$\mathbf{E}$, $\mathbf{B}$ and $V_{sc}$** are measured at higher cadence and are boxcar-averaged over windows equal to the DES sampling interval. The averaged samples are within 12.4 ms (less than half a DES interval) of the DES timestamps.
- **$\mathbf{V}_i$** (150 ms DIS cadence) is linearly interpolated onto the DES timestamps.

## 4. Velocity of each measurement

**Energy and spacecraft potential.** Each FPI-DES distribution has 32 azimuth (φ) × 16 polar (θ) × 32 energy bins, 16384 in total. Channel energies are corrected for the spacecraft potential:
$$
\mathcal{E}_{corr} = \mathcal{E}_{table} + q V_{sc},
$$
so for electrons ($q = -e$), $\mathcal{E}_{corr} = \mathcal{E}_{table} - eV_{sc}$. The 30 ms-averaged $V_{sc}$ is used at each time. For this event $V_{sc}$ = 3.3–4.1 V, well below the lowest DES channel (10.96–12.41 eV), so no channel falls below zero after the correction. Speeds are non-relativistic:
$$
v = \sqrt{2\mathcal{E}_{corr}/m_e}.
$$
The energy table interleaves between two tables on alternate sweeps during this event, so each sweep's own table is used and the velocity grid is built separately for every time sample.

**Velocity vectors.** Each bin's velocity is
$$
\mathbf{v} = -v\,(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta),
$$
- The minus sign converts the instrument look direction into the direction of particle travel.
- θ takes the 16 FPI pixel zenith angles, $\theta_k = 5.625^\circ + 11.25^\circ k$.
- φ takes the per-sweep azimuth angles.
- The angles are given in the despun spacecraft frame, and the resulting velocities are used directly with the GSE field and moment data, with no further coordinate rotation. For this event the despun and GSE frames differ by ≈2.5–3.5°.

**Phase-space volume.** The velocity-space volume of each instrument bin is
$$
\Delta^3 v = \tfrac13\left(v_{hi}^3 - v_{lo}^3\right)\left(\cos\theta_{lo}-\cos\theta_{hi}\right)\Delta\phi,
$$
- $v_{lo}$ and $v_{hi}$ are midpoints between adjacent channel speeds, with the outermost edges mirrored.
- θ edges divide $[0,\pi]$ into 16 equal parts, and $\Delta\phi = 2\pi/32$.
- Volumes are computed for every time sample, because the corrected energies change with $V_{sc}$.

## 5. Distribution processing

1. **Quality flags.** A distribution with any non-zero L2 quality flag is replaced by linear interpolation in time from neighbouring good distributions, bin by bin. The flag bits are: manually flagged interval, likely overcounting or saturation, and compression-pipeline error. No distribution is flagged during this event.
2. **Counts and empty bins.** Counts are reconstructed assuming Poisson statistics, $N = \mathrm{round}[(f/\sigma_f)^2]$. Bins with $N \le 0$ or $f \le 0$ are set to $f = 0$: a bin with no counts is treated as a measurement of no particles, not as missing data.
3. **Units.** $f$ is converted to SI units (s³ m⁻⁶).
4. **Two forms are carried forward:** the phase-space density $f$, and the number density in each instrument bin, $f\,\Delta^3 v$ (m⁻³).

## 6. Reference frames

**X-line time and frame velocity.**
- **X-line time.** The X-line crossing time $t_X$ is the time of maximum $|dB_L/dt|$ within the event interval, with $B_L$ the L component of the 30 ms-averaged field.
- **Frame velocity.** The reconnection frame moves with the ion bulk velocity at that time, $\mathbf{U} = \mathbf{V}_i(t_X)$: a single constant velocity for the whole interval.
- **Velocities.** Measurement velocities are Galilean-shifted, $\mathbf{v}' = \mathbf{v} - \mathbf{U}$. The value of $f$ is unchanged.
- **Electric field.** The field is transformed into the moving frame at each time sample:
$$
\mathbf{E}' = \mathbf{E} + \mathbf{U}\times\mathbf{B}.
$$

**Field-aligned coordinates (FAC).** A separate FAC basis is built at each time sample from the 30 ms-averaged $\mathbf{B}$, with $\hat{\mathbf{x}}_{GSE}$ as the reference direction:
$$
\hat{\mathbf{e}}_\parallel = \hat{\mathbf{b}},\qquad
\hat{\mathbf{e}}_{\perp 2} = \frac{\hat{\mathbf{b}}\times\hat{\mathbf{x}}_{GSE}}{|\hat{\mathbf{b}}\times\hat{\mathbf{x}}_{GSE}|},\qquad
\hat{\mathbf{e}}_{\perp 1} = \hat{\mathbf{e}}_{\perp 2}\times\hat{\mathbf{b}}.
$$
Velocities $\mathbf{v}'$ and the field $\mathbf{E}'$ are rotated into this basis sample by sample.

**Boundary-normal coordinates (LMN).**
- **Method.** One LMN system per spacecraft comes from minimum variance analysis (MVA) of the 30 ms-averaged $\mathbf{B}$ over the full 4 s interval. L, M and N are the maximum-, intermediate- and minimum-variance directions.
- **Sign convention.** $N$ is chosen with a positive GSE x-component and $L$ with a positive GSE z-component, and $M$ completes a right-handed system.
- **Eigenvalues.** For MMS1 they are 600.7, 37.4 and 3.2 nT², giving $\lambda_{max}/\lambda_{int} = 16.0$ and $\lambda_{int}/\lambda_{min} = 11.7$.
- **What is rotated.** Velocities, $\mathbf{E}'$ and $\mathbf{B}$ are rotated into LMN with this fixed matrix.

LMN vectors in GSE:

| | L | M | N |
|---|---|---|---|
| MMS1 | (−0.283, −0.127, 0.951) | (0.285, −0.957, −0.043) | (0.916, 0.259, 0.307) |
| MMS2 | (−0.291, −0.118, 0.949) | (0.255, −0.966, −0.042) | (0.922, 0.230, 0.312) |
| MMS3 | (−0.285, −0.132, 0.949) | (0.266, −0.962, −0.054) | (0.921, 0.237, 0.309) |
| MMS4 | (−0.285, −0.125, 0.950) | (0.264, −0.963, −0.047) | (0.921, 0.237, 0.308) |

## 7. Velocity-space grid and binning

**Normalization.**
- **Thermal speed.** Velocities are normalized by the mean electron thermal speed over the event interval:
$$
v_{th} = \sqrt{2T/m_e},\qquad T = \tfrac13\left(T_\parallel + 2T_\perp\right),
$$
  where $T_\parallel$ and $T_\perp$ are from the DES moments, averaged over the interval to give ⟨v_th,e⟩ (§2).
- **Single scale.** One ⟨v_th,e⟩ is used for all times, so the physical bin width is fixed at Δ·⟨v_th,e⟩ (≈1110 km/s for Δ = 0.25).

**Grid.**
- **Shape.** A Cartesian grid, identical in all three components and for both FAC and LMN, with $n = \lceil 7/\Delta \rceil$ bins of width exactly Δ, centred on zero.
- **Extent.** ±3.5 ⟨v_th,e⟩ when 7/Δ is a whole number, which covers all widths used except 0.30 and 0.40. For those two, the grid extends slightly, to ±3.6.
- **Where v = 0 falls.** With an even number of bins (28 for Δ = 0.25), $v_j = 0$ is a bin edge. With an odd number, it is the centre of the middle bin.
- **Excluded measurements.** A measurement is excluded if any of its three components lies outside the grid.
- **Check of the cell assignment.** The assignment was verified visually by plotting measurements in 2D velocity planes together with the outlines of the grid cells they were assigned to. This was done in the (v∥, v⊥1) and (v∥, v⊥2) planes in FAC and the (v_N, v_L) and (v_N, v_M) planes in LMN, for one time sample.

**Binning.** At each time sample, every measurement is assigned to a grid cell. Two binned forms are computed:
- **Volume-weighted (used for all results shown):** $\sum_{i \in \text{cell}} f_i\,\Delta^3 v_i$, the number density in the cell (m⁻³).
- **Averaged:** the mean of $f$ over the measurements in the cell (s³ m⁻⁶).

## 8. Field–particle correlation

**Correlation in each instrument bin.** For each velocity component $j$ (no summation over $j$), at each measurement and each 30 ms sample:
$$
C'_{E_j}(\mathbf{v}, t) = q\,v_j\,E'_j(t)\,f(\mathbf{v}, t),\qquad q = -e.
$$
- Velocities and $\mathbf{E}'$ are in the same coordinate system (FAC or LMN) in the reconnection frame.
- Using the volume-weighted form $f\Delta^3 v$ gives $C'_{E_j}$ in W m⁻³ per instrument bin.
- $C'_{E_j}$ is binned onto the velocity grid in the same way as $f$.

**Energy transfer rate.** Summing the binned volume-weighted $C'_{E_j}$ over all grid cells gives the rate of energy transfer to electrons by each field component:
$$
J_j E'_j(t) = \sum_{\text{cells}} C'_{E_j}.
$$
This includes only electrons within the ±3.5 ⟨v_th,e⟩ grid.

**Standard FPC.** The standard correlation, $C_{E_j} = -q\frac{v_j^2}{2}\frac{\partial f}{\partial v_j}E'_j$, is computed from the binned $C'_{E_j}$ using the identity
$$
C_{E_j} = -\frac{v_j}{2}\frac{\partial C'_{E_j}}{\partial v_j} + \frac{1}{2}C'_{E_j},
$$
which follows from substituting $C'_{E_j} = q v_j f E'_j$.
- **Derivative.** Evaluated with second-order central differences in the interior of the grid and first-order one-sided differences at its edges.
- **Velocity.** $v_j$ is the physical velocity at the bin centre.

**Correlation interval.** No averaging over a correlation interval is applied. The correlation is computed independently at each 30 ms sample, because the reconnection process evolves on these timescales.

**Reduced forms shown in figures.**
- **2D velocity-space signatures.** The 3D binned $C_{E_j}$ and $f\Delta^3v$ are summed over the third velocity component.
- **1D signatures.** They are summed over the other two components and plotted against normalized velocity.
- **Peak time.** Each spacecraft is shown at the time of its maximum energy transfer rate, the time when $\sum_j J_jE'_j$ peaks (§2).

**Folded FPC (FAC).** To show net energization as a function of speed along each direction, each 2D signature $C_{E_j}$ is folded about $v_j = 0$ along its own velocity axis:
$$
C_{E_j}(|v_j|) = C_{E_j}(+v_j) + C_{E_j}(-v_j).
$$
- $C_{E_{\perp1}}$ is folded in the (⊥1, ⊥2) and (⊥1, ∥) planes.
- $C_{E_{\perp2}}$ in the (⊥1, ⊥2) and (∥, ⊥2) planes.
- $C_{E_\parallel}$ in the (⊥1, ∥) and (∥, ⊥2) planes.
- With an odd number of bins, the bin centred on $v_j = 0$ is not folded.

## 9. Background subtraction (δf)

To isolate the non-thermal part of the distribution, a gyrotropic bi-Maxwellian (a 3D Maxwellian with equal perpendicular temperatures, $T_{\perp1} = T_{\perp2} = T_\perp$)
$$
f_0 = \frac{n}{\pi^{3/2}\,w_\perp^2\,w_\parallel}\exp\!\left[-\frac{v_{\perp1}^2+v_{\perp2}^2}{w_\perp^2}-\frac{v_\parallel^2}{w_\parallel^2}\right],\qquad w_{\parallel,\perp}=\sqrt{2T_{\parallel,\perp}/m_e},
$$
is subtracted from the measured distribution, giving $\delta f = f - f_0$.
- **Inputs.** $n$, $T_\parallel$ and $T_\perp$ are the DES moments at each time sample. $f_0$ is evaluated at each measurement's FAC velocity in the reconnection frame.
- **Drift.** $f_0$ has no drift in the reconnection frame: it is centred on the ion velocity at the X-line $\mathbf{U}$, not on the local electron bulk velocity. Electron flows relative to the X-line frame are therefore retained in δf.
- **Empty bins.** In bins with no counts, δf = −f₀.
- **Scope.** All binned quantities, correlations and energy transfer rates of §7–8 are then recomputed with δf in place of f.

## 10. Ancillary plasma parameters

On the same 30 ms time base:
- **Current density** from moments, $\mathbf{J} = n_e e(\mathbf{V}_i - \mathbf{V}_e)$, assuming quasi-neutrality ($n_i = n_e$). The ion velocity is interpolated to the DES cadence.
- **Energy conversion rate** $\mathbf{J}\cdot\mathbf{E}'$ and its components $J_jE'_j$, in GSE, FAC and LMN, using $\mathbf{E}'$ from §6. This provides an independent, moment-based comparison for the FPC energy transfer rate of §8.
- **Plasma beta** for electrons and ions, $\beta = nk_BT/(B^2/2\mu_0)$. It is computed separately with $T_\parallel$, with $T_\perp$, and with the scalar $T = (T_\parallel + 2T_\perp)/3$.

## 11. Software and data

- **Data:** MMS burst-mode Level 2, retrieved from the MMS Science Data Center:
  - FPI DES and DIS distributions and moments, v3.3.0
  - EDP DCE, v2.2.0
  - EDP spacecraft potential, v2.4.0
  - FGM, v4.22.0
- **Software:** analysis code `fipcore` v1.1 (Python 3.13). It uses PySPEDAS v2.0.5 for data loading, time averaging and interpolation, FAC construction and minimum variance analysis, plus NumPy 2.3, SciPy 1.16 and h5py 3.14.

---

## Appendix A. Code map (for traceability; not for the paper)

| Step | Function (module in `fipcore/`) |
|---|---|
| Whole pipeline | `fpc_mrx_main` ([analysis/fpc_mrx_main.py](../fipcore/analysis/fpc_mrx_main.py)) |
| Folding | `fpc_mrx_fold` (same file) |
| Ancillary parameters | `fpc_mrx_plasma_params` (same file) |
| Time base | `downsample_cad`, `upsample_cad` ([utils/helper_utils.py](../fipcore/utils/helper_utils.py)) |
| Energy → speed, s/c potential | `nrgy_to_vbin` ([analysis/vel_proc.py](../fipcore/analysis/vel_proc.py)) |
| Velocity vectors, bin volumes | `vbin_to_vv`, `compute_vbin_vol` (vel_proc.py) |
| Distribution cleaning | `process_vdf` ([analysis/vdf_proc.py](../fipcore/analysis/vdf_proc.py)) |
| X-line time, frame velocity | `bl_recx_idx` (helper_utils.py), `vi_bl_flip`, `vv_to_recx` (vel_proc.py) |
| E′ and its rotations | `evec_to_recx`, `evec_to_fac`, `evec_to_lmn` ([analysis/evec_proc.py](../fipcore/analysis/evec_proc.py)) |
| LMN (MVA) | `lmn_matrix_make` ([utils/coord_utils.py](../fipcore/utils/coord_utils.py)) |
| Thermal speed, grid | `compute_vth` (helper_utils.py), `vv_to_vmap` (vel_proc.py) |
| Grid-assignment check (diagnostic plot) | `plt_vmap_check` ([plotting/plt_vmap_check.py](../fipcore/plotting/plt_vmap_check.py)), used in `scripts/vdf_generation.ipynb` |
| Binning | `bin_vdf_3d` (vdf_proc.py), `bin_cprime_3d` ([analysis/fpc_proc.py](../fipcore/analysis/fpc_proc.py)) |
| C′, J·E′, C | `compute_cprime`, `jEtot_cprime`, `compute_fpc` (fpc_proc.py) |
| f₀ | `equilibrium_vdf` (vdf_proc.py) |
| J, J·E′ from moments | `compute_jdotE` ([analysis/jdotE_proc.py](../fipcore/analysis/jdotE_proc.py)) |

Results are saved per spacecraft as HDF5 files with groups `meta`, `gse`, `fac` and `lmn`. The quantities used for figures are `c_vol` (the binned $C_{E_j}$), `bvdf_vol` (binned $f\Delta^3v$ or $\delta f\Delta^3v$), `JE_tot` ($J_jE'_j$), the folds `cx_folds`, `cy_folds` and `cz_folds`, and `lmn/eigvals`.
