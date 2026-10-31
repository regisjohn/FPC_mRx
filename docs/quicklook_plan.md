# Quick-look and event screening plan

Status: plan, nothing implemented yet. Update this file as phases are done.

## Goal

Screen many MMS intervals quickly for current sheets, reconnection signatures and J·E, before running the full FPC pipeline (`fpc_mrx_main`) on the ones worth it. The quick look reads only fields and plasma moments, never the 3D distributions, which make the full pipeline slow.

## Why two stages

LMN from MVA needs a window around one current-sheet crossing. That window can be found from B alone, without LMN or J·E: by Ampère's law a current sheet is where B rotates, which shows in any frame (GSE included) as sharp changes in the components and often a dip in |B|. |J| from the curlometer peaks at the sheet and gives its centre. B and J find current sheets; reconnection needs the stage 2 signatures (B_L reversal with Hall B_M and B_N, ion and electron jets along L, J·E′ > 0).

## Choosing the MVA window

- Wilder et al. (2017), PRL 118, 265101: MVA of B over 05:03:56.5–57.2, the interval of the rapid B_L reversal, which coincides with a bipolar Hall B_M, positive B_N, a dip in |B| and an out-of-plane J_M. Method cited: Sonnerup & Scheible (1998).
- Sonnerup & Scheible (1998), ISSI SR-001: the window covers one current-sheet crossing, from the B plateau on one side to the plateau on the other, roughly centred on B_L = 0, leaving out the steady field and fluctuations outside the sheet.
- Quality: λ_int/λ_min ≳ 10 for a well-determined N; large λ_max/λ_int for a well-determined L. Stability: N should not swing much when the window edges move by a sample or two. This only checks a window chosen from the plateaus; it does not choose the window.
- Alternatives when MVA is poorly determined (cited from memory, check before citing): Shi et al. (2005), MDD, N from the four-probe gradient of B; Denton et al. (2018, JGR), comparison of MVA, MDD and hybrids on MMS events; Genestreti et al. (2018, JGR), effect of LMN errors on derived quantities.

## Phases

### Phase 1: stage 2 (one window, reconnection test)

Build first, because published intervals already give the window.

- Input: `trange`, `probe`, `data_rate`, `lmn_trange` (MVA window).
- Data: FGM, FPI moments (DES and DIS), EDP. No distributions.
- MVA over `lmn_trange` with `coord_utils.lmn_matrix_make`; report λ_max/λ_int and λ_int/λ_min.
- Panels: B in LMN and |B|; v_i and v_e in LMN; n_e and n_i; E in LMN; J in LMN (curlometer and moments); J·E′ with E′ = E + v_e × B; electron and ion energy spectrograms. Dotted lines mark `lmn_trange`.
- Returns a dict of summary numbers: eigenvalue ratios, B_L reversal (yes/no), peak |v_iL| and |v_eL|, peak J·E′.
- Validate on the published intervals: the plots should reproduce the paper figures.

### Phase 2: stage 1 (find current sheets, no LMN)

- Input: a long `trange`.
- Data: FGM for MMS1–4.
- Panels: |B|, B in GSE, |J| from the curlometer, rotation rate of the B direction (deg/s).
- Flags candidates where |J| and the rotation rate exceed thresholds, proposes a window from the plateau before the rotation to the plateau after, and returns the list.
- Validate: it should find the published current sheets inside longer windows.

### Phase 3: machine learning

- **3a. Rule baseline.** Thresholds on the stage 2 summary numbers. Any model must beat it.
- **3b. Unsupervised candidates.** Isolation Forest on windowed stage 1 features (|B|, |J|, rotation rate). Needs no labels.
- **3c. Supervised ranking.**
  - Features: the stage 2 summary numbers per candidate.
  - Labels: positives from published events; negatives from current sheets without reconnection signatures and quiet intervals; later, your own stage 2 verdicts and MMS Scientist-in-the-Loop selections.
  - Model: gradient-boosted trees (`HistGradientBoostingClassifier`), SHAP for which features drive each decision.
  - Validation: split by event or date (`GroupKFold`), never by random sample, since neighbouring windows of one event are nearly identical. Report precision and recall against the 3a baseline.
- Effort: 3a and 3b about a day each; 3c about 100–200 lines. The main cost is collecting labelled events.
- Learning path (scikit-learn user guide): Isolation Forest → `HistGradientBoostingClassifier` → cross-validation with `GroupKFold` → precision and recall.
- Related MMS work to compare against (from memory, check before citing): Argall et al. (2020), automated burst selection; Breuillard et al. (2020) and Olshevsky et al. (2021), plasma region classification; Lenouvel et al. (2021), ML identification of EDRs at the magnetopause.

## Code layout

- `fipcore/analysis/quicklook.py`: loading, MVA, curlometer, J·E′, summary numbers, candidate detection.
- `fipcore/plotting/plt_quicklook.py`: stage 1 and stage 2 figures.
- `fipcore/ml/`: features, baseline, Isolation Forest, classifier.
- Every master function starts with `del_data('*')`. Reuse `coord_utils.lmn_matrix_make` and `helper_utils.downsample_cad`.
- The quick look returns dicts and does not write HDF5, unlike `fpc_mrx_main`.
- scikit-learn and shap as optional dependencies.

## Open decisions

1. Stage 1: automatic flags, or plot only and pick by eye?
2. Stage 1 data: survey or burst?
3. Should the quick look ever write HDF5?
4. Published intervals to use for validation: list them here with references.
