# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`fipcore` (FIeld Particle COrrelation for Reconnection Events) applies field-particle correlation (FPC) analysis to MMS spacecraft data around magnetic reconnection X-line crossings. It is a Python reimplementation of an IDL pipeline originally written by Arya; the IDL code is kept in `fpc_mrx_idl_archive/` for reference only. The Python code is not required to stay compatible with it. The primary event studied is 2015-12-09 05:03:55–05:03:59 (burst mode, electrons).

## Environment

- Use the `phy` conda env (`/opt/anaconda3/envs/phy/bin/python`). The base Anaconda Python does not have pyspedas.
- `fipcore` is installed editable into `phy`. If the repo moves, reinstall it: `pip install -e . --no-deps`. Otherwise, `import fipcore` works only from the repo root.
- Dependencies are listed in `pyproject.toml`: pyspedas, numpy, scipy, matplotlib and h5py. The README's `requirements.txt` does not exist.
- There is no test suite, linter or build step. Validation is done in the notebooks under `validation_checks/` (FAC/LMN rotations, J·E, bin width, sign checks). Analysis and figure notebooks and scripts live in `scripts/`; `Publication_plots.ipynb` is the main driver.
- The MMS CDF cache is `~/Data_Speedas/mms`, set by `export SPEDAS_DATA_DIR=...` in `~/.zshrc`. Non-interactive shells (like Claude's Bash tool) may not load that variable, and pyspedas then silently downloads to `./pydata` instead. Set `SPEDAS_DATA_DIR=/Users/rejohn/Data_Speedas` explicitly when running the pipeline from a script. `no_update=True` reads from the cache and downloads only missing files.

## Architecture

The code is organized as a pipeline that writes to an HDF5 file, with separate plotting functions that read that file. Computation and plotting are decoupled.

1. **Master functions** in `fipcore/analysis/fpc_mrx_main.py`:
   - `fpc_mrx_main` loads FPI, EDP and FGM data and runs the whole chain: velocity bins, reconnection frame, FAC/LMN rotation, VDF binning, C′, FPC and J·E. It writes the results to HDF5.
   - `fpc_mrx_fold` reads that file and adds folded FPCs (FAC only).
   - `fpc_mrx_plasma_params` computes ancillary parameters (beta, J, J·E).
2. **Stage modules** in `fipcore/analysis/*_proc.py` each own one step:
   - `vel_proc`: energy to velocity bins, including spacecraft-potential and interleave correction, frame shifts, and the `vmap` bin maps
   - `vdf_proc`: VDF quality flags, volume weighting, binning, and the equilibrium tri-Maxwellian used for δf
   - `evec_proc`: Lorentz transform of E and its rotation
   - `fpc_proc`: C′ = q·v·E·f, binning, C = −(v²/2)∂C′/∂v, and J·E
3. **Plotting** in `fipcore/plotting/`. Functions such as `imagecont_vdf_fpc(trange, species, bin_width_frac, time_index, ...)` rebuild the HDF5 filename from their arguments and read from `data/`. They never recompute anything.

### Conventions that span files

- **tplot namespace:** functions pass data between each other as pyspedas tplot variable *names* (strings such as `'bvec_gse_dwn'`, `'evec_lor_fac'`), not arrays. `fpc_mrx_main` renames the raw MMS variables to short names (`nrgy`, `scpot`, `vdf_raw`, `bvec_gse`, and so on) that the downstream functions expect. tplot variables do not overwrite reliably, so every master function starts with `del_data('*')`. Keep this call in any new master function.
- **Cadence:** all quantities are put on the DES 30 ms time base. `helper_utils.downsample_cad` handles faster fields and `upsample_cad` handles the ion moments.
- **Frames:** FAC is built from B with `other_dim='Xgse'`. LMN comes from `coord_utils.lmn_matrix_make`. The reconnection-frame shift uses the ion bulk velocity, and its sign is flipped at the X-line index found from the B_L reversal (`helper_utils.bl_recx_idx`, `vel_proc.vi_bl_flip`).
- **Two binning flavours:** `*_vol` is volume-weighted (sum of f·dv³) and `*_avg` is a plain average. Both are saved.
- **Species strings:** the code uses `'e'` or `'ion'` (see `CHARGE` and `MASS_ENERGY` dicts). `fpc_mrx_main` hardcodes `des_*` variable names, so in practice it is electron-only.
- **Array shapes:**
  - raw velocities `vv`: `(3, 16384, ntime)`
  - binned C: `(3, nbin, nbin, nbin, ntime)`
  - 2D slices come from `helper_utils.slice3d_to_2d`, which returns `(xy, yz, xz)`
- **HDF5 output:**
  - Path: `data/mms{probe}_{data_rate}_{f|df}_{species}_{bin_width_frac:.2f}_{YYYYMMDD}_{hhmmss}_{hhmmss}.h5`, built by `io_utils.mms_name_make`. `df` means `subtract_f0=True`, i.e. δf = f − f₀.
  - Groups: `meta`, `gse`, `fac`, `lmn`.
  - `io_utils.h5sav` opens files in append mode and overwrites individual datasets. It also attaches a `units` attribute from `UNITS_MAP`. To add units for a new key, add it there.
- **Paths:** `data/` and the plot output directories are resolved relative to the package location (`os.path.dirname(...__file__)` three levels up), not relative to the cwd.
- **Git-ignored outputs:** `data/`, `*.h5`, `*.eps`, `*.png` and `*.tar` are not committed.
