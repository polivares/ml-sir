<!--
This file documents the *structure and meaning* of artifacts under `runs/`.
It is meant to be committed, even if the run outputs themselves are not.
-->

# `runs/` — run artifacts (what each file means)

The benchmark scripts in `scripts/` write **one folder per run** under `runs/` and store:
- The **exact configuration** used (CLI args),
- A **metrics table** (one row per method),
- **Logs** (so you can trace what happened),
- Optional **figures** and **plot data** (so plots can be rebuilt without rerunning).

This README explains what each artifact represents and how to use it.

Git note:
- By default, **run outputs are ignored** by git (`runs/*`), except this `runs/README.md`.

## Folder naming convention

By default, the run folders look like:

```
runs/
  exp0_YYYYMMDD_HHMMSS/
  exp1_YYYYMMDD_HHMMSS/
  exp2_YYYYMMDD_HHMMSS/
```

Where:
- `exp0/exp1/exp2` indicates which benchmark script generated the run.
- The timestamp is the local machine time at execution.

You can override the output folder with `--out-dir` in the scripts.

## Contents of a run folder

A typical run directory looks like:

```
runs/exp0_20251229_132423/
  config.json
  metrics.csv
  run.log
  predictions.npz        # only if --save-predictions
  predictions.json       # only if --save-predictions
  figures/               # only if --save-plots or --save-plot-data
    curves_comparison.png  # only if --save-plots
    error_curves.png       # only if --save-plots
    param_scatter.png      # only if --save-plots
    param_error_hist.png   # only if --save-plots
    metrics_comparison.png # only if --save-plots
    architectures.png      # only if --save-plots
    plot_data.npz          # only if --save-plot-data
    plot_data.json         # only if --save-plot-data
```

### `config.json`

Purpose: **reproducibility and provenance**.

- Contains the full CLI args used in the run plus `timestamp`.
- Use it to:
  - Track exactly what was executed (seed, limit, noise params, train-mode, etc.).
  - Re-run the experiment with the same settings.
  - Filter/group results after aggregation (`scripts/aggregate_runs.py --include-config`).

Notes:
- Not every key applies to every method (e.g., `epochs` only matters if you ran ML).

### `models/`

Purpose: **archive ML architectures per run** (weights + diagrams).

Contents:
- `manifest.json`: list of baselines/ML architectures used, plus file paths for each model artifact.
- `/<arch>/weights*`: TensorFlow checkpoint files for the trained model weights.
- `/<arch>/*.weights.h5`: Keras weight file (required filename suffix in Keras 3).
- `/<arch>/architecture.png`: diagram produced by `tf.keras.utils.plot_model`.
- `/<arch>/architecture.json`: model JSON config (from `model.to_json()`).
- `/<arch>/summary.txt`: `model.summary()` captured as text.

Notes:
- The `architecture.png` diagrams require Graphviz + `pydot`.
  If missing, the script logs a warning and still saves weights/metadata.

### `metrics.csv`

Purpose: **the main quantitative output** of each run.

- One row per evaluated method (e.g., `baseline_mse`, `baseline_mle_poisson`, `mlp`,
  `resmlp`, `tcn`, `transformer`, `mlp_hetero`, `mlp_mdn`, etc.).
  - Baseline rows appear only if the run used `--run-baseline` (or `--run-all`).
- Each row usually contains:
  - `method`: method identifier.
  - `scenario`: scenario string (e.g., `clean`, `train_clean_test_poisson`, etc.).
  - `n_test`: number of curves used for *that* evaluation.
  - Error metrics for each parameter (example keys):
    - `mae_beta`, `rmse_beta`, `r2_beta`
    - `mae_gamma`, `rmse_gamma`, `r2_gamma`
  - Timing summaries where applicable (example keys):
    - classical fitting: `time_p50`, `time_p90` (seconds per curve, based on the fitted subset)
    - neural inference: time per sample (from `predict_time_per_sample`)
    - training: `train_time_sec` (only for ML rows)

#### Timing fields (`time_p50`, `time_p90`, `train_time_sec`)

All timing values are in **seconds**.

- `time_p50`: the **median** (50th percentile) of the collected timing samples.
- `time_p90`: the **90th percentile** of the collected timing samples (gives a “slow-case” view).

What is being timed depends on the method:

- **Classical baselines** (`baseline_mse`, `baseline_mle_*`):
  - Only present if the run used `--run-baseline` (or `--run-all`).
  - The scripts time the **full per-curve fitting procedure** on the subset that was actually fitted
    (typically controlled by `--max-test`).
  - One timing sample ≈ “time to fit one curve”, including **all multi-start restarts**
    (`--n-starts`) for that curve.
  - Interpretation:
    - `time_p50` ≈ typical seconds per curve.
    - `time_p90` ≈ seconds per curve for “harder/slower” curves.

- **Neural models** (`mlp`, `mlp_branched`, `cnn1d`, `linear`, `resmlp`, `tcn`, `inception`,
  `attn_cnn`, `gru`, `lstm`, `conv_gru`, `transformer`, `mlp_hetero`, `mlp_mdn`):
  - `time_p50/time_p90` summarize a small set of **single-sample inference latencies** collected by
    `predict_time_per_sample()` (repeated `model.predict(X[i:i+1])` calls).
  - This measures **latency** (per-sample, one-by-one), not high-throughput batch prediction.
  - Notes:
    - GPU/TF may have warm-up/tracing overhead; results are best compared on the same machine.
    - If you care about throughput, you may want a separate benchmark that times `model.predict(X_batch)`.

- `train_time_sec` (ML only):
  - Total wall-clock training time for `model.fit(...)` in that run (includes early stopping).

Important:
- For classical baselines, the script may fit only a subset of the test set (`--max-test`), so:
  - `n_test` can be smaller than the full test size.
  - This is intentional to keep runtime manageable.
- For ML methods, evaluation is typically on the full test set (unless the script changes that).

### `predictions.npz` + `predictions.json` (optional)

Purpose: **store per-curve inputs and parameter predictions** so you can re-plot or audit later.

Stored arrays (NPZ keys):
- `times`: time grid used for the test series (after any window/downsample transforms).
- `i_true`: test I(t) series used for evaluation (shape: `n_test x T`).
- `i_obs`: only in Exp1; the observed noisy series `Y_obs` (shape: `n_test x T`).
- `y_true`: true `(beta, gamma)` for each test curve (shape: `n_test x 2`).
- `idx_test`: original dataset indices for the test split.
- `idx_fit`: indices (within the test set) that were actually fitted by the classical baseline.
  - Only present if the run used `--run-baseline` (or `--run-all`).
- `y_pred_<method>`: predicted `(beta, gamma)` per method.
  - For classical baselines, predictions are **NaN** where a curve was not fitted (outside `idx_fit`).

Metadata (JSON):
- `exp`, `scenario`/`train_mode`/`noise` (depending on experiment),
- `seed`, `rho`, `k` (when applicable),
- `s0`, `i0`, `r0`, `t0`, `dt` (needed to re-simulate I(t) from parameters).

How to use:
- To recompute **predicted I(t)** for a given method later, load `y_pred_<method>` and simulate
  with `src.sir.simulate.simulate_sir()` using the saved `s0/i0/r0/t0/dt` and the `times` grid.
- To generate plots directly from these files, run:
  `python -m src.visualization.visualize --predictions runs/<run>/predictions.npz --plots all`

### `run.log`

Purpose: **traceability** (progress + “what is running now”).

- Logs the main steps:
  - cache usage (load/save),
  - dataset shapes,
  - baseline fitting progress (every `--progress-every` curves),
  - ML training start/end + training time,
  - plot/plot-data saving.

Control:
- `--log-level {INFO,WARNING,DEBUG}` controls verbosity.
- `--progress-every` controls how often baseline progress is reported.
- `--log-file` overrides the path (default is `runs/<run>/run.log`).
- `--no-log-file` disables file logging.
- `--no-console-log` disables console logs (file-only).

Git note:
- This repository ignores `*.log` via `.gitignore`, so `run.log` is typically **not committed**.

### `figures/` (optional)

This folder is created when you pass `--save-plots` and/or `--save-plot-data`.

#### `*.png` (only if `--save-plots`)

Standard figure outputs (exact list may evolve):
- `curves_comparison.png`: example trajectories (true vs predictions and/or observations).
- `error_curves.png`: absolute error curves over time for the same examples.
- `param_scatter.png`: true vs predicted scatter plots for `(beta, gamma)` per method.
- `param_error_hist.png`: error distributions for `(beta, gamma)` per method.
- `metrics_comparison.png`: bar charts for key metrics across methods.
- `architectures.png`: text summary of which baseline(s) and ML architectures are present in the run artifacts.

#### `plot_data.npz` and `plot_data.json` (only if `--save-plot-data`)

Purpose: **rebuild figures later without rerunning the experiment**.

- `plot_data.npz` contains the numerical arrays needed for plotting.
- `plot_data.json` contains metadata and labels (curve names, method names, scenario metadata).

This is useful when:
- You want reproducible plots without re-running expensive fits/training,
- You want to regenerate figures with different styles/titles,
- You want to share plot inputs while keeping raw runs local.

## Top-level artifacts under `runs/`

### `runs/summary.csv`

Generated by `scripts/aggregate_runs.py`.

- A single table created by merging all `runs/*/metrics.csv`.
- If you use `--include-config`, the script adds config columns as `cfg_*` to make filtering/grouping easier.

### `runs/summary.log`

Optional log from `scripts/aggregate_runs.py` (default when file logging is enabled).

Git note:
- `*.log` is ignored by default, so this file is typically local-only.

## Experiment log (root)

The experiment scripts also update a repo-level log:
- `EXPERIMENTS.md` (at the repo root): tracks the **last run**, **final selection**, and a
  chronological log of runs with summary metrics.

## Rebuilding figures from saved plot data

If a run was executed with `--save-plot-data`, you can regenerate the standard set of figures without
re-running the experiment:

```bash
python scripts/rebuild_plots.py --plot-data runs/exp0_YYYYMMDD_HHMMSS/figures
```

Outputs: saved back into that folder by default (or override with `--out-dir`).

## Practical workflow (recommended)

1) Run an experiment (Exp0/Exp1/Exp2) with:
   - `--save-plot-data` (reproducible plots)
   - `--save-plots` (quick inspection)
2) Inspect:
   - `runs/<run>/metrics.csv`
   - `runs/<run>/figures/*.png`
   - `runs/<run>/run.log` (debug / progress)
3) Aggregate all runs:
   - `python scripts/aggregate_runs.py --include-config`
4) Rebuild figures anytime from the saved plot data:
   - `python scripts/rebuild_plots.py --plot-data runs/<run>/figures`

## Tips / troubleshooting

- If you see the baseline “stuck”, check `run.log` to confirm it is progressing; consider lowering:
  - `--max-test` and/or `--n-starts`.
- If you want more detail while debugging:
  - use `--log-level DEBUG`.
- If ML training is slow/unstable:
  - reduce `--batch-size`, try fewer `--epochs`, or inspect GPU usage with `nvidia-smi -l 1`.
