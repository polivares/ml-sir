# Epidemic Spreading and Machine Learning (ml-sir)

This repository benchmarks **classical parameter fitting** vs **Machine Learning / Deep Learning**
approaches to infer SIR parameters from time series.

The current benchmark focus is **parameter inference from the infected trajectory**:
given **I(t)**, predict **(beta, gamma)**.

## Current Benchmark Scope

- **Data**: simulated SIR trajectories generated with `summer`.
- **Targets**: SIR parameters `(beta, gamma)`.
- **Classical baselines**:
  - Per-curve optimization on clean data (MSE + L-BFGS-B, multi-start).
  - Per-curve maximum likelihood under observation noise (Poisson / NegBin, multi-start).
- **ML/DL baselines**:
  - MLP, branched-MLP, CNN1D, linear, ResMLP, TCN, Inception-style CNN,
    attention CNN, GRU/LSTM, Conv-GRU, Transformer, heteroscedastic and MDN heads.
- **Realistic observation mechanisms** (incremental): Poisson, Negative Binomial, windowing,
  downsampling, variable initial conditions, and observation-intensity sweeps.

## Repository Structure (relevant)

```
├── src/sir/                  # Benchmark modules (simulation, datasets, noise, baselines, ML)
├── src/visualization/        # Plotting helpers used by the scripts
├── scripts/                  # Reproducible benchmark entrypoints + aggregation
├── docs/                     # Sphinx documentation (optional)
└── runs/README.md            # Describes run folder layout (generated locally)
```

This branch is code-only. Datasets and run outputs are generated locally and are not stored in
version control.

## Setup

`requirements.txt` is a **conda explicit spec** (not a pip requirements file). Create an environment with:

```bash
conda create -n ml-sir --file requirements.txt
conda activate ml-sir
pip install -e .
```

If you prefer a lighter CPU-only environment, install the minimal dependencies manually
(`numpy`, `scipy`, `summer`, `matplotlib`, and optionally `tensorflow` for ML experiments).

GPU note (TensorFlow):
- The benchmark enables **GPU memory growth** so TensorFlow allocates VRAM gradually instead of pre-allocating most of it.
- You can monitor VRAM usage during training with `nvidia-smi -l 1`.

## Data

Default dataset path: `data/raw/simulated/SIR/sir.pkl` (not included; can be very large).

Expected `sir.pkl` schema (as used by `src/sir/datasets.py`):
- list of `(outputs, times, params)` tuples
- `outputs` is an array shaped `(T, 3)` with columns `[S, I, R]`
- `params` is `[beta, gamma]`

## Experiments (scripts)

All scripts write a run folder under `runs/` containing:
- `config.json` (CLI args + timestamp)
- `metrics.csv` (one row per method evaluated)
- `run.log` (console + file logs for traceability)
- `models/` and `figures/` (optional; generated when enabled)

Common flags:
- `--seed`: reproducibility
- `--limit`: subsample the dataset (useful for quick runs)
- `--max-test`: cap number of test curves for classical fitting (runtime control; only if `--run-baseline`)
- `--normalize {max,population}`: consistent scaling for ML models
- `--run-all`: run every ML architecture available in the experiment
- `--run-baseline`: run the classical baseline for the experiment
- `--run-*`: enable ML architectures (`mlp`, `mlp_branched`, `cnn1d`, `linear`, `resmlp`,
  `tcn`, `inception`, `attn_cnn`, `gru`, `lstm`, `conv_gru`, `transformer`, `mlp_hetero`, `mlp_mdn`)
- `--cache-dir` / `--no-cache`: caching for derived arrays/splits (default `data/processed/sir`)
- `--progress-every`: how often to log baseline progress
- `--log-level`: logging verbosity (`INFO` by default)
- `--log-file`: override log path (default: `runs/<run>/run.log`)
- `--no-log-file`: disable log file output
- `--no-console-log`: disable console logging
- `--save-predictions`: save per-curve `I(t)` plus `(beta,gamma)` predictions to `predictions.npz/json`
- `--pred-dir`: override where prediction artifacts are written (default: run folder)
- `--exp-log`: optional experiment log path (created locally if used)

Note: the classical baseline only runs when you pass `--run-baseline` (or `--run-all`).
Note: model diagrams use `tf.keras.utils.plot_model` and require `pydot` + Graphviz; if missing,
the scripts log a warning and continue saving weights/metadata.

### Exp0: Clean-data benchmark

File: `scripts/exp0_run.py`

What it does:
- Builds a train/val/test split from clean I(t).
- Fits the classical baseline (`baseline_mse`) per curve **when `--run-baseline` (or `--run-all`) is set**.
- Optionally trains ML models and evaluates on the full test set.

Run:
```bash
python scripts/exp0_run.py --limit 5000 --max-test 200 --n-starts 5 --run-baseline --run-mlp --normalize max
```

Optional plots (saved under `runs/<run>/figures/`):
```bash
python scripts/exp0_run.py --limit 5000 --max-test 200 --n-starts 5 --run-baseline --run-mlp --normalize max --save-plots --n-plot 9
```

Optional plot data (saved under `runs/<run>/figures/`):
```bash
python scripts/exp0_run.py --limit 5000 --max-test 200 --n-starts 5 --run-baseline --run-mlp --normalize max --save-plot-data --n-plot 9
```

### Exp1: Observation-noise benchmark (Poisson / NegBin)

File: `scripts/exp1_noise.py`

What it does:
- Adds Poisson or Negative Binomial noise to I(t) (treated as observed counts).
- Fits the corresponding classical MLE baseline per curve **when `--run-baseline` (or `--run-all`) is set**.
- Optionally trains ML models under different training modes:
  - `clean`: train on clean I(t), test on noisy observations
  - `noisy`: train/test with the same noise parameters
  - `mixed`: augmentation sampling noise type/params per series

Run examples:
```bash
python scripts/exp1_noise.py --noise poisson --train-mode clean --rho 0.5 --run-baseline --run-mlp --normalize max
python scripts/exp1_noise.py --noise negbin --train-mode mixed --rho 0.5 --k 10 --run-baseline --run-cnn1d --normalize max
```

Optional plots (saved under `runs/<run>/figures/`):
```bash
python scripts/exp1_noise.py --noise poisson --train-mode clean --rho 0.5 --run-baseline --run-mlp --normalize max --save-plots --n-plot 9
```

Optional plot data (saved under `runs/<run>/figures/`):
```bash
python scripts/exp1_noise.py --noise poisson --train-mode clean --rho 0.5 --run-baseline --run-mlp --normalize max --save-plot-data --n-plot 9
```

### Exp2: Windowing + downsampling benchmark

File: `scripts/exp2_window_downsample.py`

What it does:
- Applies early window truncation and/or temporal downsampling to I(t) before fitting.
- Adjusts the classical baseline grid to the effective dt and horizon induced by downsampling
  **when `--run-baseline` (or `--run-all`) is set**.

Run:
```bash
python scripts/exp2_window_downsample.py --window-days 30 --downsample 10 --max-test 200 --run-baseline --run-mlp --normalize max
```

Optional plots (saved under `runs/<run>/figures/`):
```bash
python scripts/exp2_window_downsample.py --window-days 30 --downsample 10 --max-test 200 --run-baseline --run-mlp --normalize max --save-plots --n-plot 9
```

Optional plot data (saved under `runs/<run>/figures/`):
```bash
python scripts/exp2_window_downsample.py --window-days 30 --downsample 10 --max-test 200 --run-baseline --run-mlp --normalize max --save-plot-data --n-plot 9
```

### Exp3: Observation intensity sweep

File: `scripts/exp3_run.py`

Runs a sweep over observation intensity (e.g., `rho`) using a small subset of methods for
runtime reasons.

### Exp4: Variable initial conditions (and noise)

Files: `scripts/exp4_varinit.py`, `scripts/exp4_run.py`

Benchmarks inference when initial conditions (and optionally population size) vary, with optional
observation noise and training modes.

### Aggregation

File: `scripts/aggregate_runs.py`

Merges `runs/*/metrics.csv` into a single table:
```bash
python scripts/aggregate_runs.py --include-config
```

Output: `runs/summary.csv`

### Rebuild plots from saved plot data

If you ran experiments with `--save-plot-data`, you can rebuild figures without rerunning the
experiments:

```bash
python scripts/rebuild_plots.py --plot-data runs/exp0_YYYYMMDD_HHMMSS/figures
```

## Contact

For questions or suggestions, please contact:
- **Project Lead:** Patricio Olivares R.
- **Email:** patricio.olivaresr@usm.cl

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
