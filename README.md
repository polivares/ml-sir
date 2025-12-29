# Epidemic Spreading and Machine Learning (ml-sir)

This repository benchmarks **classical parameter fitting** vs **Machine Learning / Deep Learning**
approaches to infer SIR parameters from time series.

The current benchmark focus is **parameter inference from the infected trajectory**:
given **I(t)**, predict **(beta, gamma)**.

## Current Benchmark Scope

- **Data**: simulated SIR trajectories generated with `summerepi` (imported as `summer`).
- **Targets**: SIR parameters `(beta, gamma)`.
- **Classical baselines**:
  - Per-curve optimization on clean data (MSE + L-BFGS-B, multi-start).
  - Per-curve maximum likelihood under observation noise (Poisson / NegBin, multi-start).
- **ML/DL baselines**:
  - MLP, branched-MLP, CNN1D (TensorFlow/Keras).
- **Realistic observation mechanisms** (incremental): Poisson, Negative Binomial, windows, downsampling.

## Repository Structure (relevant)

```
├── src/sir/                  # Reusable benchmark modules (simulation, datasets, noise, baselines, ML)
├── scripts/                  # Reproducible benchmark entrypoints (Exp0/Exp1/Exp2 + aggregation)
├── src/visualization/        # Plotting helpers (curve comparisons, metrics, etc.)
├── notebooks/benchmarks/     # Thin notebooks (run scripts + analyze results)
├── notebooks/exploratory/    # Exploratory notebooks (data generation + early experiments)
├── EXPERIMENTS.md            # Auto-updated experiment log (latest run + final selection)
├── data/raw/simulated/SIR/   # sir.pkl (simulated trajectories)
├── data/processed/sir/       # Cached derived arrays/splits (hash-based)
└── runs/                     # Run artifacts (config.json + metrics.csv per run)
```

Note: the repository still follows the cookiecutter-data-science layout; some older code may exist
under `src/data`, `src/models`, etc. The **active benchmark code** is under `src/sir/` and `scripts/`.

## Setup

`requirements.txt` is a **conda explicit spec** (not a pip requirements file). Create an environment with:

```bash
conda create -n ml-sir --file requirements.txt
conda activate ml-sir
pip install -e .
```

If you prefer a lighter CPU-only environment, install the minimal dependencies manually
(`numpy`, `scipy`, `summerepi`, `matplotlib`, and optionally `tensorflow` for ML experiments).

GPU note (TensorFlow):
- The benchmark enables **GPU memory growth** so TensorFlow allocates VRAM gradually instead of pre-allocating most of it.
- You can monitor VRAM usage during training with `nvidia-smi -l 1`.

## Data

Default dataset path: `data/raw/simulated/SIR/sir.pkl` (can be very large).

Expected `sir.pkl` schema (as used by `src/sir/datasets.py`):
- list of `(outputs, times, params)` tuples
- `outputs` is an array shaped `(T, 3)` with columns `[S, I, R]`
- `params` is `[beta, gamma]`

Data generation is done from `notebooks/exploratory/0.1-por-SIR_data_generation.ipynb`.

## Experiments (scripts)

All scripts write a run folder under `runs/` containing:
- `config.json` (CLI args + timestamp)
- `metrics.csv` (one row per method evaluated)
- `run.log` (console + file logs for traceability)
- `figures/` (if `--save-plots` is enabled)

Each experiment run also appends an entry to `EXPERIMENTS.md`, which tracks:
- the **last run per experiment**, and
- a **manual checkbox list** to mark final runs per experiment.

Common flags:
- `--seed`: reproducibility
- `--limit`: subsample the dataset (useful for quick runs)
- `--max-test`: cap number of test curves for classical fitting (runtime control)
- `--normalize {max,population}`: consistent scaling for ML models
- `--cache-dir` / `--no-cache`: caching for derived arrays/splits (default `data/processed/sir`)
- `--progress-every`: how often to log baseline progress (Exp0/Exp1/Exp2)
- `--log-level`: logging verbosity (`INFO` by default)
- `--log-file`: override log path (default: `runs/<run>/run.log`)
- `--no-log-file`: disable log file output
- `--no-console-log`: disable console logging
- `--save-predictions`: save per-curve `I(t)` plus `(beta,gamma)` predictions to `predictions.npz/json`
- `--pred-dir`: override where prediction artifacts are written (default: run folder)
- `--exp-log`: path to the experiment log (default: `EXPERIMENTS.md`)
- `--mark-final`: mark this run as the one used for final analysis in the experiment log
- `--final-note`: optional note stored with the final selection

### Exp0: Clean-data benchmark

File: `scripts/exp0_run.py`

What it does:
- Builds a train/val/test split from clean I(t).
- Fits the classical baseline (`baseline_mse`) per curve (multi-start L-BFGS-B).
- Optionally trains ML models and evaluates on the full test set.

Run:
```bash
python scripts/exp0_run.py --limit 5000 --max-test 200 --n-starts 5 --run-mlp --normalize max
```

Optional plots (saved under `runs/<run>/figures/`):
```bash
python scripts/exp0_run.py --limit 5000 --max-test 200 --n-starts 5 --run-mlp --normalize max --save-plots --n-plot 9
```

Optional plot data (saved under `runs/<run>/figures/`):
```bash
python scripts/exp0_run.py --limit 5000 --max-test 200 --n-starts 5 --run-mlp --normalize max --save-plot-data --n-plot 9
```

### Exp1: Observation-noise benchmark (Poisson / NegBin)

File: `scripts/exp1_noise.py`

What it does:
- Adds Poisson or Negative Binomial noise to I(t) (treated as observed counts).
- Fits the corresponding classical MLE baseline per curve.
- Optionally trains ML models under different training modes:
  - `clean`: train on clean I(t), test on noisy observations
  - `noisy`: train/test with the same noise parameters
  - `mixed`: augmentation sampling noise type/params per series

Run examples:
```bash
python scripts/exp1_noise.py --noise poisson --train-mode clean --rho 0.5 --run-mlp --normalize max
python scripts/exp1_noise.py --noise negbin --train-mode mixed --rho 0.5 --k 10 --run-cnn1d --normalize max
```

Optional plots (saved under `runs/<run>/figures/`):
```bash
python scripts/exp1_noise.py --noise poisson --train-mode clean --rho 0.5 --run-mlp --normalize max --save-plots --n-plot 9
```

Optional plot data (saved under `runs/<run>/figures/`):
```bash
python scripts/exp1_noise.py --noise poisson --train-mode clean --rho 0.5 --run-mlp --normalize max --save-plot-data --n-plot 9
```

### Exp2: Windowing + downsampling benchmark

File: `scripts/exp2_window_downsample.py`

What it does:
- Applies early window truncation and/or temporal downsampling to I(t) before fitting.
- Adjusts the classical baseline grid to the effective dt and horizon induced by downsampling.

Run:
```bash
python scripts/exp2_window_downsample.py --window-days 30 --downsample 10 --max-test 200 --run-mlp --normalize max
```

Optional plots (saved under `runs/<run>/figures/`):
```bash
python scripts/exp2_window_downsample.py --window-days 30 --downsample 10 --max-test 200 --run-mlp --normalize max --save-plots --n-plot 9
```

Optional plot data (saved under `runs/<run>/figures/`):
```bash
python scripts/exp2_window_downsample.py --window-days 30 --downsample 10 --max-test 200 --run-mlp --normalize max --save-plot-data --n-plot 9
```

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

## Notebooks

Thin benchmark notebooks:
- `notebooks/benchmarks/exp0_clean.ipynb`: runs Exp0 and inspects metrics.
- `notebooks/benchmarks/exp1_noise.ipynb`: runs Exp1 and inspects metrics.

Exploratory notebooks (older / less standardized):
- `notebooks/exploratory/0.1-por-SIR_data_generation.ipynb`: generates SIR simulations and saves sir.pkl.
- `notebooks/exploratory/1.1-por-SIR_data_fit_traditional.ipynb`: classical fitting on clean data.
- `notebooks/exploratory/1.2-por-SIR_data_fit_traditional_noise.ipynb`: classical fitting with noise.
- `notebooks/exploratory/2.1-por-SIR_data_fit_NN.ipynb`: NN baseline on clean data.
- `notebooks/exploratory/2.2-por-SIR_data_fit_NN_noise.ipynb`: NN baseline with noise.

## Contact

For questions or suggestions, please contact:
- **Project Lead:** Patricio Olivares R.
- **Email:** patricio.olivaresr@usm.cl

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
