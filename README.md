# ml-sir: SIR parameter inference benchmark

This repository provides a reproducible benchmark for inferring SIR parameters
$(beta, gamma)$ from an infected trajectory $I(t)$. It compares per-series
classical fitting methods against amortized neural inference models.

## What is included

- `src/sir/`: simulation wrapper, dataset utilities, observation noise models,
  classical baselines, neural models, and evaluation metrics.
- `scripts/`: command line entrypoints for benchmark scenarios and run
  aggregation.

Large datasets, run outputs, trained weights, and paper sources are kept out of
version control.

## Setup

`requirements.txt` is a conda explicit spec.

```bash
conda create -n ml-sir --file requirements.txt
conda activate ml-sir
pip install -e .
```

## Data

The scripts expect a dataset at `data/raw/simulated/SIR/sir.pkl` by default.
The file can be large and is not stored in version control.

## Running experiments

Experiment scripts write a timestamped run folder under `runs/` with a
`config.json` and `metrics.csv`.

Common options:
- `--seed` for reproducibility
- `--limit` to subsample the dataset for quick runs
- `--run-baseline` to enable classical fitting
- `--run-mlp` or `--run-all` to enable neural models

Examples:

```bash
python scripts/exp0_run.py --limit 5000 --max-test 200 --n-starts 5 --run-baseline --run-mlp
python scripts/exp1_noise.py --noise poisson --train-mode mixed --rho 0.5 --run-baseline --run-mlp
python scripts/aggregate_runs.py --include-config
```

## Notes

- The benchmark uses a hash-based cache under `data/processed/` unless
  `--no-cache` is set.
- When TensorFlow is available, GPU memory growth is enabled to avoid
  pre-allocating most VRAM at startup.
