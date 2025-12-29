"""Exp0 benchmark on clean simulated SIR curves.

Loads sir.pkl simulations, builds a train/val/test split, fits the classical
MSE optimizer per curve, and optionally trains neural models (MLP/CNN1D).
This is the clean-data baseline for the benchmark suite.
Writes a run folder with config.json and metrics.csv under runs/.
Typical usage:
  python scripts/exp0_run.py --run-mlp --normalize max
"""


from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from src.sir.config import DEFAULTS, set_global_seed
from src.sir.datasets import (
    load_sir_pkl,
    build_Xy_I_only,
    train_val_test_split,
    normalize_series,
)
from src.sir.metrics import per_param_metrics, timing_summary
from src.sir.baseline import fit_mse
from src.sir.io import ensure_dir, save_json, save_csv
from src.sir.cache import hash_config, cache_exists, load_cache, save_cache
from src.sir import ml


def _parse_args() -> argparse.Namespace:
    # CLI options control data size, normalization, and which models to run.
    parser = argparse.ArgumentParser(description="Run Exp0 benchmark (clean data).")
    parser.add_argument("--data-path", type=str, default=str(DEFAULTS.data_path))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--test-size", type=float, default=DEFAULTS.test_size)
    parser.add_argument("--val-size", type=float, default=DEFAULTS.val_size)
    parser.add_argument("--normalize", type=str, default=None, choices=[None, "max", "population"])
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--run-mlp", action="store_true")
    parser.add_argument("--run-mlp-branched", action="store_true")
    parser.add_argument("--run-cnn1d", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=str, default="data/processed/sir")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def _subset(arr: np.ndarray, max_n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Optionally cap the number of curves to keep runtime bounded.
    if arr.shape[0] <= max_n:
        idx = np.arange(arr.shape[0])
        return arr, idx
    idx = rng.choice(arr.shape[0], size=max_n, replace=False)
    return arr[idx], idx


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Cache derived arrays + splits to avoid recomputation.
    cache_config = {
        "data_path": str(args.data_path),
        "limit": args.limit,
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
    }
    cache_key = hash_config(cache_config)

    if not args.no_cache and cache_exists(args.cache_dir, cache_key):
        # Fast path: load arrays and split indices from cache.
        arrays, _ = load_cache(args.cache_dir, cache_key)
        X = arrays["X"]
        y = arrays["y"]
        pop = arrays["pop"]
        splits = {
            "X_train": X[arrays["idx_train"]],
            "y_train": y[arrays["idx_train"]],
            "X_val": X[arrays["idx_val"]],
            "y_val": y[arrays["idx_val"]],
            "X_test": X[arrays["idx_test"]],
            "y_test": y[arrays["idx_test"]],
            "idx_train": arrays["idx_train"],
            "idx_val": arrays["idx_val"],
            "idx_test": arrays["idx_test"],
        }
    else:
        # Load simulations and build supervised dataset.
        data = load_sir_pkl(args.data_path, limit=args.limit, rng=rng)
        X, y = build_Xy_I_only(data, normalize=None)
        # Store population size per curve for optional normalization.
        pop = np.asarray([d[0][0].sum() for d in data], dtype=np.float32)

        # Reproducible split for benchmark runs.
        splits = train_val_test_split(
            X, y, test_size=args.test_size, val_size=args.val_size, rng=rng, return_indices=True
        )
        if not args.no_cache:
            save_cache(
                args.cache_dir,
                cache_key,
                {
                    "X": X,
                    "y": y,
                    "pop": pop,
                    "idx_train": splits["idx_train"],
                    "idx_val": splits["idx_val"],
                    "idx_test": splits["idx_test"],
                },
                cache_config,
            )

    X_test = splits["X_test"]
    y_test = splits["y_test"]
    pop_train = pop[splits["idx_train"]]
    pop_val = pop[splits["idx_val"]]
    pop_test = pop[splits["idx_test"]]

    # Limit classical fitting to keep runtime manageable.
    X_test_fit, idx_fit = _subset(X_test, args.max_test, rng)
    y_test_fit = y_test[idx_fit]

    results = []

    # Baseline MSE fit per curve (multi-start).
    y_pred = []
    fit_times = []
    for i in range(X_test_fit.shape[0]):
        local_rng = np.random.default_rng(args.seed + i)
        fit = fit_mse(X_test_fit[i], n_starts=args.n_starts, rng=local_rng)
        y_pred.append(fit.params[:2])
        fit_times.append(sum(fit.times))

    y_pred = np.asarray(y_pred)
    metrics = per_param_metrics(y_test_fit, y_pred)
    metrics.update(timing_summary(np.asarray(fit_times)))
    metrics.update({
        "method": "baseline_mse",
        "scenario": "clean",
        "n_test": int(X_test_fit.shape[0]),
    })
    results.append(metrics)

    # ML models (optional)
    if args.run_mlp or args.run_mlp_branched or args.run_cnn1d:
        # Normalize with the same strategy across train/val/test.
        X_train = normalize_series(splits["X_train"], method=args.normalize, population=pop_train)
        y_train = splits["y_train"]
        X_val = normalize_series(splits["X_val"], method=args.normalize, population=pop_val)
        y_val = splits["y_val"]
        X_test_ml = normalize_series(X_test, method=args.normalize, population=pop_test)

    if args.run_mlp:
        model = ml.build_mlp(input_dim=X_train.shape[1])
        train_res = ml.train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        y_pred = model.predict(X_test_ml, verbose=0)
        metrics = per_param_metrics(y_test, y_pred)
        metrics.update(timing_summary(ml.predict_time_per_sample(model, X_test_ml)))
        metrics.update({
            "method": "mlp",
            "scenario": "clean",
            "n_test": int(X_test.shape[0]),
            "train_time_sec": float(train_res.train_time_sec),
        })
        results.append(metrics)

    if args.run_mlp_branched:
        model = ml.build_mlp_branched(input_dim=X_train.shape[1])
        train_res = ml.train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        y_pred = model.predict(X_test_ml, verbose=0)
        metrics = per_param_metrics(y_test, y_pred)
        metrics.update(timing_summary(ml.predict_time_per_sample(model, X_test_ml)))
        metrics.update({
            "method": "mlp_branched",
            "scenario": "clean",
            "n_test": int(X_test.shape[0]),
            "train_time_sec": float(train_res.train_time_sec),
        })
        results.append(metrics)

    if args.run_cnn1d:
        model = ml.build_cnn1d(input_len=X_train.shape[1])
        train_res = ml.train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        y_pred = model.predict(X_test_ml, verbose=0)
        metrics = per_param_metrics(y_test, y_pred)
        metrics.update(timing_summary(ml.predict_time_per_sample(model, X_test_ml)))
        metrics.update({
            "method": "cnn1d",
            "scenario": "clean",
            "n_test": int(X_test.shape[0]),
            "train_time_sec": float(train_res.train_time_sec),
        })
        results.append(metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULTS.runs_dir / f"exp0_{timestamp}"
    ensure_dir(out_dir)

    # Persist run configuration and metrics for aggregation.
    config = vars(args)
    config.update({"timestamp": timestamp})
    save_json(out_dir / "config.json", config)
    save_csv(out_dir / "metrics.csv", results)


if __name__ == "__main__":
    main()
