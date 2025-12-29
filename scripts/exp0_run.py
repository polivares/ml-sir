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
import logging
from pathlib import Path
import time
from typing import Optional

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
from src.sir.logging_utils import setup_logging
from src.sir.experiment_log import update_experiment_log, summarize_args
from src.sir.predictions import save_predictions
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
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--run-mlp", action="store_true")
    parser.add_argument("--run-mlp-branched", action="store_true")
    parser.add_argument("--run-cnn1d", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--save-plot-data", action="store_true")
    parser.add_argument("--n-plot", type=int, default=9)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="data/processed/sir")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--exp-log", type=str, default="EXPERIMENTS.md")
    parser.add_argument("--mark-final", action="store_true")
    parser.add_argument("--final-note", type=str, default=None)
    return parser.parse_args()


def _subset(
    arr: np.ndarray,
    max_n: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # Optionally cap the number of curves to keep runtime bounded.
    if max_n is None or max_n <= 0:
        idx = np.arange(arr.shape[0])
        return arr, idx
    if arr.shape[0] <= max_n:
        idx = np.arange(arr.shape[0])
        return arr, idx
    idx = rng.choice(arr.shape[0], size=max_n, replace=False)
    return arr[idx], idx


def _choose_plot_indices(pool: np.ndarray, n_plot: int, rng: np.random.Generator) -> np.ndarray:
    """Choose a reproducible subset of indices for plotting."""
    if pool.size <= n_plot:
        return np.sort(pool)
    return np.sort(rng.choice(pool, size=n_plot, replace=False))


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULTS.runs_dir / f"exp0_{timestamp}"
    ensure_dir(out_dir)

    log_file = None
    if not args.no_log_file:
        log_file = Path(args.log_file) if args.log_file else out_dir / "run.log"
    setup_logging(level=args.log_level, log_file=log_file, console=not args.no_console_log)
    logger = logging.getLogger(__name__)

    logger.info("Exp0 start")
    logger.info("Output dir: %s", out_dir)

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    logger.info(
        "Config: limit=%s max-test=%s n-starts=%s seed=%s normalize=%s",
        args.limit,
        args.max_test,
        args.n_starts,
        args.seed,
        args.normalize,
    )

    # Cache derived arrays + splits to avoid recomputation.
    cache_config = {
        "data_path": str(args.data_path),
        "limit": args.limit,
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
    }
    cache_key = hash_config(cache_config)
    logger.info("Cache key: %s", cache_key)

    if not args.no_cache and cache_exists(args.cache_dir, cache_key):
        # Fast path: load arrays and split indices from cache.
        logger.info("Loading cached arrays from %s", args.cache_dir)
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
        logger.info("Loading sir.pkl from %s (limit=%s)", args.data_path, args.limit)
        data = load_sir_pkl(args.data_path, limit=args.limit, rng=rng)
        X, y = build_Xy_I_only(data, normalize=None)
        # Store population size per curve for optional normalization.
        pop = np.asarray([d[0][0].sum() for d in data], dtype=np.float32)

        # Reproducible split for benchmark runs.
        splits = train_val_test_split(
            X, y, test_size=args.test_size, val_size=args.val_size, rng=rng, return_indices=True
        )
        if not args.no_cache:
            logger.info("Saving cache to %s", args.cache_dir)
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

    logger.info(
        "Dataset shapes: train=%s val=%s test=%s (T=%s)",
        splits["X_train"].shape,
        splits["X_val"].shape,
        splits["X_test"].shape,
        splits["X_train"].shape[1],
    )

    # Limit classical fitting to keep runtime manageable.
    X_test_fit, idx_fit = _subset(X_test, args.max_test, rng)
    y_test_fit = y_test[idx_fit]
    logger.info("Baseline MSE fitting on %d curves", X_test_fit.shape[0])

    results = []

    # Baseline MSE fit per curve (multi-start).
    y_pred = []
    fit_times = []
    start = time.perf_counter()
    for i in range(X_test_fit.shape[0]):
        local_rng = np.random.default_rng(args.seed + i)
        fit = fit_mse(X_test_fit[i], n_starts=args.n_starts, rng=local_rng)
        y_pred.append(fit.params[:2])
        fit_times.append(sum(fit.times))
        if (i + 1) % args.progress_every == 0 or (i + 1) == X_test_fit.shape[0]:
            elapsed = time.perf_counter() - start
            avg = elapsed / (i + 1)
            logger.info(
                "Baseline progress: %d/%d curves (avg %.3fs/curve)",
                i + 1,
                X_test_fit.shape[0],
                avg,
            )

    y_pred = np.asarray(y_pred)
    baseline_pred = {int(idx_fit[i]): y_pred[i] for i in range(idx_fit.shape[0])}
    metrics = per_param_metrics(y_test_fit, y_pred)
    metrics.update(timing_summary(np.asarray(fit_times)))
    metrics.update({
        "method": "baseline_mse",
        "scenario": "clean",
        "n_test": int(X_test_fit.shape[0]),
    })
    results.append(metrics)

    # ML models (optional)
    y_pred_mlp = None
    y_pred_mlp_branched = None
    y_pred_cnn1d = None

    if args.run_mlp or args.run_mlp_branched or args.run_cnn1d:
        # Normalize with the same strategy across train/val/test.
        logger.info("Preparing normalized inputs for ML (method=%s)", args.normalize)
        X_train = normalize_series(splits["X_train"], method=args.normalize, population=pop_train)
        y_train = splits["y_train"]
        X_val = normalize_series(splits["X_val"], method=args.normalize, population=pop_val)
        y_val = splits["y_val"]
        X_test_ml = normalize_series(X_test, method=args.normalize, population=pop_test)
    else:
        logger.info("Skipping ML models (no --run-* flags)")

    if args.run_mlp:
        logger.info("Training MLP")
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
        logger.info("MLP training done (train_time_sec=%.2f)", train_res.train_time_sec)
        y_pred_mlp = model.predict(X_test_ml, verbose=0)
        metrics = per_param_metrics(y_test, y_pred_mlp)
        metrics.update(timing_summary(ml.predict_time_per_sample(model, X_test_ml)))
        metrics.update({
            "method": "mlp",
            "scenario": "clean",
            "n_test": int(X_test.shape[0]),
            "train_time_sec": float(train_res.train_time_sec),
        })
        results.append(metrics)

    if args.run_mlp_branched:
        logger.info("Training branched MLP")
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
        logger.info("Branched MLP training done (train_time_sec=%.2f)", train_res.train_time_sec)
        y_pred_mlp_branched = model.predict(X_test_ml, verbose=0)
        metrics = per_param_metrics(y_test, y_pred_mlp_branched)
        metrics.update(timing_summary(ml.predict_time_per_sample(model, X_test_ml)))
        metrics.update({
            "method": "mlp_branched",
            "scenario": "clean",
            "n_test": int(X_test.shape[0]),
            "train_time_sec": float(train_res.train_time_sec),
        })
        results.append(metrics)

    if args.run_cnn1d:
        logger.info("Training CNN1D")
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
        logger.info("CNN1D training done (train_time_sec=%.2f)", train_res.train_time_sec)
        y_pred_cnn1d = model.predict(X_test_ml, verbose=0)
        metrics = per_param_metrics(y_test, y_pred_cnn1d)
        metrics.update(timing_summary(ml.predict_time_per_sample(model, X_test_ml)))
        metrics.update({
            "method": "cnn1d",
            "scenario": "clean",
            "n_test": int(X_test.shape[0]),
            "train_time_sec": float(train_res.train_time_sec),
        })
        results.append(metrics)

    # Persist run configuration and metrics for aggregation.
    config = vars(args)
    config.update({"timestamp": timestamp})
    save_json(out_dir / "config.json", config)
    save_csv(out_dir / "metrics.csv", results)
    logger.info("Saved metrics to %s", out_dir / "metrics.csv")

    if args.save_predictions:
        pred_dir = Path(args.pred_dir) if args.pred_dir else out_dir
        ensure_dir(pred_dir)
        times = DEFAULTS.t0 + np.arange(X_test.shape[1]) * DEFAULTS.dt
        y_pred_by_method = {}

        baseline_full = np.full_like(y_test, np.nan, dtype=float)
        if idx_fit.size > 0:
            baseline_full[idx_fit] = y_pred
        y_pred_by_method["baseline_mse"] = baseline_full
        if y_pred_mlp is not None:
            y_pred_by_method["mlp"] = y_pred_mlp
        if y_pred_mlp_branched is not None:
            y_pred_by_method["mlp_branched"] = y_pred_mlp_branched
        if y_pred_cnn1d is not None:
            y_pred_by_method["cnn1d"] = y_pred_cnn1d

        save_predictions(
            pred_dir,
            times=times,
            i_true=X_test,
            y_true=y_test,
            y_pred_by_method=y_pred_by_method,
            idx_test=splits["idx_test"],
            idx_fit=idx_fit,
            metadata={
                "exp": "exp0",
                "scenario": "clean",
                "seed": args.seed,
                "s0": DEFAULTS.s0,
                "i0": DEFAULTS.i0,
                "r0": DEFAULTS.r0,
                "t0": DEFAULTS.t0,
                "dt": DEFAULTS.dt,
            },
        )
        logger.info("Saved predictions to %s", pred_dir)

    if args.save_plots or args.save_plot_data:
        from src.visualization import visualize as viz
        from src.sir.simulate import simulate_sir

        plot_dir = Path(args.plot_dir) if args.plot_dir else out_dir / "figures"
        ensure_dir(plot_dir)
        logger.info("Saving plots/plot-data to %s", plot_dir)

        plot_rng = np.random.default_rng(args.seed + 12345)
        plot_pool = idx_fit if idx_fit.size > 0 else np.arange(X_test.shape[0])
        plot_idx = _choose_plot_indices(plot_pool, args.n_plot, plot_rng)

        t1_eff = DEFAULTS.t0 + DEFAULTS.dt * (X_test.shape[1] - 1)
        times = DEFAULTS.t0 + np.arange(X_test.shape[1]) * DEFAULTS.dt

        curves_list = []
        error_list = []
        for idx in plot_idx:
            curves = {"I_true": X_test[idx]}
            errors = {}
            if idx in baseline_pred:
                params = baseline_pred[int(idx)]
                I_pred = simulate_sir(
                    params[0],
                    params[1],
                    s0=DEFAULTS.s0,
                    i0=DEFAULTS.i0,
                    r0=DEFAULTS.r0,
                    t0=DEFAULTS.t0,
                    t1=t1_eff,
                    dt=DEFAULTS.dt,
                )
                curves["I_pred_baseline"] = I_pred
                errors["baseline_mse"] = np.abs(I_pred - X_test[idx])

            if y_pred_mlp is not None:
                params = y_pred_mlp[idx]
                I_pred = simulate_sir(
                    params[0],
                    params[1],
                    s0=DEFAULTS.s0,
                    i0=DEFAULTS.i0,
                    r0=DEFAULTS.r0,
                    t0=DEFAULTS.t0,
                    t1=t1_eff,
                    dt=DEFAULTS.dt,
                )
                curves["I_pred_mlp"] = I_pred
                errors["mlp"] = np.abs(I_pred - X_test[idx])

            if y_pred_mlp_branched is not None:
                params = y_pred_mlp_branched[idx]
                I_pred = simulate_sir(
                    params[0],
                    params[1],
                    s0=DEFAULTS.s0,
                    i0=DEFAULTS.i0,
                    r0=DEFAULTS.r0,
                    t0=DEFAULTS.t0,
                    t1=t1_eff,
                    dt=DEFAULTS.dt,
                )
                curves["I_pred_mlp_branched"] = I_pred
                errors["mlp_branched"] = np.abs(I_pred - X_test[idx])

            if y_pred_cnn1d is not None:
                params = y_pred_cnn1d[idx]
                I_pred = simulate_sir(
                    params[0],
                    params[1],
                    s0=DEFAULTS.s0,
                    i0=DEFAULTS.i0,
                    r0=DEFAULTS.r0,
                    t0=DEFAULTS.t0,
                    t1=t1_eff,
                    dt=DEFAULTS.dt,
                )
                curves["I_pred_cnn1d"] = I_pred
                errors["cnn1d"] = np.abs(I_pred - X_test[idx])

            curves_list.append(curves)
            error_list.append(errors)

        y_true_plot = y_test_fit
        y_pred_by_method = {"baseline_mse": y_pred}
        if y_pred_mlp is not None:
            y_pred_by_method["mlp"] = y_pred_mlp[idx_fit]
        if y_pred_mlp_branched is not None:
            y_pred_by_method["mlp_branched"] = y_pred_mlp_branched[idx_fit]
        if y_pred_cnn1d is not None:
            y_pred_by_method["cnn1d"] = y_pred_cnn1d[idx_fit]

        if args.save_plots:
            viz.save_experiment_figures(
                plot_dir,
                times,
                curves_list,
                error_list,
                y_true_plot,
                y_pred_by_method,
                results,
                title_prefix="Exp0",
            )

        if args.save_plot_data:
            viz.save_plot_data(
                plot_dir,
                times,
                plot_idx,
                curves_list,
                error_list,
                y_true_plot,
                y_pred_by_method,
                idx_fit=idx_fit,
                metadata={
                    "exp": "exp0",
                    "scenario": "clean",
                    "seed": args.seed,
                    "n_plot": int(args.n_plot),
                },
            )

    artifacts = ["config.json", "metrics.csv", "run.log"]
    if args.save_predictions:
        artifacts.append("predictions.npz/json")
    if args.save_plots or args.save_plot_data:
        artifacts.append("figures/")

    args_summary = summarize_args(
        vars(args),
        keys=[
            "limit",
            "max_test",
            "n_starts",
            "seed",
            "normalize",
            "run_mlp",
            "run_mlp_branched",
            "run_cnn1d",
            "epochs",
            "patience",
            "batch_size",
            "save_plots",
            "save_plot_data",
            "save_predictions",
        ],
    )
    title = f"{timestamp} — Exp0 (clean)"
    update_experiment_log(
        args.exp_log,
        exp_key="exp0",
        title=title,
        run_dir=out_dir,
        script="scripts/exp0_run.py",
        args_summary=args_summary,
        artifacts=artifacts,
        metrics_rows=results,
        mark_final=args.mark_final,
        final_note=args.final_note,
    )


if __name__ == "__main__":
    main()
