"""Exp1 benchmark for observation noise robustness.

Adds Poisson or Negative Binomial noise to I(t), fits MLE baselines, and
optionally trains ML models under different train modes (clean/noisy/mixed),
including additional neural architectures.
This evaluates how well methods generalize when observations are noisy.
Writes a run folder with config.json and metrics.csv under runs/.
Typical usage:
  python scripts/exp1_noise.py --noise poisson --train-mode mixed --run-mlp
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
from src.sir.noise import observe_poisson, observe_negbin
from src.sir.baseline import fit_poisson_mle, fit_negbin_mle
from src.sir.io import ensure_dir, save_json, save_csv
from src.sir.cache import hash_config, cache_exists, load_cache, save_cache
from src.sir.logging_utils import setup_logging
from src.sir.experiment_log import update_experiment_log, summarize_args
from src.sir.predictions import save_predictions
from src.sir import ml


def _parse_args() -> argparse.Namespace:
    # CLI options control noise model, training mode, and model choices.
    parser = argparse.ArgumentParser(description="Run Exp1 benchmark (noise robustness).")
    parser.add_argument("--data-path", type=str, default=str(DEFAULTS.data_path))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--test-size", type=float, default=DEFAULTS.test_size)
    parser.add_argument("--val-size", type=float, default=DEFAULTS.val_size)
    parser.add_argument("--normalize", type=str, default=None, choices=[None, "max", "population"])
    parser.add_argument("--noise", type=str, default="poisson", choices=["poisson", "negbin"])
    parser.add_argument("--rho", type=float, default=DEFAULTS.rho)
    parser.add_argument("--k", type=float, default=DEFAULTS.k)
    parser.add_argument("--estimate-rho", action="store_true")
    parser.add_argument("--train-mode", type=str, default="clean", choices=["clean", "noisy", "mixed"])
    parser.add_argument("--rho-range", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--k-range", type=float, nargs=2, default=(5.0, 50.0))
    parser.add_argument("--p-poisson", type=float, default=0.5)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-mlp", action="store_true")
    parser.add_argument("--run-mlp-branched", action="store_true")
    parser.add_argument("--run-cnn1d", action="store_true")
    parser.add_argument("--run-linear", action="store_true")
    parser.add_argument("--run-resmlp", action="store_true")
    parser.add_argument("--run-tcn", action="store_true")
    parser.add_argument("--run-inception", action="store_true")
    parser.add_argument("--run-attn-cnn", action="store_true")
    parser.add_argument("--run-gru", action="store_true")
    parser.add_argument("--run-lstm", action="store_true")
    parser.add_argument("--run-conv-gru", action="store_true")
    parser.add_argument("--run-transformer", action="store_true")
    parser.add_argument("--run-mlp-hetero", action="store_true")
    parser.add_argument("--run-mlp-mdn", action="store_true")
    parser.add_argument("--run-all", action="store_true")
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


def _apply_noise_batch(
    X: np.ndarray,
    noise: str,
    rho: float,
    k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Apply the same noise model to every series in the batch.
    if noise == "poisson":
        return observe_poisson(X, rho=rho, rng=rng).astype(np.float32)
    if noise == "negbin":
        return observe_negbin(X, rho=rho, k=k, rng=rng).astype(np.float32)
    raise ValueError("Unknown noise type")


def _apply_mixed_noise(
    X: np.ndarray,
    rho_range: tuple[float, float],
    k_range: tuple[float, float],
    p_poisson: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Sample noise type/params per series for mixed augmentation.
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        if rng.random() < p_poisson:
            rho = rng.uniform(*rho_range)
            out[i] = observe_poisson(X[i], rho=rho, rng=rng)
        else:
            rho = rng.uniform(*rho_range)
            k = rng.uniform(*k_range)
            out[i] = observe_negbin(X[i], rho=rho, k=k, rng=rng)
    return out


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


def _relativize_paths(payload: dict[str, object], root: Path) -> dict[str, object]:
    """Convert Path objects in a dict to root-relative strings."""
    def _convert(value: object) -> object:
        if isinstance(value, Path):
            try:
                return str(value.relative_to(root))
            except ValueError:
                return str(value)
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    return {key: _convert(val) for key, val in payload.items()}


def _apply_run_all(args: argparse.Namespace) -> None:
    """Enable every ML architecture flag when --run-all is set."""
    if not args.run_all:
        return
    args.run_baseline = True
    for flag in (
        "run_linear",
        "run_mlp",
        "run_mlp_branched",
        "run_resmlp",
        "run_cnn1d",
        "run_tcn",
        "run_inception",
        "run_attn_cnn",
        "run_gru",
        "run_lstm",
        "run_conv_gru",
        "run_transformer",
        "run_mlp_hetero",
        "run_mlp_mdn",
    ):
        setattr(args, flag, True)


def main() -> None:
    args = _parse_args()
    _apply_run_all(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULTS.runs_dir / f"exp1_{timestamp}"
    ensure_dir(out_dir)
    models_dir = out_dir / "models"

    log_file = None
    if not args.no_log_file:
        log_file = Path(args.log_file) if args.log_file else out_dir / "run.log"
    setup_logging(level=args.log_level, log_file=log_file, console=not args.no_console_log)
    logger = logging.getLogger(__name__)

    logger.info("Exp1 start")
    logger.info("Output dir: %s", out_dir)

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    logger.info(
        "Config: noise=%s train-mode=%s rho=%s k=%s n-starts=%s seed=%s normalize=%s",
        args.noise,
        args.train_mode,
        args.rho,
        args.k,
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

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
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

    # Train/val observations (raw counts).
    if args.train_mode == "clean":
        X_train_obs = X_train
        X_val_obs = X_val
    elif args.train_mode == "noisy":
        X_train_obs = _apply_noise_batch(X_train, args.noise, args.rho, args.k, rng)
        X_val_obs = _apply_noise_batch(X_val, args.noise, args.rho, args.k, rng)
    else:
        # Mixed augmentation: per-series noise type/params.
        X_train_obs = _apply_mixed_noise(
            X_train, tuple(args.rho_range), tuple(args.k_range), args.p_poisson, rng
        )
        X_val_obs = _apply_mixed_noise(
            X_val, tuple(args.rho_range), tuple(args.k_range), args.p_poisson, rng
        )

    # Test observations (always noisy for Exp1, raw counts).
    X_test_obs = _apply_noise_batch(X_test, args.noise, args.rho, args.k, rng)
    logger.info("Applied observation noise to test set (noise=%s)", args.noise)

    results = []
    scenario = f"train_{args.train_mode}_test_{args.noise}"
    baseline_method = f"baseline_mle_{args.noise}"
    baseline_pred: dict[int, np.ndarray] = {}
    model_artifacts: list[dict[str, object]] = []
    idx_fit = np.array([], dtype=int)
    y_pred = y_test[:0]
    y_test_fit = y_test[:0]

    if args.run_baseline:
        # Baseline MLE per curve (multi-start).
        X_test_fit, idx_fit = _subset(X_test_obs, args.max_test, rng)
        y_test_fit = y_test[idx_fit]
        logger.info("Baseline MLE fitting on %d curves", X_test_fit.shape[0])
        y_pred_list = []
        fit_times = []
        start = time.perf_counter()
        for i in range(X_test_fit.shape[0]):
            local_rng = np.random.default_rng(args.seed + i)
            if args.noise == "poisson":
                fit = fit_poisson_mle(
                    X_test_fit[i],
                    rho=args.rho,
                    estimate_rho=args.estimate_rho,
                    n_starts=args.n_starts,
                    rng=local_rng,
                )
            else:
                fit = fit_negbin_mle(
                    X_test_fit[i],
                    rho=args.rho,
                    k=args.k,
                    estimate_rho=args.estimate_rho,
                    n_starts=args.n_starts,
                    rng=local_rng,
                )
            y_pred_list.append(fit.params[:2])
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

        y_pred = np.asarray(y_pred_list)
        baseline_pred = {int(idx_fit[i]): y_pred[i] for i in range(idx_fit.shape[0])}
        metrics = per_param_metrics(y_test_fit, y_pred)
        metrics.update(timing_summary(np.asarray(fit_times)))
        metrics.update({
            "method": baseline_method,
            "scenario": scenario,
            "n_test": int(X_test_fit.shape[0]),
        })
        results.append(metrics)
    else:
        logger.info("Skipping baseline fit (no --run-baseline flag)")

    y_pred_ml: dict[str, np.ndarray] = {}
    ml_flags = [
        args.run_linear,
        args.run_mlp,
        args.run_mlp_branched,
        args.run_resmlp,
        args.run_cnn1d,
        args.run_tcn,
        args.run_inception,
        args.run_attn_cnn,
        args.run_gru,
        args.run_lstm,
        args.run_conv_gru,
        args.run_transformer,
        args.run_mlp_hetero,
        args.run_mlp_mdn,
    ]

    # ML models (optional)
    if any(ml_flags):
        # Normalize with the same strategy across train/val/test.
        logger.info("Preparing normalized inputs for ML (method=%s)", args.normalize)
        X_train_in = normalize_series(X_train_obs, method=args.normalize, population=pop_train)
        X_val_in = normalize_series(X_val_obs, method=args.normalize, population=pop_val)
        X_test_in = normalize_series(X_test_obs, method=args.normalize, population=pop_test)

        input_dim = X_train_in.shape[1]
        model_specs = [
            ("linear", args.run_linear, lambda: ml.build_linear(input_dim=input_dim)),
            ("mlp", args.run_mlp, lambda: ml.build_mlp(input_dim=input_dim)),
            ("mlp_branched", args.run_mlp_branched, lambda: ml.build_mlp_branched(input_dim=input_dim)),
            ("resmlp", args.run_resmlp, lambda: ml.build_resmlp(input_dim=input_dim)),
            ("cnn1d", args.run_cnn1d, lambda: ml.build_cnn1d(input_len=input_dim)),
            ("tcn", args.run_tcn, lambda: ml.build_tcn(input_len=input_dim)),
            ("inception", args.run_inception, lambda: ml.build_inception(input_len=input_dim)),
            ("attn_cnn", args.run_attn_cnn, lambda: ml.build_attn_cnn(input_len=input_dim)),
            ("gru", args.run_gru, lambda: ml.build_gru(input_len=input_dim)),
            ("lstm", args.run_lstm, lambda: ml.build_lstm(input_len=input_dim)),
            ("conv_gru", args.run_conv_gru, lambda: ml.build_conv_gru(input_len=input_dim)),
            ("transformer", args.run_transformer, lambda: ml.build_transformer(input_len=input_dim)),
            ("mlp_hetero", args.run_mlp_hetero, lambda: ml.build_mlp_heteroscedastic(input_dim=input_dim)),
            ("mlp_mdn", args.run_mlp_mdn, lambda: ml.build_mlp_mdn(input_dim=input_dim)),
        ]

        for name, enabled, builder in model_specs:
            if not enabled:
                continue
            logger.info("Training %s", name)
            model = builder()
            train_res = ml.train_model(
                model,
                X_train_in,
                y_train,
                X_val_in,
                y_val,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
            logger.info("%s training done (train_time_sec=%.2f)", name, train_res.train_time_sec)
            y_pred = ml.predict_params(model, X_test_in)
            y_pred_ml[name] = y_pred
            metrics = per_param_metrics(y_test, y_pred)
            metrics.update(timing_summary(ml.predict_time_per_sample(model, X_test_in)))
            metrics.update({
                "method": name,
                "scenario": scenario,
                "n_test": int(X_test_in.shape[0]),
                "train_time_sec": float(train_res.train_time_sec),
            })
            results.append(metrics)
            artifact = ml.save_model_artifacts(model, name, models_dir)
            model_artifacts.append(_relativize_paths(artifact, out_dir))
    else:
        logger.info("Skipping ML models (no --run-* flags)")

    baseline_methods = [baseline_method] if args.run_baseline else []
    ensure_dir(models_dir)
    save_json(
        models_dir / "manifest.json",
        {
            "exp": "exp1",
            "baseline_methods": baseline_methods,
            "ml_architectures": sorted(y_pred_ml.keys()),
            "models": model_artifacts,
        },
    )

    # Persist run configuration and metrics for aggregation.
    config = vars(args)
    config.update({"timestamp": timestamp})
    config.update({
        "baseline_methods": baseline_methods,
        "ml_architectures": sorted(y_pred_ml.keys()),
    })
    save_json(out_dir / "config.json", config)
    save_csv(out_dir / "metrics.csv", results)
    logger.info("Saved metrics to %s", out_dir / "metrics.csv")

    if args.save_predictions:
        pred_dir = Path(args.pred_dir) if args.pred_dir else out_dir
        ensure_dir(pred_dir)
        times = DEFAULTS.t0 + np.arange(X_test.shape[1]) * DEFAULTS.dt
        y_pred_by_method = {}

        if args.run_baseline:
            baseline_full = np.full_like(y_test, np.nan, dtype=float)
            if idx_fit.size > 0:
                baseline_full[idx_fit] = y_pred
            y_pred_by_method[baseline_method] = baseline_full
        y_pred_by_method.update(y_pred_ml)

        save_predictions(
            pred_dir,
            times=times,
            i_true=X_test,
            i_obs=X_test_obs,
            y_true=y_test,
            y_pred_by_method=y_pred_by_method,
            idx_test=splits["idx_test"],
            idx_fit=idx_fit if args.run_baseline else None,
            metadata={
                "exp": "exp1",
                "scenario": scenario,
                "noise": args.noise,
                "train_mode": args.train_mode,
                "baseline_method": baseline_method if args.run_baseline else None,
                "ml_architectures": sorted(y_pred_ml.keys()),
                "run_all": bool(args.run_all),
                "run_baseline": bool(args.run_baseline),
                "normalize": args.normalize,
                "n_starts": int(args.n_starts),
                "max_test": args.max_test,
                "test_size": float(args.test_size),
                "val_size": float(args.val_size),
                "limit": args.limit,
                "rho": float(args.rho),
                "k": float(args.k),
                "estimate_rho": bool(args.estimate_rho),
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
            curves = {
                "I_true": X_test[idx],
                "Y_obs": X_test_obs[idx],
            }
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
                errors[baseline_method] = np.abs(I_pred - X_test[idx])

            for method_name, y_pred_full in y_pred_ml.items():
                params = y_pred_full[idx]
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
                curves[f"I_pred_{method_name}"] = I_pred
                errors[method_name] = np.abs(I_pred - X_test[idx])

            curves_list.append(curves)
            error_list.append(errors)

        use_fit_subset = args.run_baseline and idx_fit.size > 0
        if use_fit_subset:
            y_true_plot = y_test_fit
            y_pred_by_method = {baseline_method: y_pred} if args.run_baseline else {}
            for method_name, y_pred_full in y_pred_ml.items():
                y_pred_by_method[method_name] = y_pred_full[idx_fit]
        else:
            y_true_plot = y_test
            y_pred_by_method = dict(y_pred_ml)

        if args.save_plots:
            viz.save_experiment_figures(
                plot_dir,
                times,
                curves_list,
                error_list,
                y_true_plot,
                y_pred_by_method,
                results,
                title_prefix=f"Exp1 (noise={args.noise})",
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
                idx_fit=idx_fit if args.run_baseline else None,
                metadata={
                    "exp": "exp1",
                "scenario": scenario,
                "noise": args.noise,
                "train_mode": args.train_mode,
                "baseline_method": baseline_method if args.run_baseline else None,
                "ml_architectures": sorted(y_pred_ml.keys()),
                "run_all": bool(args.run_all),
                "run_baseline": bool(args.run_baseline),
                "normalize": args.normalize,
                "n_starts": int(args.n_starts),
                "max_test": args.max_test,
                "test_size": float(args.test_size),
                    "val_size": float(args.val_size),
                    "limit": args.limit,
                    "rho": float(args.rho),
                    "k": float(args.k),
                    "estimate_rho": bool(args.estimate_rho),
                    "seed": args.seed,
                    "n_plot": int(args.n_plot),
                },
            )

    artifacts = ["config.json", "metrics.csv", "run.log"]
    artifacts.append("models/")
    if args.save_predictions:
        artifacts.append("predictions.npz/json")
    if args.save_plots or args.save_plot_data:
        artifacts.append("figures/")

    args_summary = summarize_args(
        vars(args),
        keys=[
            "noise",
            "train_mode",
            "rho",
            "k",
            "estimate_rho",
            "limit",
            "max_test",
            "n_starts",
            "seed",
            "normalize",
            "run_all",
            "run_baseline",
            "run_mlp",
            "run_mlp_branched",
            "run_cnn1d",
            "run_linear",
            "run_resmlp",
            "run_tcn",
            "run_inception",
            "run_attn_cnn",
            "run_gru",
            "run_lstm",
            "run_conv_gru",
            "run_transformer",
            "run_mlp_hetero",
            "run_mlp_mdn",
            "epochs",
            "patience",
            "batch_size",
            "save_plots",
            "save_plot_data",
            "save_predictions",
        ],
    )
    title = f"{timestamp} — Exp1 (noise={args.noise}, train={args.train_mode})"
    update_experiment_log(
        args.exp_log,
        exp_key="exp1",
        title=title,
        run_dir=out_dir,
        script="scripts/exp1_noise.py",
        args_summary=args_summary,
        artifacts=artifacts,
        metrics_rows=results,
        mark_final=args.mark_final,
        final_note=args.final_note,
    )


if __name__ == "__main__":
    main()
