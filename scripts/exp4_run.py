"""Exp4 runner: orchestrate Exp4 runs over noise types and train modes.

Executes scripts/exp4_varinit.py across a grid of noise types (poisson/negbin)
and train modes (clean/noisy/mixed) while keeping variable-init settings.
Creates one run folder per configuration (prefixed with exp4_) so results
aggregate with the rest of the benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Sequence

from src.sir.logging_utils import setup_logging


@dataclass(frozen=True)
class RunConfig:
    noise: str
    train_mode: str


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slug_float(value: float) -> str:
    out = f"{value:.3f}".rstrip("0").rstrip(".")
    return out.replace(".", "p")


def _build_run_name(base: str, idx: int, cfg: RunConfig, rho: float, k: float) -> str:
    rho_tag = _slug_float(rho)
    name = f"{base}_{idx:02d}_{cfg.noise}_{cfg.train_mode}_rho{rho_tag}"
    if cfg.noise == "negbin":
        name += f"_k{_slug_float(k)}"
    return name


def _base_exp4_args(args: argparse.Namespace) -> List[str]:
    cmd: List[str] = []
    if args.data_path:
        cmd += ["--data-path", args.data_path]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.test_size is not None:
        cmd += ["--test-size", str(args.test_size)]
    if args.val_size is not None:
        cmd += ["--val-size", str(args.val_size)]
    if args.normalize is not None:
        cmd += ["--normalize", args.normalize]

    if args.population_range is not None:
        cmd += ["--population-range", str(args.population_range[0]), str(args.population_range[1])]
    if args.i0_range is not None:
        cmd += ["--i0-range", str(args.i0_range[0]), str(args.i0_range[1])]
    if args.r0_range is not None:
        cmd += ["--r0-range", str(args.r0_range[0]), str(args.r0_range[1])]
    if args.init_fraction is False:
        cmd += ["--no-init-fraction"]
    if args.feature_mode is not None:
        cmd += ["--feature-mode", args.feature_mode]
    if args.feature_scale is not None:
        cmd += ["--feature-scale", args.feature_scale]

    if args.n_starts is not None:
        cmd += ["--n-starts", str(args.n_starts)]
    if args.max_test is not None:
        cmd += ["--max-test", str(args.max_test)]
    if args.run_all:
        cmd += ["--run-all"]
    if args.run_baseline:
        cmd += ["--run-baseline"]
    if args.baseline_methods:
        cmd += ["--baseline-methods", args.baseline_methods]
    if args.wls_eps is not None:
        cmd += ["--wls-eps", str(args.wls_eps)]
    if args.log_eps is not None:
        cmd += ["--log-eps", str(args.log_eps)]
    if args.huber_delta is not None:
        cmd += ["--huber-delta", str(args.huber_delta)]
    if args.de_maxiter is not None:
        cmd += ["--de-maxiter", str(args.de_maxiter)]
    if args.de_popsize is not None:
        cmd += ["--de-popsize", str(args.de_popsize)]
    if args.de_polish is False:
        cmd += ["--no-de-polish"]

    if args.run_linear:
        cmd += ["--run-linear"]
    if args.run_mlp:
        cmd += ["--run-mlp"]
    if args.run_mlp_branched:
        cmd += ["--run-mlp-branched"]
    if args.run_resmlp:
        cmd += ["--run-resmlp"]
    if args.run_cnn1d:
        cmd += ["--run-cnn1d"]
    if args.run_tcn:
        cmd += ["--run-tcn"]
    if args.run_inception:
        cmd += ["--run-inception"]
    if args.run_attn_cnn:
        cmd += ["--run-attn-cnn"]
    if args.run_gru:
        cmd += ["--run-gru"]
    if args.run_lstm:
        cmd += ["--run-lstm"]
    if args.run_conv_gru:
        cmd += ["--run-conv-gru"]
    if args.run_transformer:
        cmd += ["--run-transformer"]
    if args.run_mlp_hetero:
        cmd += ["--run-mlp-hetero"]
    if args.run_mlp_mdn:
        cmd += ["--run-mlp-mdn"]

    if args.auto_select is False:
        cmd += ["--no-auto-select"]
    if args.exp1_final_log:
        cmd += ["--exp1-final-log", args.exp1_final_log]
    if args.top_baselines is not None:
        cmd += ["--top-baselines", str(args.top_baselines)]
    if args.top_ml is not None:
        cmd += ["--top-ml", str(args.top_ml)]

    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.patience is not None:
        cmd += ["--patience", str(args.patience)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.save_plots:
        cmd += ["--save-plots"]
    if args.save_plot_data:
        cmd += ["--save-plot-data"]
    if args.save_predictions:
        cmd += ["--save-predictions"]
    if args.n_plot is not None:
        cmd += ["--n-plot", str(args.n_plot)]
    if args.plot_dir:
        cmd += ["--plot-dir", args.plot_dir]
    if args.plot_max_ml is not None:
        cmd += ["--plot-max-ml", str(args.plot_max_ml)]
    if args.plot_max_baseline is not None:
        cmd += ["--plot-max-baseline", str(args.plot_max_baseline)]
    if args.plot_legend is not None:
        cmd += ["--plot-legend", args.plot_legend]
    if args.pred_dir:
        cmd += ["--pred-dir", args.pred_dir]

    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    if args.no_cache:
        cmd += ["--no-cache"]
    if args.progress_every is not None:
        cmd += ["--progress-every", str(args.progress_every)]
    if args.exp_log:
        cmd += ["--exp-log", args.exp_log]
    if args.log_level:
        cmd += ["--log-level", args.log_level]
    if args.no_console_log:
        cmd += ["--no-console-log"]
    if args.no_log_file:
        cmd += ["--no-log-file"]
    if args.mark_final:
        cmd += ["--mark-final"]
    if args.final_note:
        cmd += ["--final-note", args.final_note]
    if args.extra_args:
        cmd += list(args.extra_args)
    return cmd


def _build_command(
    args: argparse.Namespace,
    cfg: RunConfig,
    out_dir: Path,
    seed: int,
) -> List[str]:
    cmd = [sys.executable, "scripts/exp4_varinit.py"]
    cmd += ["--noise", cfg.noise]
    cmd += ["--train-mode", cfg.train_mode]
    cmd += ["--rho", str(args.rho)]
    cmd += ["--k", str(args.k)]
    cmd += ["--out-dir", str(out_dir)]
    cmd += ["--seed", str(seed)]
    cmd += ["--rho-range", str(args.rho_range[0]), str(args.rho_range[1])]
    cmd += ["--k-range", str(args.k_range[0]), str(args.k_range[1])]
    cmd += ["--p-poisson", str(args.p_poisson)]
    if args.estimate_rho:
        cmd += ["--estimate-rho"]
    cmd += _base_exp4_args(args)
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp4 sweep (noise types + train modes).")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument(
        "--noise-types",
        type=str,
        default="poisson,negbin",
        help="Comma-separated noise types to sweep.",
    )
    parser.add_argument(
        "--train-modes",
        type=str,
        default="clean,noisy,mixed",
        help="Comma-separated train modes to sweep.",
    )
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--rho-range", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--k-range", type=float, nargs=2, default=(5.0, 50.0))
    parser.add_argument("--p-poisson", type=float, default=0.5)
    parser.add_argument("--estimate-rho", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Optional offset added per run index to change randomness.",
    )
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--normalize", type=str, default=None, choices=[None, "max", "population"])
    parser.add_argument("--population-range", type=float, nargs=2, default=(50.0, 500.0))
    parser.add_argument("--i0-range", type=float, nargs=2, default=(0.01, 0.2))
    parser.add_argument("--r0-range", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--init-fraction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-mode", type=str, default="none", choices=["none", "append"])
    parser.add_argument("--feature-scale", type=str, default="fraction", choices=["absolute", "fraction"])
    parser.add_argument("--n-starts", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--baseline-methods", type=str, default="default")
    parser.add_argument("--wls-eps", type=float, default=1e-3)
    parser.add_argument("--log-eps", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--de-maxiter", type=int, default=100)
    parser.add_argument("--de-popsize", type=int, default=15)
    parser.add_argument("--de-polish", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--auto-select", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exp1-final-log", type=str, default="EXPERIMENTS.md")
    parser.add_argument("--top-baselines", type=int, default=2)
    parser.add_argument("--top-ml", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--save-plot-data", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--n-plot", type=int, default=None)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--plot-max-ml", type=int, default=None)
    parser.add_argument("--plot-max-baseline", type=int, default=None)
    parser.add_argument("--plot-legend", type=str, default=None, choices=["global", "first", "all", "none"])
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--exp-log", type=str, default="EXPERIMENTS.md")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--mark-final", action="store_true")
    parser.add_argument("--final-note", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Extra args to pass to exp4_varinit.py (use after --).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None
    if not args.no_log_file:
        log_path = Path("runs") / f"exp4_runner_{timestamp}.log"
    setup_logging(level=args.log_level, log_file=log_path, console=not args.no_console_log)
    logger = logging.getLogger(__name__)

    noise_types = _parse_csv_list(args.noise_types)
    train_modes = _parse_csv_list(args.train_modes)
    configs = [RunConfig(noise=n, train_mode=t) for n in noise_types for t in train_modes]

    total = len(configs)
    logger.info("Exp4 plan: %d runs", total)
    if args.dry_run:
        logger.info("Dry-run enabled; commands will be printed but not executed.")

    base_seed = args.seed
    for i, cfg in enumerate(configs, start=1):
        run_name = _build_run_name(f"exp4_{timestamp}", i, cfg, args.rho, args.k)
        out_dir = Path("runs") / run_name
        seed = base_seed + args.seed_offset * i if args.seed_offset else base_seed

        cmd = _build_command(args, cfg, out_dir, seed)

        logger.info("[%d/%d] %s", i, total, run_name)
        logger.info("Config: noise=%s train_mode=%s rho=%.3f k=%.2f", cfg.noise, cfg.train_mode, args.rho, args.k)
        logger.info("Command: %s", " ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Run failed (code=%s): %s", result.returncode, run_name)
            break

    logger.info("Exp4 runner finished.")


if __name__ == "__main__":
    main()
