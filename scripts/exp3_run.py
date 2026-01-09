"""Exp3 runner: orchestrate Exp1 noise runs over a rho grid.

This script executes multiple Exp1 runs (scripts/exp1_noise.py) with different
rho settings to benchmark robustness under under-reporting (rho shifts).
It creates one run folder per configuration (prefixed with exp3_) so results
aggregate with the rest of the benchmark.

Optionally, Exp3 can auto-select the top baselines/ML methods from the final
Exp1 run recorded in EXPERIMENTS.md and run Exp3 with only those methods.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys
from typing import List, Mapping, Optional, Sequence

import csv
import json
import re

import numpy as np

from src.sir.logging_utils import setup_logging
from src.sir.experiment_log import update_experiment_log, summarize_args


@dataclass(frozen=True)
class RunConfig:
    noise: str
    train_mode: str
    rho: float
    k: float
    p_poisson: float
    rho_range: tuple[float, float]
    k_range: tuple[float, float]


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _slug_float(value: float) -> str:
    out = f"{value:.3f}".rstrip("0").rstrip(".")
    return out.replace(".", "p")


def _build_run_name(base: str, idx: int, cfg: RunConfig) -> str:
    rho_tag = _slug_float(cfg.rho)
    name = f"{base}_{idx:02d}_{cfg.noise}_{cfg.train_mode}_rho{rho_tag}"
    if cfg.noise == "negbin":
        name += f"_k{_slug_float(cfg.k)}"
    return name


def _extract_block(text: str, start: str, end: str) -> str:
    if start in text and end in text:
        _, rest = text.split(start, 1)
        content, _ = rest.split(end, 1)
        return content.strip()
    return ""


def _load_final_exp1_run(exp_log: Path) -> Optional[Path]:
    if not exp_log.exists():
        return None
    text = exp_log.read_text(encoding="utf-8")
    start = "<!-- EXP1_FINAL_START -->"
    end = "<!-- EXP1_FINAL_END -->"
    block = _extract_block(text, start, end)
    if not block:
        return None
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    checked = [ln for ln in lines if ln.lower().startswith("- [x]")]
    if not checked:
        return None
    # Pick the last checked line (most recent manual selection).
    line = checked[-1]
    match = re.search(r"run_dir: `([^`]+)`", line)
    if not match:
        return None
    return Path(match.group(1))


def _parse_metrics(path: Path) -> List[dict[str, object]]:
    rows: List[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = ""
                    continue
                if key in ("method", "scenario", "run_dir", "exp"):
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _score_row(row: Mapping[str, object]) -> Optional[float]:
    def _get(key: str) -> Optional[float]:
        value = row.get(key)
        if value is None or value == "":
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):
            return None
        return v

    r2_beta = _get("r2_beta")
    r2_gamma = _get("r2_gamma")
    if r2_beta is not None or r2_gamma is not None:
        vals = [v for v in (r2_beta, r2_gamma) if v is not None]
        return -float(np.mean(vals)) if vals else None

    mae_beta = _get("mae_beta")
    mae_gamma = _get("mae_gamma")
    if mae_beta is not None or mae_gamma is not None:
        vals = [v for v in (mae_beta, mae_gamma) if v is not None]
        return float(np.mean(vals)) if vals else None

    rmse_beta = _get("rmse_beta")
    rmse_gamma = _get("rmse_gamma")
    if rmse_beta is not None or rmse_gamma is not None:
        vals = [v for v in (rmse_beta, rmse_gamma) if v is not None]
        return float(np.mean(vals)) if vals else None

    return None


def _select_best_methods(
    metrics_rows: Sequence[Mapping[str, object]],
    top_baselines: int,
    top_ml: int,
) -> tuple[List[str], List[str]]:
    scores: dict[str, list[float]] = {}
    for row in metrics_rows:
        method = row.get("method")
        if not method:
            continue
        score = _score_row(row)
        if score is None:
            continue
        scores.setdefault(str(method), []).append(score)

    avg_scores = {m: float(np.mean(vals)) for m, vals in scores.items() if vals}

    baseline_scores = {m: s for m, s in avg_scores.items() if m.startswith("baseline_")}
    ml_scores = {m: s for m, s in avg_scores.items() if not m.startswith("baseline_")}

    best_baselines = [m for m, _ in sorted(baseline_scores.items(), key=lambda x: x[1])][:top_baselines]
    best_ml = [m for m, _ in sorted(ml_scores.items(), key=lambda x: x[1])][:top_ml]
    return best_baselines, best_ml


def _baseline_label_to_flag(label: str) -> Optional[str]:
    if not label.startswith("baseline_"):
        return None
    return label[len("baseline_"):]


def _ml_label_to_flag(label: str) -> Optional[str]:
    mapping = {
        "linear": "--run-linear",
        "mlp": "--run-mlp",
        "mlp_branched": "--run-mlp-branched",
        "resmlp": "--run-resmlp",
        "cnn1d": "--run-cnn1d",
        "tcn": "--run-tcn",
        "inception": "--run-inception",
        "attn_cnn": "--run-attn-cnn",
        "gru": "--run-gru",
        "lstm": "--run-lstm",
        "conv_gru": "--run-conv-gru",
        "transformer": "--run-transformer",
        "mlp_hetero": "--run-mlp-hetero",
        "mlp_mdn": "--run-mlp-mdn",
    }
    return mapping.get(label)


def _base_exp1_args(
    args: argparse.Namespace,
    run_all: Optional[bool] = None,
    run_baseline: Optional[bool] = None,
    baseline_methods: Optional[str] = None,
) -> List[str]:
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
    if args.n_starts is not None:
        cmd += ["--n-starts", str(args.n_starts)]
    if args.max_test is not None:
        cmd += ["--max-test", str(args.max_test)]
    if run_all is None:
        run_all = args.run_all
    if run_baseline is None:
        run_baseline = args.run_baseline
    if baseline_methods is None:
        baseline_methods = args.baseline_methods

    if run_all:
        cmd += ["--run-all"]
    if run_baseline:
        cmd += ["--run-baseline"]
    if baseline_methods:
        cmd += ["--baseline-methods", baseline_methods]
    if args.estimate_rho:
        cmd += ["--estimate-rho"]
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
    if args.plot_max_ml is not None:
        cmd += ["--plot-max-ml", str(args.plot_max_ml)]
    if args.plot_max_baseline is not None:
        cmd += ["--plot-max-baseline", str(args.plot_max_baseline)]
    if args.plot_legend is not None:
        cmd += ["--plot-legend", args.plot_legend]
    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    if args.no_cache:
        cmd += ["--no-cache"]
    if args.exp1_exp_log:
        cmd += ["--exp-log", args.exp1_exp_log]
    if args.log_level:
        cmd += ["--log-level", args.log_level]
    if args.no_console_log:
        cmd += ["--no-console-log"]
    if args.no_log_file:
        cmd += ["--no-log-file"]
    if args.extra_args:
        cmd += list(args.extra_args)
    return cmd


def _build_run_configs(args: argparse.Namespace) -> List[RunConfig]:
    rhos = _parse_float_list(args.rho_values)
    noise_types = _parse_csv_list(args.noise_types)
    train_modes = _parse_csv_list(args.train_modes)
    cfgs: List[RunConfig] = []
    for noise in noise_types:
        for train_mode in train_modes:
            for rho in rhos:
                if train_mode == "mixed" and args.force_single_noise:
                    p_poisson = 1.0 if noise == "poisson" else 0.0
                else:
                    p_poisson = args.p_poisson
                cfgs.append(
                    RunConfig(
                        noise=noise,
                        train_mode=train_mode,
                        rho=rho,
                        k=args.k,
                        p_poisson=p_poisson,
                        rho_range=tuple(args.rho_range),
                        k_range=tuple(args.k_range),
                    )
                )
    return cfgs


def _build_command(
    args: argparse.Namespace,
    cfg: RunConfig,
    out_dir: Path,
    seed: int,
    run_all: Optional[bool] = None,
    run_baseline: Optional[bool] = None,
    baseline_methods: Optional[str] = None,
    ml_flags: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd = [sys.executable, "scripts/exp1_noise.py"]
    cmd += ["--noise", cfg.noise]
    cmd += ["--rho", str(cfg.rho)]
    cmd += ["--k", str(cfg.k)]
    cmd += ["--train-mode", cfg.train_mode]
    cmd += ["--out-dir", str(out_dir)]
    cmd += ["--seed", str(seed)]
    if cfg.train_mode == "mixed":
        cmd += ["--rho-range", str(cfg.rho_range[0]), str(cfg.rho_range[1])]
        cmd += ["--k-range", str(cfg.k_range[0]), str(cfg.k_range[1])]
        cmd += ["--p-poisson", str(cfg.p_poisson)]
    cmd += _base_exp1_args(args, run_all=run_all, run_baseline=run_baseline, baseline_methods=baseline_methods)
    if ml_flags:
        cmd += list(ml_flags)
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp3 (rho sweep via Exp1).")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument(
        "--rho-values",
        type=str,
        default="1.0,0.7,0.5,0.3,0.1",
        help="Comma-separated rho values for test noise.",
    )
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
    parser.add_argument("--rho-range", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--k-range", type=float, nargs=2, default=(5.0, 50.0))
    parser.add_argument("--p-poisson", type=float, default=0.5)
    parser.add_argument(
        "--force-single-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force mixed mode to use only the selected noise type (default: true).",
    )
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
    parser.add_argument("--n-starts", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--baseline-methods", type=str, default="all")
    parser.add_argument(
        "--auto-select",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the final Exp1 run in EXPERIMENTS.md to select top methods.",
    )
    parser.add_argument(
        "--exp1-final-log",
        type=str,
        default="EXPERIMENTS.md",
        help="Experiment log to read the final Exp1 selection from.",
    )
    parser.add_argument("--top-baselines", type=int, default=2)
    parser.add_argument("--top-ml", type=int, default=2)
    parser.add_argument("--estimate-rho", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--save-plot-data", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--n-plot", type=int, default=None)
    parser.add_argument("--plot-max-ml", type=int, default=None)
    parser.add_argument("--plot-max-baseline", type=int, default=None)
    parser.add_argument("--plot-legend", type=str, default=None, choices=["global", "first", "all", "none"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--exp-log", type=str, default="EXPERIMENTS.md")
    parser.add_argument(
        "--exp1-exp-log",
        type=str,
        default=None,
        help="Optional exp1 log file to avoid mixing exp3 runs into EXPERIMENTS.md.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Extra args to pass to exp1_noise.py (use after --).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None
    if not args.no_log_file:
        log_path = Path("runs") / f"exp3_runner_{timestamp}.log"
    setup_logging(level=args.log_level, log_file=log_path, console=not args.no_console_log)
    logger = logging.getLogger(__name__)

    configs = _build_run_configs(args)
    total = len(configs)
    logger.info("Exp3 plan: %d runs", total)
    if args.dry_run:
        logger.info("Dry-run enabled; commands will be printed but not executed.")

    selected_baselines: Optional[List[str]] = None
    selected_ml_flags: Optional[List[str]] = None
    if args.auto_select:
        exp1_run_dir = _load_final_exp1_run(Path(args.exp1_final_log))
        if exp1_run_dir is None:
            logger.warning("No final Exp1 run found in %s; falling back to CLI flags.", args.exp1_final_log)
        else:
            metrics_path = exp1_run_dir / "metrics.csv"
            if not metrics_path.exists():
                logger.warning("Missing metrics.csv at %s; falling back to CLI flags.", metrics_path)
            else:
                metrics_rows = _parse_metrics(metrics_path)
                best_baselines, best_ml = _select_best_methods(metrics_rows, args.top_baselines, args.top_ml)
                baseline_flags = [m for m in (_baseline_label_to_flag(x) for x in best_baselines) if m]
                ml_flags = [m for m in (_ml_label_to_flag(x) for x in best_ml) if m]
                if not baseline_flags or not ml_flags:
                    logger.warning(
                        "Could not auto-select both baselines and ML methods; falling back to CLI flags."
                    )
                else:
                    selected_baselines = baseline_flags
                    selected_ml_flags = ml_flags
                    logger.info("Auto-selected baselines: %s", selected_baselines)
                    logger.info("Auto-selected ML flags: %s", selected_ml_flags)

    base_seed = args.seed
    for i, cfg in enumerate(configs, start=1):
        run_name = _build_run_name(f"exp3_{timestamp}", i, cfg)
        out_dir = Path("runs") / run_name
        seed = base_seed + args.seed_offset * i if args.seed_offset else base_seed
        run_all = args.run_all
        run_baseline = args.run_baseline
        baseline_methods = args.baseline_methods
        ml_flags = None
        if args.auto_select and (selected_baselines or selected_ml_flags):
            run_all = False
            run_baseline = True
            if selected_baselines:
                baseline_methods = ",".join(selected_baselines)
            else:
                baseline_methods = ""
            ml_flags = selected_ml_flags

        cmd = _build_command(
            args,
            cfg,
            out_dir,
            seed,
            run_all=run_all,
            run_baseline=run_baseline,
            baseline_methods=baseline_methods,
            ml_flags=ml_flags,
        )

        logger.info("[%d/%d] %s", i, total, run_name)
        logger.info("Config: noise=%s train_mode=%s rho=%.3f k=%.2f", cfg.noise, cfg.train_mode, cfg.rho, cfg.k)
        logger.info("Command: %s", " ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Run failed (code=%s): %s", result.returncode, run_name)
            break

        config_path = out_dir / "config.json"
        metrics_path = out_dir / "metrics.csv"
        if not (config_path.exists() and metrics_path.exists()):
            logger.warning("Missing config/metrics for %s; skipping EXPERIMENTS update.", run_name)
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics_rows = []
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    if value is None or value == "":
                        parsed[key] = ""
                        continue
                    if key in ("method", "scenario", "run_dir", "exp"):
                        parsed[key] = value
                        continue
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
                metrics_rows.append(parsed)

        args_summary = summarize_args(
            config,
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
                "baseline_methods",
                "epochs",
                "patience",
                "batch_size",
                "save_plots",
                "save_plot_data",
                "save_predictions",
            ],
        )

        artifacts = ["config.json", "metrics.csv", "run.log", "models/"]
        if config.get("save_predictions"):
            artifacts.append("predictions.npz/json")
            artifacts.append("predicted_sir/")
        if config.get("save_plots") or config.get("save_plot_data"):
            artifacts.append("figures/")

        title = f"Exp3 (noise={cfg.noise}, train={cfg.train_mode}, rho={cfg.rho:.3f})"
        update_experiment_log(
            args.exp_log,
            exp_key="exp3",
            title=title,
            run_dir=out_dir,
            script="scripts/exp3_run.py",
            args_summary=args_summary,
            artifacts=artifacts,
            metrics_rows=metrics_rows,
        )

    logger.info("Exp3 runner finished.")


if __name__ == "__main__":
    main()
