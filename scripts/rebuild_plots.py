"""Rebuild standard figures from saved plot_data.npz/json.

This script recreates the experiment figures without re-running the experiment.
It expects plot data saved by --save-plot-data in runs/<run>/figures/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.sir.metrics import per_param_metrics
from src.sir.logging_utils import setup_logging
from src.visualization import visualize as viz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild plots from plot_data.npz/json.")
    parser.add_argument(
        "--plot-data",
        type=str,
        required=True,
        help="Path to plot_data.npz or a run/figures directory containing it.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to the plot_data directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix used when plot_data was saved (default: '').",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix override for the figures.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args()


def _resolve_plot_paths(path: Path, prefix: str) -> Tuple[Path, Path, Path]:
    """Return (plot_dir, npz_path, json_path) for plot data."""
    if path.is_file():
        npz_path = path
        plot_dir = path.parent
    else:
        npz_name = f"{prefix}plot_data.npz"
        json_name = f"{prefix}plot_data.json"
        if (path / npz_name).exists():
            plot_dir = path
            npz_path = path / npz_name
        elif (path / "figures" / npz_name).exists():
            plot_dir = path / "figures"
            npz_path = plot_dir / npz_name
        else:
            raise FileNotFoundError(f"plot_data.npz not found under {path}")
    json_path = npz_path.with_suffix(".json")
    return plot_dir, npz_path, json_path


def _trim_nan(series: np.ndarray) -> np.ndarray | None:
    series = np.asarray(series)
    mask = ~np.isnan(series)
    if not np.any(mask):
        return None
    last = int(np.where(mask)[0][-1])
    return series[: last + 1]


def _unstack_curves(labels: List[str], stacked: np.ndarray) -> List[Dict[str, np.ndarray]]:
    curves_list: List[Dict[str, np.ndarray]] = []
    if stacked.size == 0:
        return curves_list
    for i in range(stacked.shape[1]):
        curves: Dict[str, np.ndarray] = {}
        for li, label in enumerate(labels):
            series = _trim_nan(stacked[li, i])
            if series is not None:
                curves[label] = series
        curves_list.append(curves)
    return curves_list


def _title_from_meta(meta: Dict[str, object]) -> str:
    exp = str(meta.get("exp", "exp")).lower()
    if exp == "exp1":
        return f"Exp1 (noise={meta.get('noise')}, train={meta.get('train_mode')})"
    if exp == "exp2":
        return f"Exp2 (window={meta.get('window_days')}, downsample={meta.get('downsample')})"
    if exp == "exp0":
        return "Exp0"
    return exp.upper()


def main() -> None:
    args = _parse_args()
    plot_dir, npz_path, json_path = _resolve_plot_paths(Path(args.plot_data), args.prefix)
    out_dir = Path(args.out_dir) if args.out_dir else plot_dir

    log_file = None
    if not args.no_log_file:
        log_file = Path(args.log_file) if args.log_file else out_dir / "rebuild_plots.log"
    setup_logging(level=args.log_level, log_file=log_file, console=not args.no_console_log)
    logger = logging.getLogger(__name__)

    logger.info("Rebuilding plots from %s", npz_path)

    arrays = np.load(npz_path)
    meta: Dict[str, object] = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        logger.info("Loaded plot metadata from %s", json_path)

    curve_labels = list(meta.get("curve_labels", []))
    error_labels = list(meta.get("error_labels", []))

    times = arrays["times"]
    curve_stack = arrays.get("curve_stack", np.zeros((0, 0, 0)))
    error_stack = arrays.get("error_stack", np.zeros((0, 0, 0)))

    if not curve_labels and curve_stack.size:
        curve_labels = [f"curve_{i}" for i in range(curve_stack.shape[0])]
    if not error_labels and error_stack.size:
        error_labels = [f"error_{i}" for i in range(error_stack.shape[0])]

    curves_list = _unstack_curves(curve_labels, curve_stack)
    error_list = _unstack_curves(error_labels, error_stack)
    logger.info("Loaded %d curve sets for plotting", len(curves_list))

    y_true_fit = arrays["y_true_fit"]
    y_pred_by_method: Dict[str, np.ndarray] = {}

    y_pred_labels = meta.get("y_pred_labels")
    if y_pred_labels is None:
        y_pred_labels = sorted(
            k[len("y_pred_"):] for k in arrays.files if k.startswith("y_pred_")
        )

    for label in y_pred_labels:
        key = f"y_pred_{label}"
        if key in arrays:
            y_pred_by_method[str(label)] = arrays[key]
    logger.info("Methods available for plots: %s", sorted(y_pred_by_method.keys()))

    results = []
    for label, y_pred in y_pred_by_method.items():
        metrics = per_param_metrics(y_true_fit, y_pred)
        metrics["method"] = label
        results.append(metrics)

    title_prefix = args.title if args.title else _title_from_meta(meta)

    viz.save_experiment_figures(
        out_dir,
        times,
        curves_list,
        error_list,
        y_true_fit,
        y_pred_by_method,
        results,
        title_prefix=title_prefix,
    )
    logger.info("Saved figures to %s", out_dir)


if __name__ == "__main__":
    main()
