"""Plotting utilities for SIR benchmarks.

This module provides reusable Matplotlib helpers to visualize:
- curve comparisons (true vs predicted vs observed)
- error curves over time
- parameter scatter plots (beta/gamma)
- parameter error distributions
- metric comparisons across methods

It can also be used as a script to build plots from saved experiment
artifacts (predictions.npz/json and optionally metrics.csv), e.g.:
  python -m src.visualization.visualize --predictions runs/exp0_.../predictions.npz
The plotting script limits the number of ML methods by default (for readability);
override with --plot-max-ml/--plot-max-baseline when needed.
If predictions.npz includes per-curve `s0/i0/r0` arrays, those are used when
reconstructing simulated curves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

from src.sir.config import DEFAULTS
from src.sir.io import ensure_dir
from src.sir.metrics import per_param_metrics


def save_figure(fig: plt.Figure, path: Path | str, dpi: int = 150) -> Path:
    """Save a Matplotlib figure and ensure the parent directory exists."""
    path = Path(path)
    ensure_dir(path.parent)
    has_suptitle = getattr(fig, "_suptitle", None) is not None
    has_legend = bool(fig.legends)
    top = 0.93 if has_suptitle else 1.0
    bottom = 0.12 if has_legend else 0.0
    if top < 1.0 or bottom > 0.0:
        fig.tight_layout(rect=(0.0, bottom, 1.0, top))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def stack_curves(
    curves_list: Sequence[Mapping[str, np.ndarray]],
    fill_value: float = np.nan,
) -> Tuple[List[str], np.ndarray]:
    """Stack a list of labeled curve dicts into a 3D array.

    Returns
    -------
    labels:
        Sorted list of curve labels.
    stacked:
        Array of shape (n_labels, n_samples, T) with NaN for missing curves.
    """
    labels = sorted({k for curves in curves_list for k in curves.keys()})
    if not labels:
        return [], np.zeros((0, len(curves_list), 0), dtype=float)

    max_len = 0
    for curves in curves_list:
        for series in curves.values():
            if series is None:
                continue
            max_len = max(max_len, np.asarray(series).shape[-1])

    stacked = np.full((len(labels), len(curves_list), max_len), fill_value, dtype=float)
    for li, label in enumerate(labels):
        for i, curves in enumerate(curves_list):
            series = curves.get(label)
            if series is None:
                continue
            series = np.asarray(series, dtype=float)
            stacked[li, i, : series.shape[-1]] = series
    return labels, stacked


def save_plot_data(
    out_dir: Path | str,
    times: np.ndarray,
    plot_idx: np.ndarray,
    curves_list: Sequence[Mapping[str, np.ndarray]],
    error_list: Sequence[Mapping[str, np.ndarray]],
    y_true_fit: np.ndarray,
    y_pred_by_method: Mapping[str, np.ndarray],
    idx_fit: Optional[np.ndarray] = None,
    prefix: str = "",
    metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path]:
    """Persist plot data (arrays + metadata) to disk for reproducibility."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    curve_labels, curve_stack = stack_curves(curves_list)
    error_labels, error_stack = stack_curves(error_list)

    arrays: Dict[str, np.ndarray] = {
        "times": np.asarray(times),
        "plot_idx": np.asarray(plot_idx, dtype=int),
        "curve_stack": curve_stack,
        "error_stack": error_stack,
        "y_true_fit": np.asarray(y_true_fit, dtype=float),
    }
    if idx_fit is not None:
        arrays["idx_fit"] = np.asarray(idx_fit, dtype=int)

    for label, y_pred in y_pred_by_method.items():
        arrays[f"y_pred_{label}"] = np.asarray(y_pred, dtype=float)

    npz_path = out_dir / f"{prefix}plot_data.npz"
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "curve_labels": curve_labels,
        "error_labels": error_labels,
        "y_pred_labels": list(y_pred_by_method.keys()),
    }
    if metadata:
        meta.update(metadata)

    json_path = out_dir / f"{prefix}plot_data.json"
    with json_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2, sort_keys=True)

    return npz_path, json_path


def _filter_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter rows where either y_true or y_pred has NaNs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        return y_true[:0], y_pred[:0]
    mask = np.all(np.isfinite(y_true), axis=1) & np.all(np.isfinite(y_pred), axis=1)
    return y_true[mask], y_pred[mask]


def _method_from_curve_label(label: str) -> Optional[str]:
    if label.startswith("I_pred_"):
        return label[len("I_pred_"):]
    return None


def _method_score_from_row(row: Mapping[str, object]) -> Optional[float]:
    def _get(key: str) -> Optional[float]:
        value = row.get(key)
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):
            return None
        return v

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

    r2_beta = _get("r2_beta")
    r2_gamma = _get("r2_gamma")
    if r2_beta is not None or r2_gamma is not None:
        vals = [v for v in (r2_beta, r2_gamma) if v is not None]
        return -float(np.mean(vals)) if vals else None

    return None


def _rank_methods(
    y_true: np.ndarray,
    y_pred_by_method: Mapping[str, np.ndarray],
    results: Sequence[Mapping[str, object]],
) -> List[str]:
    scores: Dict[str, float] = {}
    for row in results:
        method = row.get("method")
        if not method:
            continue
        score = _method_score_from_row(row)
        if score is not None:
            scores[str(method)] = score

    if not scores:
        for method, y_pred in y_pred_by_method.items():
            y_true_f, y_pred_f = _filter_pairs(y_true, y_pred)
            if y_true_f.size == 0:
                scores[method] = np.inf
                continue
            err = np.abs(y_pred_f - y_true_f)
            scores[method] = float(np.mean(err))

    methods = list(y_pred_by_method.keys())
    return sorted(methods, key=lambda m: (scores.get(m, np.inf), str(m)))


def _filter_pred_by_method(
    y_pred_by_method: Mapping[str, np.ndarray],
    allowed: Sequence[str],
) -> Dict[str, np.ndarray]:
    allowed_set = set(allowed)
    return {k: v for k, v in y_pred_by_method.items() if k in allowed_set}


def _filter_curves_list(
    curves_list: Sequence[Mapping[str, np.ndarray]],
    allowed_methods: Sequence[str],
) -> List[Dict[str, np.ndarray]]:
    allowed = set(allowed_methods)
    filtered: List[Dict[str, np.ndarray]] = []
    for curves in curves_list:
        out: Dict[str, np.ndarray] = {}
        for label, series in curves.items():
            if label in ("I_true", "I_obs", "Y_obs"):
                out[label] = series
                continue
            method = _method_from_curve_label(label)
            if method and method in allowed:
                out[label] = series
        filtered.append(out)
    return filtered


def _filter_error_list(
    error_list: Sequence[Mapping[str, np.ndarray]],
    allowed_methods: Sequence[str],
) -> List[Dict[str, np.ndarray]]:
    allowed = set(allowed_methods)
    filtered: List[Dict[str, np.ndarray]] = []
    for errors in error_list:
        out = {label: series for label, series in errors.items() if label in allowed}
        filtered.append(out)
    return filtered


def _build_color_map(method_order: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    baselines, architectures = _split_method_labels(method_order)
    base_cmap = plt.get_cmap("Dark2")
    ml_cmap = plt.get_cmap("tab10")
    colors: Dict[str, Tuple[float, float, float, float]] = {}
    for i, method in enumerate(baselines):
        colors[method] = base_cmap(i % base_cmap.N)
    for i, method in enumerate(architectures):
        colors[method] = ml_cmap(i % ml_cmap.N)
    return colors


_BASELINE_DASHES = [
    (0, (6, 2)),
    (0, (3, 2, 1, 2)),
    (0, (1, 1)),
    (0, (4, 2, 1, 2, 1, 2)),
    (0, (8, 2, 2, 2)),
]
_ML_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", ">", "<"]


def _build_style_map(method_order: Sequence[str]) -> Dict[str, Dict[str, object]]:
    baselines, architectures = _split_method_labels(method_order)
    styles: Dict[str, Dict[str, object]] = {}
    for i, method in enumerate(baselines):
        styles[method] = {"linestyle": _BASELINE_DASHES[i % len(_BASELINE_DASHES)]}
    for i, method in enumerate(architectures):
        styles[method] = {"linestyle": "solid", "marker": _ML_MARKERS[i % len(_ML_MARKERS)]}
    return styles


def _outline_effects(linewidth: float) -> List[pe.AbstractPathEffect]:
    return [
        pe.Stroke(linewidth=linewidth + 1.6, foreground="white", alpha=0.85),
        pe.Normal(),
    ]


def _style_for_label(
    label: str,
    method: Optional[str],
    color_map: Mapping[str, object],
    style_map: Optional[Mapping[str, Mapping[str, object]]] = None,
    outline: bool = True,
) -> Dict[str, object]:
    if label == "I_true":
        style = {"color": "black", "linewidth": 2.2, "zorder": 2, "alpha": 0.85}
        if outline:
            style["path_effects"] = _outline_effects(style["linewidth"])
        return style
    if label in ("I_obs", "Y_obs"):
        return {"color": "gray", "linewidth": 1.2, "linestyle": ":", "alpha": 0.8, "zorder": 1}
    if method and method.startswith("baseline_"):
        style = {"color": color_map.get(method), "linewidth": 1.6, "alpha": 0.9, "zorder": 3}
        if style_map and method in style_map:
            style.update(style_map[method])
        else:
            style.setdefault("linestyle", "--")
        if outline:
            style["path_effects"] = _outline_effects(style["linewidth"])
        return style
    if method:
        style = {"color": color_map.get(method), "linewidth": 1.3, "alpha": 0.9, "zorder": 4}
        if style_map and method in style_map:
            style.update(style_map[method])
        else:
            style.setdefault("linestyle", "solid")
        if outline:
            style["path_effects"] = _outline_effects(style["linewidth"])
        return style
    return {"linewidth": 1.0, "alpha": 0.8}


def _label_to_method(label: str) -> Optional[str]:
    if label in ("I_true", "I_obs", "Y_obs"):
        return None
    method = _method_from_curve_label(label)
    return method if method is not None else label


def _legend_label(label: str) -> str:
    if label.startswith("I_pred_"):
        return label[len("I_pred_"):]
    return label


def _order_labels(labels: Sequence[str], method_order: Optional[Sequence[str]]) -> List[str]:
    labels = list(labels)
    if not method_order:
        return labels
    order = {method: idx for idx, method in enumerate(method_order)}

    def _key(label: str) -> Tuple[int, int, str]:
        if label == "I_true":
            return (-2, -1, label)
        if label in ("I_obs", "Y_obs"):
            return (-1, -1, label)
        method = _label_to_method(label)
        if method in order:
            return (0, order[method], label)
        return (1, len(order), label)

    return sorted(labels, key=_key)


def _order_methods(methods: Iterable[str], method_order: Optional[Sequence[str]]) -> List[str]:
    methods_set = {str(m) for m in methods}
    if not method_order:
        return sorted(methods_set)
    ordered: List[str] = []
    for method in method_order:
        if method in methods_set:
            ordered.append(method)
    for method in sorted(methods_set):
        if method not in ordered:
            ordered.append(method)
    return ordered


def plot_curve_comparison(
    times: np.ndarray,
    curves: Mapping[str, np.ndarray],
    title: Optional[str] = None,
    xlabel: str = "t",
    ylabel: str = "I(t)",
    ax: Optional[plt.Axes] = None,
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
    style_map: Optional[Mapping[str, Mapping[str, object]]] = None,
    show_markers: bool = True,
) -> plt.Axes:
    """Plot multiple curves on the same axis for comparison."""
    ax = ax or plt.gca()
    color_map = color_map or {}
    style_map = style_map or {}
    for label in _order_labels(curves.keys(), method_order):
        series = curves.get(label)
        if series is None:
            continue
        series = np.asarray(series)
        t = times[: series.shape[-1]]
        method = _label_to_method(label)
        style = _style_for_label(label, method, color_map, style_map=style_map)
        if show_markers and method and "marker" in style:
            markevery = max(1, int(series.shape[-1] // 12))
            style.setdefault("markersize", 4.2)
            style.setdefault("markeredgewidth", 0.6)
            style.setdefault("markeredgecolor", "white")
            style["markevery"] = markevery
        style.setdefault("solid_capstyle", "round")
        ax.plot(t, series, label=_legend_label(label), **style)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.3)
    return ax


def plot_curves_grid(
    times: np.ndarray,
    curves_list: Sequence[Mapping[str, np.ndarray]],
    n_cols: int = 3,
    title: Optional[str] = None,
    xlabel: str = "t",
    ylabel: str = "I(t)",
    legend: str = "global",
    figsize: Optional[Tuple[float, float]] = None,
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
    legend_ncol: Optional[int] = None,
    style_map: Optional[Mapping[str, Mapping[str, object]]] = None,
    show_markers: bool = True,
) -> plt.Figure:
    """Plot a grid of curve comparisons (one panel per sample)."""
    n = len(curves_list)
    if n == 0:
        fig, _ = plt.subplots(1, 1)
        if title:
            fig.suptitle(title)
        return fig

    n_cols = min(max(1, n_cols), n)
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 3.2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    legend_handles = None
    legend_labels = None

    for i, curves in enumerate(curves_list):
        ax = axes[i]
        plot_curve_comparison(
            times,
            curves,
            xlabel=xlabel,
            ylabel=ylabel,
            ax=ax,
            method_order=method_order,
            color_map=color_map,
            style_map=style_map,
            show_markers=show_markers,
        )
        ax.set_title(f"Sample {i + 1}")
        if legend == "all" or (legend == "first" and i == 0):
            ax.legend(fontsize=8)
        if legend == "global" and legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                legend_handles, legend_labels = handles, labels

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if legend == "global" and legend_handles:
        if legend_ncol is None:
            legend_ncol = min(len(legend_labels), max(1, n_cols + 1))
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=legend_ncol,
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.16)

    if title:
        fig.suptitle(title)
    return fig


def plot_param_scatter(
    y_true: np.ndarray,
    y_pred_by_label: Mapping[str, np.ndarray],
    param_names: Tuple[str, str] = ("beta", "gamma"),
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
    legend: str = "global",
) -> plt.Figure:
    """Scatter true vs predicted parameters for multiple methods."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels = _order_methods(y_pred_by_label.keys(), method_order)
    color_map = color_map or {}
    legend_handles = None
    legend_labels = None

    for p_idx, pname in enumerate(param_names):
        ax = axes[p_idx]
        y_min = float(np.min(y_true[:, p_idx]))
        y_max = float(np.max(y_true[:, p_idx]))
        ax.plot([y_min, y_max], [y_min, y_max], "k--", lw=1, label="perfect")
        for label in labels:
            y_pred = y_pred_by_label.get(label)
            if y_pred is None:
                continue
            y_pred = np.asarray(y_pred)
            y_true_f, y_pred_f = _filter_pairs(y_true, y_pred)
            if y_true_f.size == 0:
                continue
            ax.scatter(
                y_true_f[:, p_idx],
                y_pred_f[:, p_idx],
                s=12,
                alpha=0.6,
                label=label,
                color=color_map.get(label),
            )
        ax.set_xlabel(f"true {pname}")
        ax.set_ylabel(f"pred {pname}")
        ax.set_title(f"{pname} scatter")
        if legend == "all" or (legend == "first" and p_idx == 0):
            ax.legend(fontsize=8)
        if legend == "global" and legend_handles is None:
            handles, labels_out = ax.get_legend_handles_labels()
            if labels_out:
                legend_handles, legend_labels = handles, labels_out

    if title:
        fig.suptitle(title)
    if legend == "global" and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.18)
    return fig


def plot_param_error_hist(
    y_true: np.ndarray,
    y_pred_by_label: Mapping[str, np.ndarray],
    param_names: Tuple[str, str] = ("beta", "gamma"),
    bins: int = 40,
    kind: str = "abs",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
    legend: str = "global",
) -> plt.Figure:
    """Histogram of parameter errors for multiple methods."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels = _order_methods(y_pred_by_label.keys(), method_order)
    color_map = color_map or {}
    legend_handles = None
    legend_labels = None

    for p_idx, pname in enumerate(param_names):
        ax = axes[p_idx]
        for label in labels:
            y_pred = y_pred_by_label.get(label)
            if y_pred is None:
                continue
            y_pred = np.asarray(y_pred)
            y_true_f, y_pred_f = _filter_pairs(y_true, y_pred)
            if y_true_f.size == 0:
                continue
            err = y_pred_f[:, p_idx] - y_true_f[:, p_idx]
            if kind == "abs":
                err = np.abs(err)
            ax.hist(
                err,
                bins=bins,
                histtype="step",
                linewidth=1.4,
                alpha=0.9,
                label=label,
                color=color_map.get(label),
            )
        ax.set_title(f"{pname} error ({kind})")
        ax.set_xlabel("error")
        ax.set_ylabel("count")
        if legend == "all" or (legend == "first" and p_idx == 0):
            ax.legend(fontsize=8)
        if legend == "global" and legend_handles is None:
            handles, labels_out = ax.get_legend_handles_labels()
            if labels_out:
                legend_handles, legend_labels = handles, labels_out

    if title:
        fig.suptitle(title)
    if legend == "global" and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.18)
    return fig


def plot_param_error_box(
    y_true: np.ndarray,
    y_pred_by_label: Mapping[str, np.ndarray],
    param_names: Tuple[str, str] = ("beta", "gamma"),
    kind: str = "abs",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
) -> plt.Figure:
    """Boxplots of parameter errors for multiple methods."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels_ordered = _order_methods(y_pred_by_label.keys(), method_order)
    color_map = color_map or {}

    for p_idx, pname in enumerate(param_names):
        ax = axes[p_idx]
        data = []
        labels = []
        colors = []
        for label in labels_ordered:
            y_pred = y_pred_by_label.get(label)
            if y_pred is None:
                continue
            y_pred = np.asarray(y_pred)
            y_true_f, y_pred_f = _filter_pairs(y_true, y_pred)
            if y_true_f.size == 0:
                continue
            err = y_pred_f[:, p_idx] - y_true_f[:, p_idx]
            if kind == "abs":
                err = np.abs(err)
            data.append(err)
            labels.append(label)
            colors.append(color_map.get(label))
        if data:
            box = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
            for patch, color in zip(box["boxes"], colors):
                if color is None:
                    continue
                patch.set_facecolor(color)
                patch.set_alpha(0.4)
        ax.set_title(f"{pname} error ({kind})")
        ax.set_ylabel("error")
        ax.tick_params(axis="x", rotation=45)

    if title:
        fig.suptitle(title)
    return fig


def plot_param_error_cdf(
    y_true: np.ndarray,
    y_pred_by_label: Mapping[str, np.ndarray],
    param_names: Tuple[str, str] = ("beta", "gamma"),
    kind: str = "abs",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
    legend: str = "global",
) -> plt.Figure:
    """CDF of parameter errors for multiple methods."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels = _order_methods(y_pred_by_label.keys(), method_order)
    color_map = color_map or {}
    legend_handles = None
    legend_labels = None

    for p_idx, pname in enumerate(param_names):
        ax = axes[p_idx]
        for label in labels:
            y_pred = y_pred_by_label.get(label)
            if y_pred is None:
                continue
            y_pred = np.asarray(y_pred)
            y_true_f, y_pred_f = _filter_pairs(y_true, y_pred)
            if y_true_f.size == 0:
                continue
            err = y_pred_f[:, p_idx] - y_true_f[:, p_idx]
            if kind == "abs":
                err = np.abs(err)
            err = np.sort(err)
            cdf = np.arange(1, err.size + 1) / err.size
            ax.plot(err, cdf, label=label, color=color_map.get(label))
        ax.set_title(f"{pname} error CDF ({kind})")
        ax.set_xlabel("error")
        ax.set_ylabel("cdf")
        if legend == "all" or (legend == "first" and p_idx == 0):
            ax.legend(fontsize=8)
        if legend == "global" and legend_handles is None:
            handles, labels_out = ax.get_legend_handles_labels()
            if labels_out:
                legend_handles, legend_labels = handles, labels_out

    if title:
        fig.suptitle(title)
    if legend == "global" and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.18)
    return fig


def plot_curve_mae_hist(
    error_list: Sequence[Mapping[str, np.ndarray]],
    bins: int = 30,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
) -> plt.Figure:
    """Histogram of per-curve MAE from error curves."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if not error_list:
        if title:
            fig.suptitle(title)
        return fig

    by_method: Dict[str, List[float]] = {}
    for errors in error_list:
        for label, series in errors.items():
            if series is None:
                continue
            val = float(np.nanmean(np.asarray(series)))
            if np.isfinite(val):
                by_method.setdefault(label, []).append(val)

    labels = _order_methods(by_method.keys(), method_order)
    color_map = color_map or {}
    for label in labels:
        values = by_method.get(label, [])
        if not values:
            continue
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.4,
            alpha=0.9,
            label=label,
            color=color_map.get(label),
        )

    ax.set_xlabel("MAE per curve")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


def plot_curve_quantiles(
    times: np.ndarray,
    curves_list: Sequence[Mapping[str, np.ndarray]],
    quantiles: Tuple[float, float, float] = (0.25, 0.5, 0.75),
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4),
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
    style_map: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> plt.Figure:
    """Plot median and quantile bands for each curve label across samples."""
    labels, stacked = stack_curves(curves_list)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if not labels:
        if title:
            fig.suptitle(title)
        return fig

    ordered_labels = _order_labels(labels, method_order)
    color_map = color_map or {}
    style_map = style_map or {}
    label_to_idx = {label: li for li, label in enumerate(labels)}
    for label in ordered_labels:
        li = label_to_idx[label]
        series_stack = stacked[li]
        q_vals = np.nanquantile(series_stack, quantiles, axis=0)
        t = times[: q_vals.shape[-1]]
        method = _label_to_method(label)
        style = _style_for_label(label, method, color_map, style_map=style_map)
        ax.plot(t, q_vals[1], label=_legend_label(label), **style)
        fill_color = style.get("color")
        if fill_color is None:
            ax.fill_between(t, q_vals[0], q_vals[2], alpha=0.2)
        else:
            ax.fill_between(t, q_vals[0], q_vals[2], alpha=0.2, color=fill_color)

    ax.set_xlabel("t")
    ax.set_ylabel("I(t)")
    ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


def plot_metric_bars(
    rows: Sequence[Mapping[str, object]],
    metric_keys: Optional[Sequence[str]] = None,
    label_key: str = "method",
    title: Optional[str] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    method_order: Optional[Sequence[str]] = None,
    color_map: Optional[Mapping[str, object]] = None,
) -> plt.Figure:
    """Plot bar charts for metrics across methods in a single run."""
    rows = list(rows)
    if not rows:
        fig, _ = plt.subplots(1, 1)
        if title:
            fig.suptitle(title)
        return fig

    if metric_keys is None:
        metric_keys = [k for k in rows[0].keys() if k.startswith(("mae_", "rmse_", "r2_"))]
    metric_keys = list(metric_keys)

    if method_order:
        order = {method: idx for idx, method in enumerate(method_order)}
        rows = sorted(
            rows,
            key=lambda row: (order.get(str(row.get(label_key, "")), len(order)), str(row.get(label_key, ""))),
        )
    labels = [str(row.get(label_key, "")) for row in rows]
    color_map = color_map or {}
    n = len(metric_keys)
    n_cols = min(max(1, n_cols), max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 3.2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        values = [float(row.get(key, np.nan)) for row in rows]
        pairs = list(zip(labels, values))
        if key.startswith("r2_"):
            pairs.sort(key=lambda pair: (not np.isfinite(pair[1]), -pair[1]))
        else:
            pairs.sort(key=lambda pair: (not np.isfinite(pair[1]), pair[1]))
        labels_sorted = [p[0] for p in pairs]
        values_sorted = [p[1] for p in pairs]
        colors = [color_map.get(label) for label in labels_sorted]
        ax.barh(labels_sorted, values_sorted, color=colors)
        ax.set_title(key)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title)
    return fig


def _split_method_labels(labels: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split method labels into baselines and ML architectures by convention."""
    baselines: List[str] = []
    architectures: List[str] = []
    for label in labels:
        label = str(label)
        if label.startswith("baseline_"):
            baselines.append(label)
        else:
            architectures.append(label)
    return baselines, architectures


def plot_architectures_summary(
    labels_in_file: Sequence[str],
    title: str,
    meta: Optional[Mapping[str, object]] = None,
    labels_selected: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (8.5, 5.5),
) -> plt.Figure:
    """Render a text-only figure listing which methods/architectures were used."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")

    labels_in_file = [str(x) for x in labels_in_file]
    labels_selected = [str(x) for x in labels_selected] if labels_selected else []

    baselines, architectures = _split_method_labels(labels_in_file)

    lines: List[str] = [title, ""]

    if meta:
        context_keys = [
            "exp",
            "scenario",
            "noise",
            "train_mode",
            "rho",
            "k",
            "estimate_rho",
            "window_days",
            "downsample",
            "normalize",
            "run_baseline",
            "seed",
            "n_starts",
            "max_test",
            "test_size",
            "limit",
        ]
        ctx: List[str] = []
        for key in context_keys:
            if key in meta and meta.get(key) is not None:
                ctx.append(f"{key}={meta.get(key)}")
        if ctx:
            lines.append("Context:")
            lines.extend(textwrap.wrap(", ".join(ctx), width=95))
            lines.append("")

    lines.append("Baselines:")
    lines.append(f"- {', '.join(baselines) if baselines else '(none)'}")
    lines.append("")
    lines.append(f"ML architectures ({len(architectures)}):")
    if architectures:
        for label in architectures:
            lines.append(f"- {label}")
    else:
        lines.append("- (none)")

    if labels_selected and set(labels_selected) != set(labels_in_file):
        lines.append("")
        lines.append("Selected subset (plotting):")
        lines.extend(textwrap.wrap(", ".join(sorted(labels_selected)), width=95))

    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )

    return fig


def save_experiment_figures(
    plot_dir: Path | str,
    times: np.ndarray,
    curves_list: Sequence[Mapping[str, np.ndarray]],
    error_list: Sequence[Mapping[str, np.ndarray]],
    y_true_fit: np.ndarray,
    y_pred_by_method: Mapping[str, np.ndarray],
    results: Sequence[Mapping[str, object]],
    title_prefix: str,
    max_ml_methods: Optional[int] = 4,
    max_baseline_methods: Optional[int] = None,
    legend: str = "global",
) -> Dict[str, Path]:
    """Save the standard set of figures for an experiment.

    By default, plots include all baselines plus the top-k ML methods (ranked by
    parameter error) to keep figures readable; override with max_* arguments.
    """
    plot_dir = Path(plot_dir)
    ensure_dir(plot_dir)

    paths: Dict[str, Path] = {}
    method_order = _rank_methods(y_true_fit, y_pred_by_method, results)
    baselines, architectures = _split_method_labels(method_order)
    if max_baseline_methods is None or max_baseline_methods <= 0:
        baselines_sel = baselines
    else:
        baselines_sel = baselines[: max_baseline_methods]
    if max_ml_methods is None or max_ml_methods <= 0:
        architectures_sel = architectures
    else:
        architectures_sel = architectures[: max_ml_methods]
    selected_methods = baselines_sel + architectures_sel
    selected_set = set(selected_methods)

    if selected_methods:
        curves_plot = _filter_curves_list(curves_list, selected_methods)
        error_plot = _filter_error_list(error_list, selected_methods)
        y_pred_plot = _filter_pred_by_method(y_pred_by_method, selected_methods)
        results_plot = [row for row in results if row.get("method") in selected_set]
    else:
        curves_plot = list(curves_list)
        error_plot = list(error_list)
        y_pred_plot = dict(y_pred_by_method)
        results_plot = list(results)

    method_order_plot = _order_methods(y_pred_plot.keys(), selected_methods or method_order)
    color_map = _build_color_map(method_order_plot)
    style_map = _build_style_map(method_order_plot)
    style_map = _build_style_map(method_order_plot)

    fig = plot_curves_grid(
        times,
        curves_plot,
        n_cols=3,
        title=f"{title_prefix}: curve comparison",
        ylabel="I(t)",
        legend=legend,
        method_order=method_order_plot,
        color_map=color_map,
        style_map=style_map,
        show_markers=True,
    )
    paths["curves_comparison"] = save_figure(fig, plot_dir / "curves_comparison.png")

    fig = plot_curves_grid(
        times,
        error_plot,
        n_cols=3,
        title=f"{title_prefix}: absolute error curves",
        ylabel="|I_pred - I_true|",
        legend=legend,
        method_order=method_order_plot,
        color_map=color_map,
        style_map=style_map,
        show_markers=False,
    )
    paths["error_curves"] = save_figure(fig, plot_dir / "error_curves.png")

    fig = plot_param_scatter(
        y_true_fit,
        y_pred_plot,
        title=f"{title_prefix}: parameter scatter",
        method_order=method_order_plot,
        color_map=color_map,
    )
    paths["param_scatter"] = save_figure(fig, plot_dir / "param_scatter.png")

    fig = plot_param_error_hist(
        y_true_fit,
        y_pred_plot,
        title=f"{title_prefix}: parameter error distributions",
        method_order=method_order_plot,
        color_map=color_map,
    )
    paths["param_error_hist"] = save_figure(fig, plot_dir / "param_error_hist.png")

    fig = plot_metric_bars(
        results_plot,
        title=f"{title_prefix}: metrics by method",
        method_order=method_order_plot,
        color_map=color_map,
    )
    paths["metrics_comparison"] = save_figure(fig, plot_dir / "metrics_comparison.png")

    fig = plot_architectures_summary(
        labels_in_file=list(y_pred_by_method.keys()),
        labels_selected=method_order_plot,
        title=f"{title_prefix}: architectures used",
    )
    paths["architectures"] = save_figure(fig, plot_dir / "architectures.png")

    return paths


def _parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_out_dir(pred_path: Path, out_dir: Optional[str]) -> Path:
    if out_dir:
        return Path(out_dir)
    if pred_path.is_dir():
        return pred_path
    if pred_path.parent.name == "figures":
        return pred_path.parent
    return pred_path.parent / "figures"


def _load_predictions(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    arrays = np.load(path)
    meta: Dict[str, object] = {}
    json_path = path.with_suffix(".json")
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    return dict(arrays), meta


def _load_metrics_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, object] = {}
            for key, value in row.items():
                if value is None or value == "":
                    out[key] = np.nan
                    continue
                if key in ("method", "scenario", "run_dir", "exp"):
                    out[key] = value
                    continue
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
            rows.append(out)
    return rows


def _select_indices(
    n_samples: int,
    n_curves: int,
    seed: int,
    curve_idx: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if curve_idx:
        idx = [i for i in curve_idx if 0 <= i < n_samples]
        return np.asarray(sorted(set(idx)), dtype=int)
    if n_curves <= 0 or n_curves >= n_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=n_curves, replace=False))


def _title_from_meta(meta: Dict[str, object]) -> str:
    exp = str(meta.get("exp", "")).lower()
    if exp == "exp1":
        return f"Exp1 (noise={meta.get('noise')}, train={meta.get('train_mode')})"
    if exp == "exp2":
        return f"Exp2 (window={meta.get('window_days')}, downsample={meta.get('downsample')})"
    if exp == "exp0":
        return "Exp0"
    if exp == "exp3":
        return "Exp3"
    if exp == "exp4":
        return "Exp4 (variable init)"
    return "Experiment"


def _extract_params(meta: Dict[str, object], times: np.ndarray) -> Tuple[float, float, float, float, float]:
    t0 = float(meta.get("t0", times[0] if times.size else DEFAULTS.t0))
    dt = meta.get("dt")
    if dt is None:
        if times.size > 1:
            dt = float(np.median(np.diff(times)))
        else:
            dt = float(DEFAULTS.dt)
    dt = float(dt)
    s0 = float(meta.get("s0", DEFAULTS.s0))
    i0 = float(meta.get("i0", DEFAULTS.i0))
    r0 = float(meta.get("r0", DEFAULTS.r0))
    return t0, dt, s0, i0, r0


def _build_curves(
    times: np.ndarray,
    i_true: np.ndarray,
    y_pred_by_method: Mapping[str, np.ndarray],
    plot_idx: np.ndarray,
    meta: Dict[str, object],
    include_obs: bool = False,
    i_obs: Optional[np.ndarray] = None,
    init_states: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    from src.sir.simulate import simulate_sir

    t0, dt, s0, i0, r0 = _extract_params(meta, times)
    t1 = t0 + dt * (i_true.shape[1] - 1)

    curves_list: List[Dict[str, np.ndarray]] = []
    error_list: List[Dict[str, np.ndarray]] = []

    init_states_arr: Optional[np.ndarray] = None
    if init_states is not None:
        init_states_arr = np.asarray(init_states, dtype=float)

    for idx in plot_idx:
        curves: Dict[str, np.ndarray] = {"I_true": i_true[idx]}
        if include_obs and i_obs is not None:
            curves["I_obs"] = i_obs[idx]
        errors: Dict[str, np.ndarray] = {}

        if init_states_arr is not None and idx < init_states_arr.shape[0]:
            s0_i, i0_i, r0_i = init_states_arr[idx][:3]
        else:
            s0_i, i0_i, r0_i = s0, i0, r0

        for label, y_pred in y_pred_by_method.items():
            if y_pred is None:
                continue
            params = np.asarray(y_pred[idx], dtype=float)
            if params.shape[-1] < 2 or not np.all(np.isfinite(params[:2])):
                continue
            I_pred = simulate_sir(
                params[0],
                params[1],
                s0=s0_i,
                i0=i0_i,
                r0=r0_i,
                t0=t0,
                t1=t1,
                dt=dt,
            )
            curves[f"I_pred_{label}"] = I_pred
            errors[label] = np.abs(I_pred - i_true[idx][: I_pred.shape[-1]])

        curves_list.append(curves)
        error_list.append(errors)

    return curves_list, error_list


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_pred_by_method: Mapping[str, np.ndarray],
    meta: Dict[str, object],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    scenario = str(meta.get("scenario", ""))
    if meta.get("exp") == "exp1":
        scenario = f"train_{meta.get('train_mode')}_test_{meta.get('noise')}"
    for label, y_pred in y_pred_by_method.items():
        y_true_f, y_pred_f = _filter_pairs(y_true, y_pred)
        if y_true_f.size == 0:
            continue
        metrics = per_param_metrics(y_true_f, y_pred_f)
        metrics["method"] = label
        if scenario:
            metrics["scenario"] = scenario
        results.append(metrics)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from experiment artifacts.")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.npz saved by --save-predictions.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Optional metrics.csv to plot (defaults to recompute from predictions).",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--plots",
        type=str,
        default="curves,errors,param_scatter,param_error_hist,metric_bars",
        help="Comma-separated list or 'all'.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to include (default: all).",
    )
    parser.add_argument("--n-curves", type=int, default=9)
    parser.add_argument(
        "--curve-idx",
        type=int,
        nargs="*",
        default=None,
        help="Explicit curve indices (overrides --n-curves).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--legend", type=str, default="global", choices=["global", "first", "all", "none"])
    parser.add_argument("--n-cols", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--include-obs", action="store_true")
    parser.add_argument(
        "--plot-max-ml",
        type=int,
        default=4,
        help="Max ML methods to include in plots (<=0 keeps all).",
    )
    parser.add_argument(
        "--plot-max-baseline",
        type=int,
        default=0,
        help="Max baseline methods to include in plots (<=0 keeps all).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pred_path = Path(args.predictions)
    arrays, meta = _load_predictions(pred_path)

    required = ["times", "i_true", "y_true"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Missing required arrays in predictions.npz: {missing}")

    times = np.asarray(arrays["times"], dtype=float)
    i_true = np.asarray(arrays["i_true"])
    y_true = np.asarray(arrays["y_true"])
    i_obs = arrays.get("i_obs")
    init_states = None
    if "s0" in arrays and "i0" in arrays and "r0" in arrays:
        init_states = np.column_stack([arrays["s0"], arrays["i0"], arrays["r0"]])

    y_pred_by_method: Dict[str, np.ndarray] = {}
    labels = meta.get("y_pred_labels")
    if labels is None:
        labels = sorted(k[len("y_pred_"):] for k in arrays.keys() if k.startswith("y_pred_"))
    for label in labels:
        key = f"y_pred_{label}"
        if key in arrays:
            y_pred_by_method[str(label)] = np.asarray(arrays[key])

    methods = _parse_list(args.methods)
    if methods:
        y_pred_by_method = {k: v for k, v in y_pred_by_method.items() if k in methods}

    out_dir = _resolve_out_dir(pred_path, args.out_dir)
    ensure_dir(out_dir)

    title_prefix = args.title if args.title else _title_from_meta(meta)
    plots = _parse_list(args.plots)
    if "all" in plots:
        plots = [
            "curves",
            "errors",
            "architectures",
            "param_scatter",
            "param_error_hist",
            "param_error_box",
            "param_error_cdf",
            "metric_bars",
            "curve_mae_hist",
            "curve_quantiles",
        ]

    metrics_rows: List[Dict[str, object]]
    if args.metrics:
        metrics_rows = _load_metrics_csv(Path(args.metrics))
        if methods:
            metrics_rows = [row for row in metrics_rows if row.get("method") in methods]
    else:
        metrics_rows = _metrics_from_predictions(y_true, y_pred_by_method, meta)

    method_order = _rank_methods(y_true, y_pred_by_method, metrics_rows)
    baselines, architectures = _split_method_labels(method_order)
    if args.plot_max_baseline and args.plot_max_baseline > 0:
        baselines = baselines[: args.plot_max_baseline]
    if args.plot_max_ml and args.plot_max_ml > 0:
        architectures = architectures[: args.plot_max_ml]
    selected_methods = baselines + architectures
    selected_set = set(selected_methods)
    if selected_methods:
        y_pred_plot = _filter_pred_by_method(y_pred_by_method, selected_methods)
        metrics_rows = [row for row in metrics_rows if row.get("method") in selected_set]
    else:
        y_pred_plot = dict(y_pred_by_method)
    method_order_plot = _order_methods(y_pred_plot.keys(), selected_methods or method_order)
    color_map = _build_color_map(method_order_plot)

    curve_plots = {"curves", "errors", "curve_mae_hist", "curve_quantiles"}
    if plots and curve_plots.intersection(plots):
        plot_idx = _select_indices(i_true.shape[0], args.n_curves, args.seed, args.curve_idx)
        curves_list, error_list = _build_curves(
            times,
            i_true,
            y_pred_plot,
            plot_idx,
            meta,
            include_obs=args.include_obs,
            i_obs=i_obs if args.include_obs else None,
            init_states=init_states,
        )
    else:
        curves_list = []
        error_list = []
    if selected_methods:
        curves_list = _filter_curves_list(curves_list, selected_methods)
        error_list = _filter_error_list(error_list, selected_methods)

    if "curves" in plots:
        fig = plot_curves_grid(
            times,
            curves_list,
            n_cols=args.n_cols,
            title=f"{title_prefix}: curve comparison",
            ylabel="I(t)",
            legend=args.legend,
            method_order=method_order_plot,
            color_map=color_map,
            style_map=style_map,
            show_markers=True,
        )
        save_figure(fig, out_dir / f"{args.prefix}curves_comparison.png", dpi=args.dpi)

    if "errors" in plots:
        fig = plot_curves_grid(
            times,
            error_list,
            n_cols=args.n_cols,
            title=f"{title_prefix}: absolute error curves",
            ylabel="|I_pred - I_true|",
            legend=args.legend,
            method_order=method_order_plot,
            color_map=color_map,
            style_map=style_map,
            show_markers=False,
        )
        save_figure(fig, out_dir / f"{args.prefix}error_curves.png", dpi=args.dpi)

    if "param_scatter" in plots:
        fig = plot_param_scatter(
            y_true,
            y_pred_plot,
            title=f"{title_prefix}: parameter scatter",
            method_order=method_order_plot,
            color_map=color_map,
        )
        save_figure(fig, out_dir / f"{args.prefix}param_scatter.png", dpi=args.dpi)

    if "param_error_hist" in plots:
        fig = plot_param_error_hist(
            y_true,
            y_pred_plot,
            title=f"{title_prefix}: parameter error distributions",
            method_order=method_order_plot,
            color_map=color_map,
        )
        save_figure(fig, out_dir / f"{args.prefix}param_error_hist.png", dpi=args.dpi)

    if "param_error_box" in plots:
        fig = plot_param_error_box(
            y_true,
            y_pred_plot,
            title=f"{title_prefix}: parameter error boxplots",
            method_order=method_order_plot,
            color_map=color_map,
        )
        save_figure(fig, out_dir / f"{args.prefix}param_error_box.png", dpi=args.dpi)

    if "param_error_cdf" in plots:
        fig = plot_param_error_cdf(
            y_true,
            y_pred_plot,
            title=f"{title_prefix}: parameter error CDF",
            method_order=method_order_plot,
            color_map=color_map,
        )
        save_figure(fig, out_dir / f"{args.prefix}param_error_cdf.png", dpi=args.dpi)

    if "metric_bars" in plots:
        fig = plot_metric_bars(
            metrics_rows,
            title=f"{title_prefix}: metrics by method",
            method_order=method_order_plot,
            color_map=color_map,
        )
        save_figure(fig, out_dir / f"{args.prefix}metrics_comparison.png", dpi=args.dpi)

    if "architectures" in plots:
        labels_in_file = meta.get("y_pred_labels")
        if labels_in_file is None:
            labels_in_file = sorted(y_pred_by_method.keys())
        fig = plot_architectures_summary(
            labels_in_file=list(labels_in_file),
            labels_selected=method_order_plot,
            meta=meta,
            title=f"{title_prefix}: architectures used",
        )
        save_figure(fig, out_dir / f"{args.prefix}architectures.png", dpi=args.dpi)

    if "curve_mae_hist" in plots:
        fig = plot_curve_mae_hist(
            error_list,
            title=f"{title_prefix}: per-curve MAE histogram",
            method_order=method_order_plot,
            color_map=color_map,
        )
        save_figure(fig, out_dir / f"{args.prefix}curve_mae_hist.png", dpi=args.dpi)

    if "curve_quantiles" in plots:
        fig = plot_curve_quantiles(
            times,
            curves_list,
            title=f"{title_prefix}: curve quantiles",
            method_order=method_order_plot,
            color_map=color_map,
            style_map=_build_style_map(method_order_plot),
        )
        save_figure(fig, out_dir / f"{args.prefix}curve_quantiles.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
