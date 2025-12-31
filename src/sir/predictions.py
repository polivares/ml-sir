"""Persist per-curve predictions and inputs for later analysis.

This module stores the minimal artifacts needed to re-plot or audit a run
without re-running the experiment: time grid, input I(t), true parameters,
predicted parameters per method, and (optionally) sir.pkl-like trajectories
simulated from those predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple
import json
import pickle

import numpy as np

from src.sir.io import ensure_dir
from src.sir.simulate import simulate_sir


def save_predictions(
    out_dir: Path | str,
    times: np.ndarray,
    i_true: np.ndarray,
    y_true: np.ndarray,
    y_pred_by_method: Mapping[str, np.ndarray],
    idx_test: Optional[np.ndarray] = None,
    idx_fit: Optional[np.ndarray] = None,
    i_obs: Optional[np.ndarray] = None,
    prefix: str = "",
    metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path]:
    """Save predictions + inputs for a run as NPZ + JSON metadata."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    arrays: Dict[str, np.ndarray] = {
        "times": np.asarray(times, dtype=float),
        "i_true": np.asarray(i_true, dtype=float),
        "y_true": np.asarray(y_true, dtype=float),
    }
    if idx_test is not None:
        arrays["idx_test"] = np.asarray(idx_test, dtype=int)
    if idx_fit is not None:
        arrays["idx_fit"] = np.asarray(idx_fit, dtype=int)
    if i_obs is not None:
        arrays["i_obs"] = np.asarray(i_obs, dtype=float)

    for label, y_pred in y_pred_by_method.items():
        arrays[f"y_pred_{label}"] = np.asarray(y_pred, dtype=float)

    npz_path = out_dir / f"{prefix}predictions.npz"
    np.savez_compressed(npz_path, **arrays)

    meta: Dict[str, object] = {
        "y_pred_labels": list(y_pred_by_method.keys()),
    }
    if metadata:
        meta.update(metadata)

    json_path = out_dir / f"{prefix}predictions.json"
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return npz_path, json_path


def save_predicted_sir(
    out_dir: Path | str,
    times: np.ndarray,
    y_pred_by_method: Mapping[str, np.ndarray],
    s0: float,
    i0: float,
    r0: float,
    t0: float,
    dt: float,
    y_true: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, Path]:
    """Save SIR trajectories simulated from predicted parameters.

    Each output file is a list of tuples `(outputs, times, params)` that mimics
    the `sir.pkl` structure (outputs shape: T x 3 with [S, I, R]).
    If y_true is provided, an additional `predicted_sir_ground_truth.pkl` file
    is written using the true parameters.
    """
    out_dir = Path(out_dir) / "predicted_sir"
    ensure_dir(out_dir)

    times = np.asarray(times, dtype=float)
    t1 = t0 + dt * (times.shape[0] - 1)

    def _empty_outputs() -> np.ndarray:
        return np.full((times.shape[0], 3), np.nan, dtype=float)

    def _build_records(params_array: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        records = []
        for params in params_array:
            params = np.asarray(params, dtype=float)
            params_out = params[:2] if params.shape[0] >= 2 else np.array([np.nan, np.nan])
            if params.shape[0] < 2 or not np.all(np.isfinite(params[:2])):
                outputs = _empty_outputs()
                records.append((outputs, times, params_out))
                continue
            try:
                sim_times, outputs = simulate_sir(
                    params_out[0],
                    params_out[1],
                    s0=s0,
                    i0=i0,
                    r0=r0,
                    t0=t0,
                    t1=t1,
                    dt=dt,
                    return_full=True,
                )
                outputs = np.asarray(outputs, dtype=float)
                if outputs.shape[0] != times.shape[0]:
                    padded = _empty_outputs()
                    max_len = min(outputs.shape[0], times.shape[0])
                    padded[:max_len] = outputs[:max_len]
                    outputs = padded
                records.append((outputs, times, params_out))
            except Exception:
                outputs = _empty_outputs()
                records.append((outputs, times, params_out))
        return records

    paths: Dict[str, Path] = {}
    for method, y_pred in y_pred_by_method.items():
        y_pred = np.asarray(y_pred, dtype=float)
        records = _build_records(y_pred)
        safe_name = str(method).replace("/", "_")
        path = out_dir / f"{prefix}predicted_sir_{safe_name}.pkl"
        with path.open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        paths[str(method)] = path

    if y_true is not None:
        y_true = np.asarray(y_true, dtype=float)
        records = _build_records(y_true)
        path = out_dir / f"{prefix}predicted_sir_ground_truth.pkl"
        with path.open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        paths["ground_truth"] = path

    manifest = out_dir / f"{prefix}predicted_sir_manifest.json"
    manifest.write_text(
        json.dumps({k: str(v) for k, v in paths.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["manifest"] = manifest

    return paths
