"""Persist per-curve predictions and inputs for later analysis.

This module stores the minimal artifacts needed to re-plot or audit a run
without re-running the experiment: time grid, input I(t), true parameters,
and predicted parameters per method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple
import json

import numpy as np

from src.sir.io import ensure_dir


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
