"""Dataset helpers for SIR benchmarks.

Loads sir.pkl, builds X/y arrays, and creates reproducible splits."""


from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import pickle

import numpy as np

from .config import DEFAULTS


def load_sir_pkl(
    path: Union[Path, str] = DEFAULTS.data_path,
    limit: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """Load sir.pkl and optionally subsample to limit entries."""
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    # Expected schema: list of (outputs, times, params) from summer runs.

    if limit is None or limit >= len(data):
        return data

    rng = rng or np.random.default_rng(DEFAULTS.seed)
    # Subsample without replacement for smaller experiments.
    idx = rng.choice(len(data), size=limit, replace=False)
    return [data[i] for i in idx]


def build_Xy_I_only(
    data: Iterable,
    normalize: Optional[str] = None,
    dtype: np.dtype = np.float32,
    return_times: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build X=I(t) and y=(beta,gamma) from simulation list."""
    X_list = []
    y_list = []
    times_ref = None

    for outputs, times, params in data:
        # outputs is SIR over time; we use only the I(t) column as input.
        # Use only I(t) as input features.
        I = outputs[:, 1].astype(dtype, copy=False)
        if normalize == "population":
            pop = float(outputs[0].sum())
            I = I / (pop if pop > 0 else 1.0)
        elif normalize == "max":
            max_val = float(np.max(I))
            I = I / (max_val + 1e-8)
        X_list.append(I)
        # params is assumed to be [beta, gamma].
        y_list.append(params)
        if times_ref is None:
            # Keep a reference time grid for plotting.
            times_ref = np.asarray(times)

    X = np.asarray(X_list, dtype=dtype)
    y = np.asarray(y_list, dtype=dtype)

    if return_times:
        return X, y, times_ref
    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = DEFAULTS.test_size,
    val_size: float = DEFAULTS.val_size,
    rng: Optional[np.random.Generator] = None,
    shuffle: bool = True,
    return_indices: bool = False,
) -> Dict[str, np.ndarray]:
    """Split arrays into train/val/test with reproducible shuffling."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0,1)")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0,1)")

    n = X.shape[0]
    rng = rng or np.random.default_rng(DEFAULTS.seed)
    indices = np.arange(n)
    if shuffle:
        # Deterministic shuffle for reproducibility.
        rng.shuffle(indices)

    # Convert fractions to counts with sane bounds.
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_test = min(max(n_test, 1), n - 1)
    n_val = min(max(n_val, 0), n - n_test)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    splits = {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
    }

    if return_indices:
        splits.update({
            "idx_train": train_idx,
            "idx_val": val_idx,
            "idx_test": test_idx,
        })

    return splits


def normalize_series(
    X: np.ndarray,
    method: Optional[str] = None,
    population: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize time series by max or population."""
    if method is None:
        return X
    X = np.asarray(X, dtype=float)
    if method == "max":
        # Per-sample max scaling to stabilize training.
        scale = np.max(X, axis=-1, keepdims=True) + 1e-8
        return X / scale
    if method == "population":
        if population is None:
            raise ValueError("population is required for population normalization")
        # Normalize by population size when available.
        pop = np.asarray(population, dtype=float).reshape(-1, 1)
        return X / (pop + 1e-8)
    raise ValueError("method must be None, 'max', or 'population'")
