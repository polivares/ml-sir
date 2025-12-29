"""Observation models and preprocessing utilities.

Implements Poisson and Negative Binomial observation models plus utilities
for windowing, downsampling, and missingness with simple imputation.
Used to build more realistic observation pipelines in Exp1/Exp2 and beyond.
"""


from typing import Optional, Tuple, Union
import numpy as np


def _ensure_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    # Use a default generator when none is provided.
    return rng or np.random.default_rng()


def observe_poisson(
    I: np.ndarray, rho: float, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Apply Poisson observation model with reporting rate rho."""
    rng = _ensure_rng(rng)
    # rho scales the latent infections into observed counts.
    # Clamp to avoid negative/invalid rates.
    lam = np.clip(rho * I, 0.0, None)
    return rng.poisson(lam).astype(np.int64, copy=False)


def observe_negbin(
    I: np.ndarray, rho: float, k: float, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Apply Negative Binomial observation model.

    Parameterization: mean mu = rho * I, variance = mu + mu^2 / k.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rng = _ensure_rng(rng)
    # k controls over-dispersion: lower k => higher variance.
    mu = np.clip(rho * I, 0.0, None)
    p = k / (k + mu)
    # When mu=0 -> p=1 -> sample 0; ensure stable.
    return rng.negative_binomial(k, p).astype(np.int64, copy=False)


def apply_downsample(x: np.ndarray, step: int) -> np.ndarray:
    """Downsample along the last axis by taking every `step` point."""
    if step <= 0:
        raise ValueError("step must be positive")
    return x[..., ::step]


def apply_window(x: np.ndarray, T: float, dt: float) -> np.ndarray:
    """Keep only the first T days (based on dt)."""
    if T <= 0:
        raise ValueError("T must be positive")
    n = int(round(T / dt)) + 1
    return x[..., :n]


def apply_missing(
    x: np.ndarray,
    p: float,
    rng: Optional[np.random.Generator] = None,
    method: Optional[str] = "ffill",
    return_mask: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Randomly mask values with probability p and optionally impute.

    method: None, "ffill", or "interp".
    """
    if not 0 <= p < 1:
        raise ValueError("p must be in [0,1)")
    rng = _ensure_rng(rng)

    x = np.asarray(x)
    # Boolean mask of missing entries; True means "missing".
    mask = rng.random(x.shape) < p
    x_missing = x.astype(float, copy=True)
    x_missing[mask] = np.nan

    if method is None:
        out = x_missing
    elif method == "ffill":
        out = _ffill(x_missing)
    elif method == "interp":
        out = _interp(x_missing)
    else:
        raise ValueError("method must be None, 'ffill', or 'interp'")

    if return_mask:
        return out, mask
    return out


def _ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along last axis, then backfill if needed."""
    out = x.copy()
    if out.ndim == 1:
        # Promote to 2D for unified logic.
        out = out[None, :]
        squeeze = True
    else:
        squeeze = False

    for i in range(out.shape[0]):
        row = out[i]
        isnan = np.isnan(row)
        if np.all(isnan):
            row[:] = 0.0
            continue
        # Forward fill interior gaps.
        for j in range(1, row.size):
            if np.isnan(row[j]) and not np.isnan(row[j - 1]):
                row[j] = row[j - 1]
        # Backfill leading NaNs from first valid value.
        if np.isnan(row[0]):
            first_valid = np.flatnonzero(~np.isnan(row))[0]
            row[:first_valid] = row[first_valid]
    return out[0] if squeeze else out


def _interp(x: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs along last axis."""
    out = x.copy()
    if out.ndim == 1:
        # Promote to 2D for unified logic.
        out = out[None, :]
        squeeze = True
    else:
        squeeze = False

    for i in range(out.shape[0]):
        row = out[i]
        isnan = np.isnan(row)
        if np.all(isnan):
            row[:] = 0.0
            continue
        # Interpolate only missing positions.
        idx = np.arange(row.size)
        row[isnan] = np.interp(idx[isnan], idx[~isnan], row[~isnan])
    return out[0] if squeeze else out
