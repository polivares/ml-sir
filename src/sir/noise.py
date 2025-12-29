"""Observation models and preprocessing utilities.

This module provides *observation models* and *data availability transforms* for SIR benchmarks.

Important distinction (common confusion):
- The functions here do **not** add Gaussian noise (i.e., not `I + eps`) by default.
- Instead, they model an *observation process* where a latent trajectory `I(t)` produces an
  observed count time series `Y(t)` via a distribution such as Poisson or Negative Binomial.

In practice, benchmark scripts often treat `Y(t)` as the observed version of `I(t)` and feed it to
either classical likelihood baselines (MLE) or ML models.
"""


from typing import Optional, Tuple, Union
import numpy as np


def _ensure_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    """Return a NumPy Generator, creating a default one when not provided."""
    # Use a default generator when none is provided.
    return rng or np.random.default_rng()


def observe_poisson(
    I: np.ndarray, rho: float, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate observed counts under a Poisson observation model.

    Parameters
    ----------
    I:
        Latent infected trajectory `I(t)` (can be 1D `(T,)` or batch `(..., T)`), typically float.
        Values are treated as non-negative intensities after scaling.
    rho:
        Reporting rate / scaling factor. The Poisson mean is `lambda_t = rho * I_t`.
    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Observed counts `Y(t)` drawn from Poisson, same shape as `I`, dtype `int64`.

    Notes
    -----
    This is not additive noise (`I + eps`): it produces a *new* observed series by sampling.
    """
    rng = _ensure_rng(rng)
    # rho scales the latent infections into observed counts.
    # Clamp to avoid negative/invalid rates.
    lam = np.clip(rho * I, 0.0, None)
    return rng.poisson(lam).astype(np.int64, copy=False)


def observe_negbin(
    I: np.ndarray, rho: float, k: float, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate observed counts under a Negative Binomial observation model.

    Parameters
    ----------
    I:
        Latent infected trajectory `I(t)` (can be 1D `(T,)` or batch `(..., T)`), typically float.
    rho:
        Reporting rate / scaling factor. The NegBin mean is `mu_t = rho * I_t`.
    k:
        Dispersion parameter (> 0). Lower values imply higher over-dispersion.
        The variance is `mu + mu^2 / k`.
    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Observed counts `Y(t)` drawn from NegBin, same shape as `I`, dtype `int64`.

    Notes
    -----
    This is an observation model (count sampling), not additive noise.
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
    """Downsample a time series along the last axis.

    Parameters
    ----------
    x:
        Array with time along the last axis, e.g. `(T,)` or `(n, T)`.
    step:
        Keep every `step`-th point (must be > 0). For example, `step=10` reduces resolution by 10x.

    Returns
    -------
    np.ndarray
        Downsampled array with shape `x[..., ::step]`.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    return x[..., ::step]


def apply_window(x: np.ndarray, T: float, dt: float) -> np.ndarray:
    """Keep only the first `T` days of a time series (based on `dt`).

    Parameters
    ----------
    x:
        Array with time along the last axis.
    T:
        Window length in days (must be > 0).
    dt:
        Timestep used to interpret `T` in number of points.

    Returns
    -------
    np.ndarray
        Truncated series containing the first `round(T/dt)+1` points.
    """
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
    """Apply random missingness and optionally impute the missing values.

    Parameters
    ----------
    x:
        Time series array (1D `(T,)` or batch `(..., T)`).
    p:
        Probability of masking each entry independently (`0 <= p < 1`).
    rng:
        Optional NumPy random generator for reproducibility.
    method:
        Imputation method:
        - `None`: keep missing values as NaN.
        - `"ffill"`: forward-fill along the last axis (with backfill for leading NaNs).
        - `"interp"`: linear interpolation along the last axis.
    return_mask:
        If True, also return the boolean mask where True means "missing".

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        The output series is float (because it may contain NaNs / imputed values).
        If `return_mask=True`, returns `(x_out, mask)`.

    Notes
    -----
    This transform models missing reporting / incomplete observation, not measurement noise.
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
