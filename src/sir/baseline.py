"""Classical baseline fitting for SIR parameters.

Implements MSE fitting to I(t) and MLE for Poisson/NegBin observation models.
Includes multi-start optimization for robustness and returns timing info.
Used by Exp0/Exp1/Exp2 as the classical comparison baseline.
"""


from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .simulate import simulate_sir
from .config import DEFAULTS


def _mse_objective(
    params: np.ndarray,
    I_obs: np.ndarray,
    t0: float,
    t1: float,
    dt: float,
    s0: float,
    i0: float,
    r0: float,
) -> float:
    # MSE between simulated and observed I(t).
    beta, gamma = params
    # Simulate on the same grid as observations.
    I_sim = simulate_sir(beta, gamma, s0=s0, i0=i0, r0=r0, t0=t0, t1=t1, dt=dt)
    return float(np.mean((I_sim - I_obs) ** 2))


def _poisson_nll(
    params: np.ndarray,
    Y_obs: np.ndarray,
    rho: float,
    t0: float,
    t1: float,
    dt: float,
    s0: float,
    i0: float,
    r0: float,
    include_const: bool = False,
) -> float:
    # Allow optional rho estimation if params includes it.
    if params.size == 3:
        beta, gamma, rho = params
    else:
        beta, gamma = params
    I_sim = simulate_sir(beta, gamma, s0=s0, i0=i0, r0=r0, t0=t0, t1=t1, dt=dt)
    # Expected counts under Poisson observation model.
    lam = np.clip(rho * I_sim, 1e-8, None)
    nll = np.sum(lam - Y_obs * np.log(lam))
    if include_const:
        # Optional additive constant for full likelihood value.
        nll += np.sum(gammaln(Y_obs + 1))
    return float(nll)


def _negbin_nll(
    params: np.ndarray,
    Y_obs: np.ndarray,
    rho: float,
    k: float,
    t0: float,
    t1: float,
    dt: float,
    s0: float,
    i0: float,
    r0: float,
) -> float:
    # Allow optional rho estimation if params includes it.
    if params.size == 3:
        beta, gamma, rho = params
    else:
        beta, gamma = params
    I_sim = simulate_sir(beta, gamma, s0=s0, i0=i0, r0=r0, t0=t0, t1=t1, dt=dt)
    # Mean/variance parameterization for negative binomial.
    mu = np.clip(rho * I_sim, 1e-8, None)
    p = k / (k + mu)
    # NLL for NB with mean mu and size k.
    nll = -np.sum(
        gammaln(Y_obs + k) - gammaln(k) - gammaln(Y_obs + 1)
        + k * np.log(p) + Y_obs * np.log(1 - p)
    )
    return float(nll)


@dataclass
class FitResult:
    params: np.ndarray
    loss: float
    times: List[float]


def _fit_with_multistart(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    n_starts: int,
    rng: np.random.Generator,
    method: str = "L-BFGS-B",
) -> FitResult:
    best_params = None
    best_loss = np.inf
    times = []

    # Multi-start optimization for robustness to local minima.
    for _ in range(n_starts):
        # Sample an initial point uniformly within bounds.
        x0 = np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)
        start = time.perf_counter()
        res = minimize(objective, x0=x0, bounds=bounds, method=method)
        times.append(time.perf_counter() - start)
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_params = res.x

    return FitResult(params=best_params, loss=best_loss, times=times)


def fit_mse(
    I_obs: np.ndarray,
    beta_bounds: Tuple[float, float] = DEFAULTS.beta_range,
    gamma_bounds: Tuple[float, float] = DEFAULTS.gamma_range,
    n_starts: int = 5,
    rng: Optional[np.random.Generator] = None,
    t0: float = DEFAULTS.t0,
    t1: float = DEFAULTS.t1,
    dt: float = DEFAULTS.dt,
    s0: float = DEFAULTS.s0,
    i0: float = DEFAULTS.i0,
    r0: float = DEFAULTS.r0,
) -> FitResult:
    rng = rng or np.random.default_rng(DEFAULTS.seed)

    # Wrap objective to match the optimizer signature.
    def obj(params: np.ndarray) -> float:
        return _mse_objective(params, I_obs, t0, t1, dt, s0, i0, r0)

    return _fit_with_multistart(obj, [beta_bounds, gamma_bounds], n_starts, rng)


def fit_poisson_mle(
    Y_obs: np.ndarray,
    rho: float = DEFAULTS.rho,
    estimate_rho: bool = False,
    beta_bounds: Tuple[float, float] = DEFAULTS.beta_range,
    gamma_bounds: Tuple[float, float] = DEFAULTS.gamma_range,
    rho_bounds: Tuple[float, float] = (0.1, 1.0),
    n_starts: int = 5,
    rng: Optional[np.random.Generator] = None,
    t0: float = DEFAULTS.t0,
    t1: float = DEFAULTS.t1,
    dt: float = DEFAULTS.dt,
    s0: float = DEFAULTS.s0,
    i0: float = DEFAULTS.i0,
    r0: float = DEFAULTS.r0,
) -> FitResult:
    rng = rng or np.random.default_rng(DEFAULTS.seed)

    # Optionally include rho in the parameter vector.
    def obj(params: np.ndarray) -> float:
        return _poisson_nll(params, Y_obs, rho, t0, t1, dt, s0, i0, r0)

    bounds = [beta_bounds, gamma_bounds]
    if estimate_rho:
        bounds.append(rho_bounds)

    return _fit_with_multistart(obj, bounds, n_starts, rng)


def fit_negbin_mle(
    Y_obs: np.ndarray,
    rho: float = DEFAULTS.rho,
    k: float = DEFAULTS.k,
    estimate_rho: bool = False,
    beta_bounds: Tuple[float, float] = DEFAULTS.beta_range,
    gamma_bounds: Tuple[float, float] = DEFAULTS.gamma_range,
    rho_bounds: Tuple[float, float] = (0.1, 1.0),
    n_starts: int = 5,
    rng: Optional[np.random.Generator] = None,
    t0: float = DEFAULTS.t0,
    t1: float = DEFAULTS.t1,
    dt: float = DEFAULTS.dt,
    s0: float = DEFAULTS.s0,
    i0: float = DEFAULTS.i0,
    r0: float = DEFAULTS.r0,
) -> FitResult:
    rng = rng or np.random.default_rng(DEFAULTS.seed)

    # Optionally include rho in the parameter vector.
    def obj(params: np.ndarray) -> float:
        return _negbin_nll(params, Y_obs, rho, k, t0, t1, dt, s0, i0, r0)

    bounds = [beta_bounds, gamma_bounds]
    if estimate_rho:
        bounds.append(rho_bounds)

    return _fit_with_multistart(obj, bounds, n_starts, rng)
