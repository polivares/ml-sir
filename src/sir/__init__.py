"""Utilities for SIR benchmarking.

Provides a small namespace that re-exports common helpers so notebooks and
scripts can import from src.sir without deep module paths.
"""

from __future__ import annotations

# Keep import-time dependencies minimal.
#
# Some submodules (e.g. `simulate`) depend on optional or heavy third-party
# packages. Importing them eagerly would make *any* `import src.sir.*` fail if a
# subset of dependencies is missing (e.g. when only plotting paper figures).
# We therefore expose the public convenience API via lazy imports.
from importlib import import_module
from typing import Any

from .config import DEFAULTS, set_global_seed

__all__ = [
    "DEFAULTS",
    "set_global_seed",
    # simulate
    "simulate_sir",
    # datasets
    "load_sir_pkl",
    "build_Xy_I_only",
    "train_val_test_split",
    # metrics
    "per_param_metrics",
    "timing_summary",
    # noise
    "observe_poisson",
    "observe_negbin",
    "apply_downsample",
    "apply_window",
    "apply_missing",
    # baseline
    "fit_mse",
    "fit_poisson_mle",
    "fit_negbin_mle",
    # cache
    "hash_config",
    "cache_exists",
    "load_cache",
    "save_cache",
]

_LAZY: dict[str, tuple[str, str]] = {
    # simulate
    "simulate_sir": (".simulate", "simulate_sir"),
    # datasets
    "load_sir_pkl": (".datasets", "load_sir_pkl"),
    "build_Xy_I_only": (".datasets", "build_Xy_I_only"),
    "train_val_test_split": (".datasets", "train_val_test_split"),
    # metrics
    "per_param_metrics": (".metrics", "per_param_metrics"),
    "timing_summary": (".metrics", "timing_summary"),
    # noise
    "observe_poisson": (".noise", "observe_poisson"),
    "observe_negbin": (".noise", "observe_negbin"),
    "apply_downsample": (".noise", "apply_downsample"),
    "apply_window": (".noise", "apply_window"),
    "apply_missing": (".noise", "apply_missing"),
    # baseline
    "fit_mse": (".baseline", "fit_mse"),
    "fit_poisson_mle": (".baseline", "fit_poisson_mle"),
    "fit_negbin_mle": (".baseline", "fit_negbin_mle"),
    # cache
    "hash_config": (".cache", "hash_config"),
    "cache_exists": (".cache", "cache_exists"),
    "load_cache": (".cache", "load_cache"),
    "save_cache": (".cache", "save_cache"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazily import re-exported symbols to keep `import src.sir` lightweight."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    mod = import_module(module_name, __name__)
    value = getattr(mod, attr)
    globals()[name] = value  # Cache for subsequent lookups.
    return value
