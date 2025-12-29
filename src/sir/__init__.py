"""Utilities for SIR benchmarking.

Provides a small namespace that re-exports common helpers so notebooks and
scripts can import from src.sir without deep module paths.
"""


# Re-export core helpers for convenience (avoid heavy imports here).
from .config import DEFAULTS, set_global_seed  # noqa: F401
from .simulate import simulate_sir  # noqa: F401
from .datasets import load_sir_pkl, build_Xy_I_only, train_val_test_split  # noqa: F401
from .metrics import per_param_metrics, timing_summary  # noqa: F401
from .noise import (  # noqa: F401
    observe_poisson,
    observe_negbin,
    apply_downsample,
    apply_window,
    apply_missing,
)
from .baseline import fit_mse, fit_poisson_mle, fit_negbin_mle  # noqa: F401
from .cache import hash_config, cache_exists, load_cache, save_cache  # noqa: F401
