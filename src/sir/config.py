"""Central defaults for SIR benchmarks.

Holds shared configuration and a global seed setter."""


from dataclasses import dataclass
from pathlib import Path
import random
from typing import Tuple

import numpy as np


# Central defaults for reproducible experiments.
@dataclass(frozen=True)
class Defaults:
    seed: int = 42
    t0: float = 0.0
    t1: float = 100.0
    dt: float = 0.1
    s0: float = 90.0
    i0: float = 10.0
    r0: float = 0.0
    beta_range: Tuple[float, float] = (0.1, 3.0)
    gamma_range: Tuple[float, float] = (0.1, 1.0)
    test_size: float = 0.1
    val_size: float = 0.1
    rho: float = 1.0
    k: float = 10.0
    data_path: Path = Path("data/raw/simulated/SIR/sir.pkl")
    runs_dir: Path = Path("runs")


# Shared defaults instance used across scripts.
DEFAULTS = Defaults()


def set_global_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    # Keep all PRNGs aligned across libraries used in experiments.
    # Cover Python and NumPy; TensorFlow is optional.
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow may not be installed; ignore if unavailable.
        pass
