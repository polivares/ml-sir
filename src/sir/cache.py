"""Cache helpers for derived datasets and splits.

Stores arrays and config hashes to reuse preprocessing."""


from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Dict, Tuple

import numpy as np


def _stable_json(payload: Dict) -> str:
    """Serialize config deterministically for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def hash_config(payload: Dict, length: int = 12) -> str:
    """Create a short stable hash from a config dict."""
    raw = _stable_json(payload).encode("utf-8")
    # SHA-256 to avoid collisions; truncate for readable folder names.
    digest = hashlib.sha256(raw).hexdigest()
    # Truncate for readable folder names.
    return digest[:length]


def cache_paths(base_dir: Path | str, key: str) -> Tuple[Path, Path, Path]:
    """Return (dir, arrays_path, config_path) for a cache key."""
    base = Path(base_dir) / key
    arrays_path = base / "arrays.npz"
    config_path = base / "config.json"
    return base, arrays_path, config_path


def save_cache(base_dir: Path | str, key: str, arrays: Dict, config: Dict) -> None:
    """Persist arrays and config under a cache key."""
    cache_dir, arrays_path, config_path = cache_paths(base_dir, key)
    # Keep arrays and config together to preserve provenance.
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_cache(base_dir: Path | str, key: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Load arrays and config for a cache key."""
    _, arrays_path, config_path = cache_paths(base_dir, key)
    arrays = np.load(arrays_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    # Materialize arrays into a plain dict for callers.
    return {k: arrays[k] for k in arrays.files}, config


def cache_exists(base_dir: Path | str, key: str) -> bool:
    """Check if the cache entry exists on disk."""
    _, arrays_path, config_path = cache_paths(base_dir, key)
    return arrays_path.exists() and config_path.exists()
