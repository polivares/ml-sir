"""Run I/O helpers.

Small utilities to create output folders and persist configs/metrics as
JSON and CSV. Used by scripts to standardize run artifacts in runs/.
"""


from pathlib import Path
import json
import csv
from typing import Dict, Iterable, Union


def ensure_dir(path: Union[Path, str]) -> Path:
    path = Path(path)
    # Create output folder if needed.
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Union[Path, str], payload: Dict) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        # Stable formatting helps diffs and reproducibility.
        json.dump(payload, f, indent=2, sort_keys=True)


def save_csv(path: Union[Path, str], rows: Iterable[Dict]) -> None:
    path = Path(path)
    rows = list(rows)
    if not rows:
        # Avoid creating empty CSVs.
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        # Use keys from the first row as column order, then append new keys seen later.
        fieldnames = list(rows[0].keys())
        seen = set(fieldnames)
        for row in rows[1:]:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
