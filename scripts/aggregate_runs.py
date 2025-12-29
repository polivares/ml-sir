"""Aggregate metrics across benchmark runs.

Scans runs/*/metrics.csv, adds run metadata (and optionally config values),
and writes a single summary CSV for analysis.
Typical usage:
  python scripts/aggregate_runs.py --include-config
"""


from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from src.sir.logging_utils import setup_logging


def _parse_args() -> argparse.Namespace:
    # CLI options control input/output paths and config inclusion.
    parser = argparse.ArgumentParser(description="Aggregate run metrics into one CSV.")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--out", type=str, default="runs/summary.csv")
    parser.add_argument("--include-config", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_exp_name(run_dir: Path) -> str:
    # Use the prefix before the first underscore, e.g. exp0_2025...
    return run_dir.name.split("_")[0]


def main() -> None:
    args = _parse_args()
    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)

    log_file = None
    if not args.no_log_file:
        log_file = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    setup_logging(level=args.log_level, log_file=log_file, console=not args.no_console_log)
    logger = logging.getLogger(__name__)

    logger.info("Aggregating metrics under %s", runs_dir)

    rows: List[Dict] = []

    # Collect metrics.csv files under runs/*/metrics.csv
    for metrics_path in sorted(runs_dir.glob("*/metrics.csv")):
        run_dir = metrics_path.parent
        exp_name = _infer_exp_name(run_dir)
        config = _load_config(run_dir / "config.json")

        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add run metadata to each row.
                row = dict(row)
                row["run_dir"] = str(run_dir)
                row["exp"] = exp_name

                if args.include_config and config:
                    # Prefix config values to avoid name collisions.
                    for k, v in config.items():
                        row[f"cfg_{k}"] = v

                rows.append(row)

    if not rows:
        logger.warning("No metrics.csv files found under %s", runs_dir)
        return

    # Build a union of all keys to avoid dropping columns.
    all_fields = sorted({k for row in rows for k in row.keys()})

    # Ensure output directory exists before writing.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
