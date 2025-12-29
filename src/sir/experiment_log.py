"""Experiment log utilities.

Maintains a Markdown log that tracks runs per experiment, the latest run per
experiment, and manual checkbox selections for final analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EXPERIMENT_TITLES = {
    "exp0": "Exp0 (clean)",
    "exp1": "Exp1 (noise)",
    "exp2": "Exp2 (window/downsample)",
}


def _markers(exp_key: str, section: str) -> Tuple[str, str]:
    prefix = exp_key.upper()
    return (f"<!-- {prefix}_{section}_START -->", f"<!-- {prefix}_{section}_END -->")


def _section_template(exp_key: str, title: str) -> str:
    last_start, last_end = _markers(exp_key, "LAST")
    final_start, final_end = _markers(exp_key, "FINAL")
    runs_start, runs_end = _markers(exp_key, "RUNS")
    return "\n".join([
        f"## {title}",
        "",
        "### Last run",
        last_start,
        "- _No runs yet._",
        last_end,
        "",
        "### Final selection",
        final_start,
        "- [ ] _No run selected yet._",
        final_end,
        "",
        "### Runs",
        runs_start,
        runs_end,
        "",
    ])


TEMPLATE = "\n".join([
    "# Experiment Log",
    "",
    "This file is auto-updated by the experiment scripts.",
    "Avoid editing text between markers unless noted below.",
    "",
    "Manual final selection:",
    "- You may edit the **Final selection** checkbox list manually.",
    "- It will only be overwritten for a given experiment when a script is run with `--mark-final`.",
    "",
    _section_template("exp0", EXPERIMENT_TITLES["exp0"]),
    _section_template("exp1", EXPERIMENT_TITLES["exp1"]),
    _section_template("exp2", EXPERIMENT_TITLES["exp2"]),
]).strip() + "\n"


def _ensure_log(path: Path) -> None:
    if not path.exists():
        path.write_text(TEMPLATE, encoding="utf-8")


def _ensure_section(text: str, exp_key: str) -> str:
    runs_start, runs_end = _markers(exp_key, "RUNS")
    if runs_start in text and runs_end in text:
        return text
    title = EXPERIMENT_TITLES.get(exp_key, exp_key.upper())
    return text.rstrip() + "\n\n" + _section_template(exp_key, title)


def _replace_block(text: str, start: str, end: str, content: str) -> str:
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        return f"{before}{start}\n{content.rstrip()}\n{end}{after}"
    return text.rstrip() + f"\n\n{start}\n{content.rstrip()}\n{end}\n"


def _get_block(text: str, start: str, end: str) -> str:
    if start in text and end in text:
        _, rest = text.split(start, 1)
        content, _ = rest.split(end, 1)
        return content.strip()
    return ""


def _append_run(text: str, start: str, end: str, entry: str) -> str:
    if start in text and end in text:
        before, rest = text.split(start, 1)
        middle, after = rest.split(end, 1)
        middle = middle.strip()
        if middle:
            middle = f"{middle}\n\n{entry.strip()}\n"
        else:
            middle = f"{entry.strip()}\n"
        return f"{before}{start}\n{middle}{end}{after}"
    runs_section = f"{start}\n{entry.strip()}\n{end}\n"
    return text.rstrip() + f"\n\n### Runs\n{runs_section}"


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def format_metrics_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "_No metrics available._"

    columns = [
        "method",
        "scenario",
        "mae_beta",
        "rmse_beta",
        "r2_beta",
        "mae_gamma",
        "rmse_gamma",
        "r2_gamma",
        "time_p50",
        "time_p90",
        "train_time_sec",
        "n_test",
    ]

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = [_format_value(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_args(args: Mapping[str, object], keys: Iterable[str]) -> str:
    parts: List[str] = []
    for key in keys:
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, bool):
            if value:
                parts.append(f"{key}=true")
            continue
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _final_line(title: str, run_dir: Path, checked: bool, note: Optional[str]) -> str:
    box = "x" if checked else " "
    line = f"- [{box}] {title} | run_dir: `{run_dir}`"
    if note:
        line += f" | note: {note}"
    return line


def _update_final_block(
    text: str,
    start: str,
    end: str,
    title: str,
    run_dir: Path,
    mark_final: bool,
    final_note: Optional[str],
) -> str:
    block = _get_block(text, start, end)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    lines = [ln for ln in lines if "_No run selected yet._" not in ln]

    run_token = f"`{run_dir}`"
    found = False
    for i, line in enumerate(lines):
        if run_token in line:
            found = True
            if mark_final:
                lines[i] = _final_line(title, run_dir, True, final_note)
            break

    if not found:
        lines.append(_final_line(title, run_dir, mark_final, final_note if mark_final else None))

    content = "\n".join(lines) if lines else "- [ ] _No run selected yet._"
    return _replace_block(text, start, end, content)


def update_experiment_log(
    path: Path | str,
    exp_key: str,
    title: str,
    run_dir: Path | str,
    script: str,
    args_summary: str,
    artifacts: Sequence[str],
    metrics_rows: Sequence[Mapping[str, object]],
    mark_final: bool = False,
    final_note: Optional[str] = None,
) -> None:
    """Append a run entry and update per-experiment sections."""
    path = Path(path)
    _ensure_log(path)
    text = path.read_text(encoding="utf-8")

    text = _ensure_section(text, exp_key)

    run_dir = Path(run_dir)
    artifacts_line = ", ".join(artifacts)

    entry = "\n".join([
        f"#### {title}",
        f"- run_dir: `{run_dir}`",
        f"- script: `{script}`",
        f"- args: `{args_summary}`" if args_summary else "- args: _n/a_",
        f"- artifacts: {artifacts_line}",
        "",
        "Metrics:",
        format_metrics_table(metrics_rows),
    ])

    runs_start, runs_end = _markers(exp_key, "RUNS")
    text = _append_run(text, runs_start, runs_end, entry)

    last_start, last_end = _markers(exp_key, "LAST")
    last_line = f"- {title} | run_dir: `{run_dir}` | metrics: `{run_dir / 'metrics.csv'}`"
    text = _replace_block(text, last_start, last_end, last_line)

    final_start, final_end = _markers(exp_key, "FINAL")
    text = _update_final_block(
        text,
        final_start,
        final_end,
        title=title,
        run_dir=run_dir,
        mark_final=mark_final,
        final_note=final_note,
    )

    path.write_text(text, encoding="utf-8")
