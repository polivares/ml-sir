"""Metrics for SIR parameter estimation.

Includes MAE/RMSE/R2 per parameter and timing summaries."""


from typing import Dict
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Guard against zero-variance targets.
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def per_param_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE/RMSE/R2 per parameter (beta, gamma)."""
    metrics = {}
    # Parameter order matches y = [beta, gamma] with shape (n_samples, 2).
    for idx, name in enumerate(["beta", "gamma"]):
        metrics[f"mae_{name}"] = mae(y_true[:, idx], y_pred[:, idx])
        metrics[f"rmse_{name}"] = rmse(y_true[:, idx], y_pred[:, idx])
        metrics[f"r2_{name}"] = r2(y_true[:, idx], y_pred[:, idx])
    return metrics


def timing_summary(times: np.ndarray) -> Dict[str, float]:
    """Summarize timings with p50/p90 (seconds)."""
    times = np.asarray(times)
    if times.size == 0:
        # Keep metrics schema consistent if no timing samples.
        return {"time_p50": 0.0, "time_p90": 0.0}
    return {
        "time_p50": float(np.percentile(times, 50)),
        "time_p90": float(np.percentile(times, 90)),
    }
