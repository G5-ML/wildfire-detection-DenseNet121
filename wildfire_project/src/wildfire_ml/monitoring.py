"""Lightweight post-deployment prediction-distribution monitoring."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_ml.utils import write_json


def population_stability_index(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Measure shift between reference and current score distributions."""
    reference_array = np.asarray(reference, dtype=float).reshape(-1)
    current_array = np.asarray(current, dtype=float).reshape(-1)
    if reference_array.size == 0 or current_array.size == 0:
        raise ValueError("reference and current samples must be non-empty")
    if not np.isfinite(reference_array).all() or not np.isfinite(current_array).all():
        raise ValueError("monitoring samples must contain only finite values")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    boundaries = np.unique(np.quantile(reference_array, quantiles))
    edges = np.concatenate(([-np.inf], boundaries, [np.inf]))
    reference_counts = np.histogram(reference_array, bins=edges)[0]
    current_counts = np.histogram(current_array, bins=edges)[0]
    reference_ratio = np.clip(reference_counts / reference_array.size, epsilon, None)
    current_ratio = np.clip(current_counts / current_array.size, epsilon, None)
    terms = (current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio)
    return float(np.sum(terms))


def _read_reference_scores(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        scores = [float(row["wildfire_probability"]) for row in rows]
    return np.asarray(scores, dtype=float)


def _read_production_scores(path: Path) -> np.ndarray:
    scores: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                scores.append(float(record["wildfire_probability"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid telemetry at {path}:{line_number}") from exc
    return np.asarray(scores, dtype=float)


def build_drift_report(
    reference_predictions: Path,
    production_log: Path,
    output_path: Path,
    *,
    threshold: float,
    psi_warning: float = 0.2,
    minimum_samples: int = 100,
) -> dict[str, Any]:
    """Compare production scores with held-out reference prediction scores."""
    if not reference_predictions.is_file():
        raise FileNotFoundError(f"Reference predictions not found: {reference_predictions}")
    if not production_log.is_file():
        raise FileNotFoundError(f"Production telemetry not found: {production_log}")
    reference = _read_reference_scores(reference_predictions)
    current = _read_production_scores(production_log)
    psi = population_stability_index(reference, current)
    status = "insufficient_data" if current.size < minimum_samples else "ok"
    if current.size >= minimum_samples and psi >= psi_warning:
        status = "warning"
    report = {
        "status": status,
        "psi": psi,
        "psi_warning_threshold": psi_warning,
        "minimum_production_samples": minimum_samples,
        "reference_samples": int(reference.size),
        "production_samples": int(current.size),
        "reference_mean_wildfire_probability": float(np.mean(reference)),
        "production_mean_wildfire_probability": float(np.mean(current)),
        "reference_alert_rate": float(np.mean(reference >= threshold)),
        "production_alert_rate": float(np.mean(current >= threshold)),
        "production_low_confidence_rate": float(np.mean(np.abs(current - threshold) <= 0.1)),
        "threshold": threshold,
        "interpretation": (
            "PSI is a screening signal, not a performance metric. Investigate warnings and join "
            "delayed "
            "ground truth before retraining or promotion decisions."
        ),
    }
    write_json(output_path, report)
    return report
