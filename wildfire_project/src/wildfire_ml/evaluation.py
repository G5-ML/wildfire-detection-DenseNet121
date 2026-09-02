"""Exact scikit-learn evaluation metrics and diagnostic plot generation."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from wildfire_ml.config import ProjectConfig
from wildfire_ml.data import DatasetBundle, build_image_dataset
from wildfire_ml.utils import ensure_directories, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    curves: dict[str, list[float]]


def _as_binary_arrays(
    y_true: np.ndarray | list[int], y_probability: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)
    if truth.size == 0 or truth.size != probability.size:
        raise ValueError("y_true and y_probability must be non-empty and have equal length")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("y_probability must contain finite values in [0, 1]")
    return truth, probability


def select_fbeta_threshold(
    y_true: np.ndarray | list[int],
    y_probability: np.ndarray | list[float],
    *,
    beta: float = 2.0,
) -> tuple[float, float]:
    """Select a threshold on validation data only by maximizing F-beta."""
    truth, probability = _as_binary_arrays(y_true, y_probability)
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    if not thresholds.size:
        return 0.5, 0.0
    beta_squared = beta**2
    denominator = beta_squared * precision[:-1] + recall[:-1]
    scores = np.divide(
        (1 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def calculate_binary_metrics(
    y_true: np.ndarray | list[int],
    y_probability: np.ndarray | list[float],
    *,
    threshold: float = 0.5,
    beta: float = 2.0,
) -> EvaluationResult:
    """Compute requested thresholded and ranking metrics plus curve coordinates."""
    truth, probability = _as_binary_arrays(y_true, y_probability)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predictions = (probability >= threshold).astype(np.int32)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(truth, probability)
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "samples": int(truth.size),
        "positive_rate": float(np.mean(truth)),
        "accuracy_score": float(accuracy_score(truth, predictions)),
        "average_precision_score": float(average_precision_score(truth, probability)),
        "brier_score_loss": float(brier_score_loss(truth, probability)),
        "confusion_matrix": matrix.tolist(),
        "f1_score": float(f1_score(truth, predictions, zero_division=0)),
        "fbeta_score": float(fbeta_score(truth, predictions, beta=beta, zero_division=0)),
        "precision_score": float(precision_score(truth, predictions, zero_division=0)),
        "recall_score": float(recall_score(truth, predictions, zero_division=0)),
        "pr_auc": float(auc(pr_recall, pr_precision)),
    }
    curves: dict[str, list[float]] = {
        "precision_recall_curve.precision": pr_precision.tolist(),
        "precision_recall_curve.recall": pr_recall.tolist(),
        "precision_recall_curve.thresholds": pr_thresholds.tolist(),
    }
    if np.unique(truth).size == 2:
        false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(truth, probability)
        metrics["roc_auc_score"] = float(roc_auc_score(truth, probability))
        metrics["auc"] = float(auc(false_positive_rate, true_positive_rate))
        curves.update(
            {
                "roc_curve.false_positive_rate": false_positive_rate.tolist(),
                "roc_curve.true_positive_rate": true_positive_rate.tolist(),
                "roc_curve.thresholds": roc_thresholds.tolist(),
            }
        )
    else:
        metrics["roc_auc_score"] = float("nan")
        metrics["auc"] = float("nan")
        curves.update(
            {
                "roc_curve.false_positive_rate": [],
                "roc_curve.true_positive_rate": [],
                "roc_curve.thresholds": [],
            }
        )
    return EvaluationResult(metrics=metrics, curves=curves)


def collect_predictions(model: Any, bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, batch_labels in bundle.dataset:
        batch_probabilities = model(images, training=False)
        labels.append(np.asarray(batch_labels).reshape(-1))
        probabilities.append(np.asarray(batch_probabilities).reshape(-1))
    return np.concatenate(labels).astype(np.int32), np.concatenate(probabilities).astype(float)


def _save_predictions(
    path: Path,
    bundle: DatasetBundle,
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> None:
    if len(bundle.file_paths) != len(truth):
        raise ValueError("File path and prediction counts do not match")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_label", "wildfire_probability", "predicted_label"])
        for file_path, label, score in zip(bundle.file_paths, truth, probability, strict=True):
            writer.writerow([file_path, int(label), float(score), int(score >= threshold)])


def _save_plots(result: EvaluationResult, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(result.metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Oranges")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["nowildfire", "wildfire"],
        yticklabels=["nowildfire", "wildfire"],
        title="Confusion matrix",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(
        result.curves["roc_curve.false_positive_rate"],
        result.curves["roc_curve.true_positive_rate"],
        label=f"ROC AUC = {result.metrics['roc_auc_score']:.3f}",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(
        result.curves["precision_recall_curve.recall"],
        result.curves["precision_recall_curve.precision"],
        label=f"AP = {result.metrics['average_precision_score']:.3f}",
    )
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(figure)


def evaluate_bundle(
    model: Any,
    bundle: DatasetBundle,
    output_dir: Path,
    *,
    threshold: float,
    beta: float,
) -> EvaluationResult:
    ensure_directories(output_dir)
    truth, probability = collect_predictions(model, bundle)
    result = calculate_binary_metrics(truth, probability, threshold=threshold, beta=beta)
    write_json(output_dir / "metrics.json", result.metrics)
    write_json(output_dir / "curves.json", result.curves)
    _save_predictions(output_dir / "predictions.csv", bundle, truth, probability, threshold)
    _save_plots(result, output_dir)
    return result


def _log_evaluation_to_mlflow(
    config: ProjectConfig, summary: dict[str, Any], reports_dir: Path
) -> None:
    tracking = config.section("mlflow")
    if not bool(tracking.get("enabled", True)):
        return
    training_summary_path = config.path("paths", "logs_dir") / "training_summary.json"
    if not training_summary_path.is_file():
        return
    run_id = json.loads(training_summary_path.read_text(encoding="utf-8")).get("mlflow_run_id")
    if not run_id:
        return
    try:
        import mlflow
    except ImportError:
        LOGGER.warning("MLflow is not installed; evaluation metrics were not logged to MLflow")
        return
    tracking_uri = str(tracking.get("tracking_uri", "file:./mlruns"))
    if tracking_uri.startswith("file:./"):
        tracking_path = (config.project_root / tracking_uri.removeprefix("file:./")).resolve()
        tracking_uri = tracking_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        for split in ("validation", "test"):
            for name, value in summary[split].items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric(f"{split}.{name}", float(value))
        mlflow.log_metric(
            "threshold_selected_on_validation",
            float(summary["threshold_selected_on_validation"]),
        )
        mlflow.log_artifacts(str(reports_dir), artifact_path="evaluation")


def evaluate(config: ProjectConfig) -> dict[str, Any]:
    """Tune the operating threshold on validation data, then evaluate untouched test data."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for evaluation") from exc

    evaluation_cfg = config.section("evaluation")
    model_path = config.path("paths", "model_dir") / "final.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    valid_bundle = build_image_dataset(config, "valid", training=False)
    test_bundle = build_image_dataset(config, "test", training=False)
    beta = float(evaluation_cfg.get("beta", 2.0))

    validation_truth, validation_probability = collect_predictions(model, valid_bundle)
    configured_threshold = evaluation_cfg.get("threshold", "auto")
    if isinstance(configured_threshold, str) and configured_threshold.lower() == "auto":
        threshold, validation_best_fbeta = select_fbeta_threshold(
            validation_truth, validation_probability, beta=beta
        )
    else:
        threshold = float(configured_threshold)
        validation_best_fbeta = float(
            fbeta_score(
                validation_truth,
                validation_probability >= threshold,
                beta=beta,
                zero_division=0,
            )
        )

    reports_dir = config.path("paths", "reports_dir")
    validation_result = evaluate_bundle(
        model, valid_bundle, reports_dir / "validation", threshold=threshold, beta=beta
    )
    test_result = evaluate_bundle(
        model, test_bundle, reports_dir / "test", threshold=threshold, beta=beta
    )
    summary = {
        "threshold_selected_on_validation": threshold,
        "validation_selection_fbeta": validation_best_fbeta,
        "validation": validation_result.metrics,
        "test": test_result.metrics,
    }
    write_json(reports_dir / "metrics.json", summary)
    dvc_metrics = {
        "threshold_selected_on_validation": threshold,
        **{
            f"test_{name}": value
            for name, value in test_result.metrics.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        },
    }
    write_json(reports_dir / "dvc_metrics.json", dvc_metrics)
    _log_evaluation_to_mlflow(config, summary, reports_dir)
    return summary
