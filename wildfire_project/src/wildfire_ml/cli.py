"""Command-line entry points for every pipeline stage."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from wildfire_ml.config import ProjectConfig, load_config
from wildfire_ml.data import prepare_data
from wildfire_ml.utils import configure_reproducibility


def _threshold(config: ProjectConfig) -> float:
    metrics_path = config.path("paths", "reports_dir") / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics["threshold_selected_on_validation"])
    configured = config.section("evaluation").get("threshold", "auto")
    return 0.5 if isinstance(configured, str) else float(configured)


def _load_model(config: ProjectConfig) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for prediction") from exc
    model_path = config.path("paths", "model_dir") / "final.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wildfire-ml", description="TensorFlow DenseNet wildfire classification pipeline"
    )
    parser.add_argument(
        "--config", default="params.yaml", help="Path to project YAML configuration"
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Safely extract and validate datasets")
    prepare_parser.add_argument("--force", action="store_true", help="Replace stale extracted data")
    subparsers.add_parser("train", help="Train and fine-tune DenseNet121")
    subparsers.add_parser("evaluate", help="Evaluate validation and test splits")
    subparsers.add_parser("pipeline", help="Run prepare, train, and evaluate")

    predict_parser = subparsers.add_parser("predict", help="Classify one image")
    predict_parser.add_argument("image", type=Path)
    predict_parser.add_argument(
        "--gradcam-output", type=Path, help="Optional path for an overlaid Grad-CAM image"
    )
    predict_parser.add_argument("--target-class", choices=(0, 1), type=int)

    monitor_parser = subparsers.add_parser(
        "monitor", help="Compare production prediction telemetry with test reference scores"
    )
    monitor_parser.add_argument("production_log", type=Path)
    monitor_parser.add_argument(
        "--output", type=Path, default=Path("artifacts/monitoring/drift_report.json")
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    configure_reproducibility(int(config.values.get("seed", 42)))

    if args.command == "prepare":
        result = prepare_data(config, force=args.force)
    elif args.command == "train":
        from wildfire_ml.training import train

        result = train(config)
    elif args.command == "evaluate":
        from wildfire_ml.evaluation import evaluate

        result = evaluate(config)
    elif args.command == "pipeline":
        from wildfire_ml.evaluation import evaluate
        from wildfire_ml.training import train

        result = {
            "prepare": prepare_data(config),
            "train": train(config),
            "evaluate": evaluate(config),
        }
    elif args.command == "predict":
        from wildfire_ml.gradcam import explain_image, load_image_batch

        model = _load_model(config)
        image_size = tuple(int(value) for value in config.section("data")["image_size"])
        threshold = _threshold(config)
        if args.gradcam_output:
            result = explain_image(
                model,
                args.image,
                args.gradcam_output,
                image_size=image_size,
                threshold=threshold,
                target_class=args.target_class,
            )
        else:
            batch, _ = load_image_batch(args.image, image_size)
            probability = float(model(batch, training=False).numpy()[0, 0])
            result = {
                "image": str(args.image),
                "wildfire_probability": probability,
                "predicted_label": "wildfire" if probability >= threshold else "nowildfire",
                "threshold": threshold,
            }
    elif args.command == "monitor":
        from wildfire_ml.monitoring import build_drift_report

        reference = config.path("paths", "reports_dir") / "test" / "predictions.csv"
        output = args.output
        if not output.is_absolute():
            output = config.project_root / output
        production_log = args.production_log
        if not production_log.is_absolute():
            production_log = config.project_root / production_log
        result = build_drift_report(
            reference,
            production_log,
            output,
            threshold=_threshold(config),
        )
    else:  # pragma: no cover - argparse enforces the command choices
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(result, indent=2, default=str))
