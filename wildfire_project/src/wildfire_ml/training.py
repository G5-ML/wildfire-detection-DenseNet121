"""Two-stage transfer learning with resumable and periodic checkpointing."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from wildfire_ml.config import ProjectConfig
from wildfire_ml.data import build_image_dataset, compute_class_weights
from wildfire_ml.model import build_model, compile_model, enable_fine_tuning
from wildfire_ml.utils import configure_reproducibility, ensure_directories, write_json

LOGGER = logging.getLogger(__name__)


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for training") from exc
    return tf


def _periodic_checkpoint_callback(directory: Path, every_n_epochs: int) -> Any:
    tf = _require_tensorflow()

    class PeriodicWeightsCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            completed_epoch = epoch + 1
            if completed_epoch % every_n_epochs == 0:
                self.model.save_weights(directory / f"epoch_{completed_epoch:03d}.weights.h5")

    return PeriodicWeightsCheckpoint()


def _callbacks(config: ProjectConfig) -> list[Any]:
    tf = _require_tensorflow()
    training_cfg = config.section("training")
    model_dir = config.path("paths", "model_dir")
    logs_dir = config.path("paths", "logs_dir")
    checkpoints_dir = model_dir / "checkpoints"
    ensure_directories(checkpoints_dir, logs_dir / "tensorboard", model_dir / "backup")
    monitor = str(training_cfg.get("monitor", "val_pr_auc"))
    return [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_dir / "best.weights.h5",
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        _periodic_checkpoint_callback(
            checkpoints_dir, int(training_cfg.get("checkpoint_every_n_epochs", 2))
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=int(training_cfg.get("early_stopping_patience", 6)),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.2,
            patience=int(training_cfg.get("reduce_lr_patience", 3)),
            min_lr=float(training_cfg.get("min_learning_rate", 1e-7)),
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir / "tensorboard", histogram_freq=0, update_freq="epoch"
        ),
        tf.keras.callbacks.CSVLogger(logs_dir / "training.csv", append=True),
        tf.keras.callbacks.BackupAndRestore(backup_dir=model_dir / "backup"),
    ]


def _merge_history(target: dict[str, list[float]], history: Any) -> None:
    for name, values in history.history.items():
        target.setdefault(name, []).extend(float(value) for value in values)


def _flatten(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(name, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[name] = value
        elif isinstance(value, (list, tuple)):
            flattened[name] = ",".join(map(str, value))
    return flattened


@contextlib.contextmanager
def _tracking_run(config: ProjectConfig):
    tracking = config.section("mlflow")
    if not bool(tracking.get("enabled", True)):
        yield None
        return
    try:
        import mlflow
    except ImportError:
        LOGGER.warning(
            "MLflow is not installed; training will continue without experiment tracking"
        )
        yield None
        return

    tracking_uri = str(tracking.get("tracking_uri", "file:./mlruns"))
    if tracking_uri.startswith("file:./"):
        tracking_path = (config.project_root / tracking_uri.removeprefix("file:./")).resolve()
        tracking_uri = tracking_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(tracking.get("experiment_name", "wildfire-densenet")))
    with mlflow.start_run() as run:
        mlflow.log_params(_flatten("", config.values))
        yield (mlflow, run)


def train(config: ProjectConfig) -> dict[str, Any]:
    """Train the classifier head, fine-tune DenseNet, and save the best model."""
    tf = _require_tensorflow()
    seed = int(config.values.get("seed", 42))
    configure_reproducibility(seed)
    training_cfg = config.section("training")
    model_cfg = config.section("model")
    data_cfg = config.section("data")
    model_dir = config.path("paths", "model_dir")
    logs_dir = config.path("paths", "logs_dir")
    ensure_directories(model_dir, logs_dir)

    if bool(training_cfg.get("mixed_precision", True)) and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_bundle = build_image_dataset(config, "train", training=True)
    valid_bundle = build_image_dataset(config, "valid", training=False)
    class_weights = compute_class_weights(config)
    image_size = tuple(int(value) for value in data_cfg["image_size"])
    model = build_model(model_cfg, image_size=image_size)
    compile_model(
        model,
        learning_rate=float(training_cfg["learning_rate"]),
        label_smoothing=float(model_cfg.get("label_smoothing", 0.0)),
    )
    callbacks = _callbacks(config)
    combined_history: dict[str, list[float]] = {}

    with _tracking_run(config) as tracking_run:
        head_epochs = int(training_cfg.get("head_epochs", 0))
        if head_epochs:
            history = model.fit(
                train_bundle.dataset,
                validation_data=valid_bundle.dataset,
                epochs=head_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )
            _merge_history(combined_history, history)

        completed_epochs = len(combined_history.get("loss", []))
        fine_tune_epochs = int(training_cfg.get("fine_tune_epochs", 0))
        trainable_backbone_layers = 0
        if fine_tune_epochs:
            trainable_backbone_layers = enable_fine_tuning(
                model, int(training_cfg.get("unfreeze_last_n", 60))
            )
            compile_model(
                model,
                learning_rate=float(training_cfg["fine_tune_learning_rate"]),
                label_smoothing=float(model_cfg.get("label_smoothing", 0.0)),
            )
            history = model.fit(
                train_bundle.dataset,
                validation_data=valid_bundle.dataset,
                initial_epoch=completed_epochs,
                epochs=completed_epochs + fine_tune_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )
            _merge_history(combined_history, history)

        best_weights = model_dir / "best.weights.h5"
        if best_weights.is_file():
            model.load_weights(best_weights)
        final_model_path = model_dir / "final.keras"
        model.save(final_model_path)
        model.save_weights(model_dir / "final.weights.h5")

        summary = {
            "epochs_completed": len(combined_history.get("loss", [])),
            "trainable_fine_tune_backbone_layers": trainable_backbone_layers,
            "class_weights": class_weights,
            "best_model": str(final_model_path),
            "history": combined_history,
        }
        if tracking_run is not None:
            _, active_run = tracking_run
            summary["mlflow_run_id"] = active_run.info.run_id
        write_json(logs_dir / "history.json", combined_history)
        write_json(logs_dir / "training_summary.json", summary)

        if tracking_run is not None:
            mlflow, _ = tracking_run
            for metric_name, values in combined_history.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=step)
            mlflow.log_artifacts(str(logs_dir), artifact_path="training")
            registered_name = config.section("mlflow").get("registered_model_name")
            mlflow.tensorflow.log_model(
                model,
                artifact_path="model",
                registered_model_name=str(registered_name) if registered_name else None,
            )
    return summary
