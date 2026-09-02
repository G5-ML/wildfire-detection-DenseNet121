"""FastAPI model-serving endpoint with optional Grad-CAM explanations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from wildfire_ml.config import load_config
from wildfire_ml.gradcam import make_gradcam_heatmap, overlay_heatmap

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
app = FastAPI(title="Wildfire DenseNet API", version="0.1.0")
STATE: dict[str, Any] = {}
LOGGER = logging.getLogger(__name__)


def _configured_threshold(config: Any) -> float:
    report_path = config.path("paths", "reports_dir") / "metrics.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return float(report["threshold_selected_on_validation"])
    value = config.section("evaluation").get("threshold", "auto")
    return 0.5 if isinstance(value, str) else float(value)


@app.on_event("startup")
def load_artifacts() -> None:
    import tensorflow as tf

    config_path = os.getenv("WILDFIRE_CONFIG", "params.yaml")
    config = load_config(config_path)
    default_model = config.path("paths", "model_dir") / "final.keras"
    model_path = Path(os.getenv("WILDFIRE_MODEL", default_model))
    STATE.update(
        config=config,
        model=tf.keras.models.load_model(model_path, compile=False),
        threshold=_configured_threshold(config),
    )


def _decode_image(payload: bytes) -> tuple[np.ndarray, Image.Image]:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Upload is not a readable image") from exc
    image_size = tuple(int(value) for value in STATE["config"].section("data")["image_size"])
    resized = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0), image


def _log_prediction(payload: bytes, result: dict[str, Any], endpoint: str) -> None:
    telemetry_path = os.getenv("WILDFIRE_PREDICTION_LOG")
    if not telemetry_path:
        return
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        **{key: value for key, value in result.items() if key != "gradcam_png_base64"},
    }
    try:
        destination = Path(telemetry_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        LOGGER.exception("Could not append prediction telemetry")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if "model" in STATE else "loading"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    payload = await file.read()
    batch, _ = _decode_image(payload)
    probability = float(STATE["model"](batch, training=False).numpy()[0, 0])
    threshold = float(STATE["threshold"])
    result = {
        "wildfire_probability": probability,
        "predicted_label": "wildfire" if probability >= threshold else "nowildfire",
        "threshold": threshold,
        "model_version": "0.1.0",
    }
    _log_prediction(payload, result, "/predict")
    return result


@app.post("/explain")
async def explain(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    payload = await file.read()
    batch, original = _decode_image(payload)
    threshold = float(STATE["threshold"])
    heatmap, probability, explained_class = make_gradcam_heatmap(
        batch, STATE["model"], decision_threshold=threshold
    )
    overlay = overlay_heatmap(original, heatmap)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    result = {
        "wildfire_probability": probability,
        "predicted_label": "wildfire" if probability >= threshold else "nowildfire",
        "threshold": threshold,
        "gradcam_explained_class": "wildfire" if explained_class else "nowildfire",
        "gradcam_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "model_version": "0.1.0",
    }
    _log_prediction(payload, result, "/explain")
    return result
