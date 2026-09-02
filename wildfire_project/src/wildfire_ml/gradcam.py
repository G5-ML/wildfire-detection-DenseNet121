"""Grad-CAM explanations and image overlays for binary wildfire predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from wildfire_ml.utils import ensure_directories, write_json


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for Grad-CAM") from exc
    return tf


def load_image_batch(
    image_path: str | Path, image_size: tuple[int, int]
) -> tuple[np.ndarray, Image.Image]:
    """Load one RGB image as an unnormalized batch; preprocessing lives in the model."""
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return batch, image


def make_gradcam_heatmap(
    image_batch: np.ndarray,
    model: Any,
    *,
    target_class: int | None = None,
    decision_threshold: float = 0.5,
    feature_layer_name: str = "gradcam_features",
) -> tuple[np.ndarray, float, int]:
    """Return normalized Grad-CAM, wildfire probability, and explained class."""
    tf = _require_tensorflow()
    feature_layer = model.get_layer(feature_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[feature_layer.output, model.output],
        name="gradcam_model",
    )
    tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    if not 0 <= decision_threshold <= 1:
        raise ValueError("decision_threshold must be in [0, 1]")
    with tf.GradientTape() as tape:
        convolution_output, prediction = grad_model(tensor, training=False)
        probability = prediction[:, 0]
        if target_class is None:
            explained_class = int(probability[0] >= decision_threshold)
        elif target_class in (0, 1):
            explained_class = int(target_class)
        else:
            raise ValueError("target_class must be 0, 1, or None")
        class_score = probability if explained_class == 1 else 1.0 - probability

    gradients = tape.gradient(class_score, convolution_output)
    if gradients is None:
        raise RuntimeError("Could not compute gradients for the selected feature layer")
    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(channel_weights * convolution_output, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
    return np.asarray(heatmap), float(probability[0]), explained_class


def overlay_heatmap(
    original: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a colorized heatmap over an image without OpenCV."""
    import matplotlib as mpl

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        original.size, Image.Resampling.BILINEAR
    )
    color_values = mpl.colormaps[colormap](np.asarray(resized, dtype=np.float32) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(color_values * 255), mode="RGB")
    return Image.blend(original.convert("RGB"), colored, alpha=alpha)


def explain_image(
    model: Any,
    image_path: str | Path,
    output_path: str | Path,
    *,
    image_size: tuple[int, int],
    threshold: float = 0.5,
    target_class: int | None = None,
) -> dict[str, Any]:
    batch, original = load_image_batch(image_path, image_size)
    heatmap, probability, explained_class = make_gradcam_heatmap(
        batch, model, target_class=target_class, decision_threshold=threshold
    )
    output = Path(output_path)
    ensure_directories(output.parent)
    overlay_heatmap(original, heatmap).save(output)
    result = {
        "image": str(image_path),
        "wildfire_probability": probability,
        "predicted_label": "wildfire" if probability >= threshold else "nowildfire",
        "threshold": threshold,
        "gradcam_explained_class": "wildfire" if explained_class == 1 else "nowildfire",
        "gradcam_path": str(output),
    }
    write_json(output.with_suffix(".json"), result)
    return result
