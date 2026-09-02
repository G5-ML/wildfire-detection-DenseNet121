"""DenseNet transfer-learning model construction."""

from __future__ import annotations

from typing import Any


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to construct the DenseNet model") from exc
    return tf


def build_augmentation(config: dict[str, Any]) -> Any:
    tf = _require_tensorflow()
    augmentation = config.get("augmentation", {})
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", name="random_flip"),
            tf.keras.layers.RandomRotation(
                float(augmentation.get("rotation", 0.08)),
                fill_mode="reflect",
                name="random_rotation",
            ),
            tf.keras.layers.RandomZoom(
                float(augmentation.get("zoom", 0.12)), fill_mode="reflect", name="random_zoom"
            ),
            tf.keras.layers.RandomContrast(
                float(augmentation.get("contrast", 0.15)), name="random_contrast"
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=float(augmentation.get("translation", 0.08)),
                width_factor=float(augmentation.get("translation", 0.08)),
                fill_mode="reflect",
                name="random_translation",
            ),
        ],
        name="data_augmentation",
    )


def build_model(config: dict[str, Any], image_size: tuple[int, int]) -> Any:
    """Build an ImageNet-initialized DenseNet121 binary classifier."""
    tf = _require_tensorflow()
    if config.get("backbone", "DenseNet121") != "DenseNet121":
        raise ValueError("This project currently supports model.backbone=DenseNet121")

    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    x = build_augmentation(config)(inputs)
    # DenseNet's documented "torch" preprocessing, expressed as serializable
    # built-in layers: scale to [0, 1], then apply ImageNet channel statistics.
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="imagenet_rescale")(x)
    x = tf.keras.layers.Normalization(
        mean=(0.485, 0.456, 0.406),
        variance=(0.229**2, 0.224**2, 0.225**2),
        name="imagenet_normalize",
    )(x)
    weights = config.get("imagenet_weights", "imagenet")
    if isinstance(weights, str) and weights.lower() in {"none", "null", ""}:
        weights = None
    backbone = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=(*image_size, 3),
    )
    backbone.trainable = False
    # Keep BatchNorm in inference mode; this avoids destabilizing its pretrained
    # moving statistics with comparatively small fine-tuning batches.
    x = backbone(x, training=False)
    # A named outer-graph feature tensor makes Grad-CAM reliable even though the
    # DenseNet application model is nested inside the classifier.
    x = tf.keras.layers.Activation("linear", name="gradcam_features")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dense(
        int(config.get("dense_units", 256)),
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(
            float(config.get("l2_regularization", 1e-4))
        ),
        name="classification_dense",
    )(x)
    x = tf.keras.layers.Dropout(float(config.get("dropout_rate", 0.4)), name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="wildfire")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="wildfire_densenet121")


def compile_model(
    model: Any,
    *,
    learning_rate: float,
    label_smoothing: float,
) -> None:
    """Compile with stable optimization and training-time monitoring metrics."""
    tf = _require_tensorflow()
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )


def get_backbone(model: Any) -> Any:
    """Find the nested DenseNet backbone without relying on an auto-numbered name."""
    tf = _require_tensorflow()
    candidates = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("densenet")
    ]
    if len(candidates) != 1:
        names = [layer.name for layer in candidates]
        raise ValueError(f"Expected one DenseNet backbone; found {names}")
    return candidates[0]


def enable_fine_tuning(model: Any, unfreeze_last_n: int) -> int:
    """Unfreeze the last DenseNet layers while keeping BatchNorm frozen."""
    tf = _require_tensorflow()
    backbone = get_backbone(model)
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - int(unfreeze_last_n))
    trainable_count = 0
    for index, layer in enumerate(backbone.layers):
        trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
        layer.trainable = trainable
        trainable_count += int(trainable)
    return trainable_count
