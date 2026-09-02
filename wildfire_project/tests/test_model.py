import numpy as np
import pytest

from wildfire_ml.gradcam import make_gradcam_heatmap
from wildfire_ml.model import build_model, enable_fine_tuning, get_backbone

tf = pytest.importorskip("tensorflow")


MODEL_CONFIG = {
    "backbone": "DenseNet121",
    "imagenet_weights": None,
    "dense_units": 16,
    "dropout_rate": 0.1,
    "l2_regularization": 1e-4,
    "augmentation": {
        "rotation": 0.0,
        "zoom": 0.0,
        "contrast": 0.0,
        "translation": 0.0,
    },
}


def test_densenet_model_output_and_fine_tuning_policy() -> None:
    model = build_model(MODEL_CONFIG, image_size=(32, 32))
    prediction = model(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)

    assert prediction.shape == (1, 1)
    trainable_count = enable_fine_tuning(model, unfreeze_last_n=10)
    backbone = get_backbone(model)
    assert trainable_count > 0
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )


def test_gradcam_returns_normalized_spatial_map() -> None:
    inputs = tf.keras.Input((16, 16, 3))
    features = tf.keras.layers.Conv2D(4, 3, activation="relu")(inputs)
    features = tf.keras.layers.Activation("linear", name="gradcam_features")(features)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(pooled)
    model = tf.keras.Model(inputs, output)

    heatmap, probability, explained_class = make_gradcam_heatmap(
        np.ones((1, 16, 16, 3), dtype=np.float32), model, target_class=1
    )

    assert heatmap.shape == (14, 14)
    assert np.min(heatmap) >= 0
    assert np.max(heatmap) <= 1
    assert 0 <= probability <= 1
    assert explained_class == 1
