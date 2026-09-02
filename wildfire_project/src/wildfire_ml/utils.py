"""Shared reproducibility and serialization helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def configure_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Set process-level and TensorFlow random seeds."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except (ImportError, RuntimeError):
        # ImportError allows lightweight data/config commands. Some accelerators
        # cannot enable every deterministic kernel and raise RuntimeError.
        pass


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_directories(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
