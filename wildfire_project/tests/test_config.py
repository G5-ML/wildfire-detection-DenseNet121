from pathlib import Path

import pytest

from wildfire_ml.config import load_config


VALID_CONFIG = """
seed: 1
data:
  archives_dir: datasets
  processed_dir: data/processed
  class_names: [nowildfire, wildfire]
  image_size: [64, 64]
model: {}
training: {}
evaluation: {}
paths: {}
mlflow: {}
"""


def test_load_config_resolves_paths_from_config_location(tmp_path: Path) -> None:
    config_path = tmp_path / "params.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    config = load_config(config_path)

    assert config.path("data", "archives_dir") == (tmp_path / "datasets").resolve()


def test_load_config_rejects_reversed_class_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        VALID_CONFIG.replace("[nowildfire, wildfire]", "[wildfire, nowildfire]"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label 1 means wildfire"):
        load_config(config_path)

