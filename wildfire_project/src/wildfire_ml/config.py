"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration with paths resolved from the project root."""

    values: dict[str, Any]
    project_root: Path
    source_path: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Missing or invalid configuration section: {name}")
        return value

    def path(self, section: str, key: str) -> Path:
        raw = self.section(section).get(key)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"Missing path value: {section}.{key}")
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (self.project_root / candidate).resolve()


def load_config(path: str | Path = "params.yaml") -> ProjectConfig:
    """Load YAML config and resolve project-relative paths."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")

    required = {"data", "model", "training", "evaluation", "paths", "mlflow"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    class_names = values["data"].get("class_names", [])
    if class_names != ["nowildfire", "wildfire"]:
        raise ValueError(
            "data.class_names must be [nowildfire, wildfire] so label 1 means wildfire"
        )
    image_size = values["data"].get("image_size", [])
    if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
        raise ValueError("data.image_size must contain two positive integers")
    return ProjectConfig(values=values, project_root=source.parent, source_path=source)

