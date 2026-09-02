"""Safe archive preparation, validation, and deterministic tf.data input pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from wildfire_ml.config import ProjectConfig
from wildfire_ml.utils import ensure_directories, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class DatasetBundle:
    dataset: Any
    file_paths: list[str]
    class_names: list[str]


def _normalized_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if ":" in path.parts[0]:
        raise ValueError(f"Drive-qualified archive member path: {name!r}")
    return path


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = info.external_attr >> 16
    return stat.S_ISLNK(unix_mode)


def archive_fingerprint(archive_path: Path) -> str:
    """Fingerprint the ZIP central directory without rereading all compressed bytes."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            digest.update(info.filename.encode("utf-8", errors="surrogateescape"))
            digest.update(str(info.CRC).encode())
            digest.update(str(info.file_size).encode())
    return digest.hexdigest()


def _archive_inventory(
    archive_path: Path, split: str, class_names: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[PurePosixPath] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = _normalized_member_path(info.filename)
            if info.is_dir():
                continue
            if _is_zip_symlink(info):
                raise ValueError(f"Symbolic links are not accepted: {info.filename!r}")
            if relative in seen:
                raise ValueError(f"Duplicate archive path: {info.filename!r}")
            seen.add(relative)
            if relative.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if len(relative.parts) != 2 or relative.parts[0] not in class_names:
                raise ValueError(
                    f"Expected '<class>/<image>' in {archive_path.name}; got {info.filename!r}"
                )
            class_name = relative.parts[0]
            records.append(
                {
                    "split": split,
                    "relative_path": relative.as_posix(),
                    "class_name": class_name,
                    "label": class_names.index(class_name),
                    "bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )
    if not records:
        raise ValueError(f"No supported images found in {archive_path}")
    missing = set(class_names).difference(record["class_name"] for record in records)
    if missing:
        raise ValueError(f"{archive_path.name} is missing classes: {sorted(missing)}")
    return records


def safe_extract_archive(
    archive_path: Path,
    destination: Path,
    class_names: list[str],
    *,
    force: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """Safely extract one class-directory ZIP and return its inventory.

    Extraction is idempotent. A completion marker records the ZIP fingerprint; a
    changed source archive must be explicitly re-prepared with ``force=True``.
    """
    split = archive_path.stem
    records = _archive_inventory(archive_path, split, class_names)
    fingerprint = archive_fingerprint(archive_path)
    marker = destination / ".complete.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("archive_fingerprint") == fingerprint:
            return records, False
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"Incomplete or stale destination exists: {destination}. Rerun with --force."
            )
        shutil.rmtree(destination)

    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                relative = _normalized_member_path(info.filename)
                if info.is_dir():
                    continue
                if _is_zip_symlink(info):
                    raise ValueError(f"Symbolic links are not accepted: {info.filename!r}")
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        write_json(
            marker,
            {
                "archive": archive_path.name,
                "archive_fingerprint": fingerprint,
                "image_count": len(records),
            },
        )
    except Exception:
        # A failed extraction is deliberately left without a completion marker;
        # --force can safely replace it on the next run.
        raise
    return records, True


def _verify_images(paths: Iterable[Path]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required when data.verify_images is true") from exc

    invalid: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid.append(str(path))
            if len(invalid) >= 10:
                break
    if invalid:
        raise ValueError(f"Unreadable images found (showing up to 10): {invalid}")


def prepare_data(config: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    """Extract all supplied splits, validate images, and write a stable manifest."""
    data_cfg = config.section("data")
    archives_dir = config.path("data", "archives_dir")
    processed_dir = config.path("data", "processed_dir")
    class_names = list(data_cfg["class_names"])
    ensure_directories(processed_dir)

    all_records: list[dict[str, Any]] = []
    extracted: list[str] = []
    for split in SPLITS:
        archive_path = archives_dir / f"{split}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Required archive not found: {archive_path}")
        records, did_extract = safe_extract_archive(
            archive_path, processed_dir / split, class_names, force=force
        )
        all_records.extend(records)
        if did_extract:
            extracted.append(split)

    if bool(data_cfg.get("verify_images", True)):
        _verify_images(
            processed_dir / record["split"] / record["relative_path"]
            for record in all_records
        )

    manifest_path = processed_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {
            class_name: sum(
                record["split"] == split and record["class_name"] == class_name
                for record in all_records
            )
            for class_name in class_names
        }
    summary = {
        "class_names": class_names,
        "counts": counts,
        "total_images": len(all_records),
        "newly_extracted_splits": extracted,
        "manifest": str(manifest_path),
    }
    write_json(processed_dir / "dataset_summary.json", summary)
    return summary


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model commands. Install this package in Python 3.10-3.12."
        ) from exc
    return tf


def build_image_dataset(
    config: ProjectConfig,
    split: str,
    *,
    training: bool = False,
) -> DatasetBundle:
    """Create an efficient tf.data pipeline while retaining stable file ordering."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    tf = _require_tensorflow()
    data_cfg = config.section("data")
    split_dir = config.path("data", "processed_dir") / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Prepared split not found: {split_dir}. Run `prepare` first.")

    class_names = list(data_cfg["class_names"])
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        color_mode="rgb",
        batch_size=int(data_cfg["batch_size"]),
        image_size=tuple(int(value) for value in data_cfg["image_size"]),
        shuffle=training,
        seed=int(config.values.get("seed", 42)),
        interpolation="bilinear",
    )
    file_paths = list(dataset.file_paths)
    options = tf.data.Options()
    options.experimental_deterministic = not training
    dataset = dataset.with_options(options)

    cache = data_cfg.get("cache", False)
    if cache is True:
        dataset = dataset.cache()
    elif isinstance(cache, str) and cache:
        cache_path = Path(cache)
        if not cache_path.is_absolute():
            cache_path = config.project_root / cache_path
        ensure_directories(cache_path)
        dataset = dataset.cache(str(cache_path / f"{split}.cache"))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return DatasetBundle(dataset=dataset, file_paths=file_paths, class_names=class_names)


def compute_class_weights(config: ProjectConfig) -> dict[int, float]:
    """Return balanced training weights from the prepared manifest."""
    summary_path = config.path("data", "processed_dir") / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    class_names = list(config.section("data")["class_names"])
    counts = [int(summary["counts"]["train"][name]) for name in class_names]
    total = sum(counts)
    if any(count <= 0 for count in counts):
        raise ValueError(f"Every training class must be non-empty; got {counts}")
    return {index: total / (len(counts) * count) for index, count in enumerate(counts)}
