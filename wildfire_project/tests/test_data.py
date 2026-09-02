from pathlib import Path
from zipfile import ZipFile

import pytest

from wildfire_ml.data import safe_extract_archive


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "train.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.jpg", b"not-an-image")
        archive.writestr("nowildfire/a.jpg", b"not-an-image")
        archive.writestr("wildfire/b.jpg", b"not-an-image")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract_archive(
            archive_path, tmp_path / "output", ["nowildfire", "wildfire"]
        )
    assert not (tmp_path / "escape.jpg").exists()


def test_safe_extract_is_idempotent(tmp_path: Path) -> None:
    archive_path = tmp_path / "train.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nowildfire/a.jpg", b"a")
        archive.writestr("wildfire/b.jpg", b"b")
    destination = tmp_path / "output"

    records, first_extract = safe_extract_archive(
        archive_path, destination, ["nowildfire", "wildfire"]
    )
    _, second_extract = safe_extract_archive(
        archive_path, destination, ["nowildfire", "wildfire"]
    )

    assert first_extract is True
    assert second_extract is False
    assert len(records) == 2
    assert (destination / "wildfire" / "b.jpg").read_bytes() == b"b"

