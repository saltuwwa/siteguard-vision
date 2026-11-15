import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from siteguard.training.datasets import DatasetSource, download_dataset, inspect_source, sha256


def test_raw_import_manifest_and_reimport_guard(tmp_path: Path) -> None:
    archive = tmp_path/"fixture.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("fixture/README.txt", "SYNTHETIC IMPORT UNIT TEST; not a real dataset")
    root = download_dataset("construction-ppe", tmp_path/"raw", archive=archive,
                            version="synthetic-test-only")
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["archive_sha256"] == sha256(archive)
    assert manifest["version"] == "synthetic-test-only"
    assert manifest["downloaded_at"]
    with pytest.raises(ValueError, match="exists"):
        download_dataset("construction-ppe", tmp_path/"raw", archive=archive)


def test_authenticated_download_requires_explicit_action(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires"):
        download_dataset("hard-hat-workers", tmp_path)
    with pytest.raises(ValueError, match="SH17"):
        download_dataset("sh17", tmp_path)


def test_roboflow_directory_adapter(tmp_path: Path) -> None:
    root = tmp_path/"export"
    image = root/"train/images/a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "gray").save(image)
    label = root/"train/labels/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 .5 .5 .3 .3")
    yaml_file = root/"data.yaml"
    yaml_file.write_text("names: [head]\ntrain: ../train/images\n")
    (root/"manifest.json").write_text('{"license":"Public-Domain"}')
    report = inspect_source(DatasetSource(name="hard-hat-workers", root=root, yaml_file=yaml_file,
                                          mapping={"head": None}), tmp_path/"samples")
    assert not report["errors"]
    assert len(report["records"]) == 1
    assert any("normalized" in w for w in report["warnings"])
