import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from siteguard.training.datasets import (
    DatasetSource,
    build_dataset,
    inspect_source,
    safe_extract,
    validate_datasets,
)


def make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        # Intentional duplicate to test leakage detection, never model ground truth.
        Image.fromarray(np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            root / "images" / split / "a.png")
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"names": ["nohelmet"], "train": "images/train", "val": "images/val"}))
    (root / "manifest.json").write_text(json.dumps({"license": "CC-BY-4.0"}))
    config = tmp_path / "datasets.yaml"
    config.write_text(yaml.safe_dump({"sources": [{"name": "synthetic-test-fixture",
        "root": str(root), "yaml_file": str(root/"data.yaml"),
        "mapping": {"nohelmet": "barehead"}}], "report": str(tmp_path/"report.json"),
        "output": str(tmp_path/"built")}))
    return config


def test_duplicate_leakage_and_review_gate(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)
    report = validate_datasets(config)
    assert report["leakage"][0]["exact"]
    assert not report["sources"][0]["reviewed"]
    assert not report["independence_proven"]
    with pytest.raises(ValueError, match="leakage"):
        build_dataset(config)


def test_invalid_and_missing_labels(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)
    raw = yaml.safe_load(config.read_text())["sources"][0]
    source = DatasetSource.model_validate(raw)
    (source.root / "labels/train/a.txt").write_text("0 nan 0.5 0.2 0.2")
    (source.root / "labels/val/a.txt").unlink()
    report = inspect_source(source, tmp_path/"samples")
    assert len(report["errors"]) >= 2


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path/"output")


def test_decimal_rounding_tolerance_does_not_accept_real_out_of_bounds(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)
    source = DatasetSource.model_validate(yaml.safe_load(config.read_text())["sources"][0])
    label = source.root/"labels/train/a.txt"
    label.write_text("0 .01 .5 .02001 .3\n")
    report = inspect_source(source, tmp_path/"samples")
    assert not report["errors"]
    assert len(report["rounded_boundary_rows"]) == 1
    label.write_text("0 .01 .5 .03 .3\n")
    report = inspect_source(source, tmp_path/"samples")
    assert any("outside normalized" in error for error in report["errors"])
