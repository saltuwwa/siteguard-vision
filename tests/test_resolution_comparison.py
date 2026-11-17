"""Synthetic metadata checks comparison gates, never real detector accuracy."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import yaml

from siteguard.evaluation.objects import validation_content_fingerprint


@pytest.fixture
def experiment(tmp_path):
    images, labels = tmp_path/"images/val", tmp_path/"labels/val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (640, 640)).save(images/"unit.jpg")
    (labels/"unit.txt").write_text("0 .25 .25 .01 .01\n0 .75 .25 .05 .05\n")
    data = tmp_path/"data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "val": "images/val",
                                   "names": {0: "no_helmet", 1: "Person"}}))
    metadata = {"protocol": {"imgsz": 960, "batch": 2, "conf": .001},
                "checkpoint_sha256": "synthetic-metadata-only",
                "validation_content_sha256": validation_content_fingerprint(data),
                "data_yaml_sha256": "unit-config", "ultralytics_version": "unit-version",
                "confusion_classes": ["no_helmet", "Person", "background"],
                "results": {}, "classwise": [], "focus_fixed_counts": {},
                "no_helmet_to_helmet": 0}
    folders = [tmp_path/"reference", tmp_path/"candidate"]
    for index, folder in enumerate(folders):
        (folder/"errors").mkdir(parents=True)
        info = copy.deepcopy(metadata)
        info["protocol"]["imgsz"] = 960 if index == 0 else 1280
        (folder/"metrics.json").write_text(json.dumps(info))
        examples = ([{"image": str(images/"unit.jpg"), "errors": [
            {"gt": [0., .25, .25, .01, .01], "kind": "FN"}]}] if index == 0 else [])
        (folder/"errors/diagnostics.json").write_text(json.dumps({
            "classes": info["confusion_classes"], "examples": examples}))
    path = Path(__file__).resolve().parents[1]/"scripts/compare_checkpoints.py"
    spec = importlib.util.spec_from_file_location("unit_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, data, folders, tmp_path/"comparison.json"


def invoke(monkeypatch, experiment, *flags):
    module, data, folders, output = experiment
    monkeypatch.setattr(sys, "argv", ["compare", "--reference", str(folders[0]),
        "--candidate", str(folders[1]), "--data", str(data), "--output", str(output), *flags])
    module.main()
    return json.loads(output.read_text())


def test_960_to_1280_keeps_fixed_source_size_bins(monkeypatch, experiment):
    before = [(folder/"metrics.json").read_bytes() for folder in experiment[2]]
    report = invoke(monkeypatch, experiment, "--allow-resolution-change", "--require-same-checkpoint")
    assert report["same_checkpoint"]
    reference = report["no_helmet_size_recall"]["smoke"]
    candidate = report["no_helmet_size_recall"]["baseline"]
    assert reference["under8"]["ground_truth"] == candidate["under8"]["ground_truth"] == 1
    assert reference["under8"]["correct"] == 0 and candidate["under8"]["correct"] == 1
    assert reference["32plus"]["correct"] == candidate["32plus"]["correct"] == 1
    assert [(folder/"metrics.json").read_bytes() for folder in experiment[2]] == before


def test_resolution_change_is_explicit_and_weights_are_bound(monkeypatch, experiment):
    with pytest.raises(ValueError, match="settings differ"):
        invoke(monkeypatch, experiment)
    metrics = experiment[2][1]/"metrics.json"
    info = json.loads(metrics.read_text())
    info["checkpoint_sha256"] = "different-unit-checkpoint"
    metrics.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="identical checkpoint"):
        invoke(monkeypatch, experiment, "--allow-resolution-change", "--require-same-checkpoint")
