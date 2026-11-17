"""Synthetic checkpoint metadata exercises recovery; it is not a trained model."""
import csv
import json
from pathlib import Path

import pytest

from siteguard.training.runner import TrainConfig, prepare_resume
from siteguard.training.datasets import write_json


def test_resume_uses_checkpoint_history_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path/"old"
    weights = source/"weights"
    weights.mkdir(parents=True)
    best = weights/"best.pt"
    best.write_bytes(b"unit fixture only")
    (source/"results.csv").write_bytes(b"torn unrelated CSV\x00\x00")
    original = TrainConfig(name="old", device="1")
    (source/"experiment.json").write_text(json.dumps({"config": original.model_dump(mode="json"),
        "evaluation_fingerprint": "same-data", "initial_checkpoint_sha256": "original-init"}))
    (source/"per-class-epochs.json").write_text(json.dumps([
        {"epoch": 1}, {"epoch": 2}, {"epoch": 3}]))
    target = tmp_path/"new"
    target.mkdir()
    cfg = original.model_copy(update={"resume": best, "name": "new", "device": "0"})
    checkpoint = {"epoch": 1, "optimizer": {}, "train_results": {
        "epoch": [1, 2], "time": [1., 2.], "metrics/mAP50-95(B)": [.1, .2]}}
    lineage = prepare_resume(cfg, {"evaluation_fingerprint": "same-data"}, checkpoint, target)
    assert lineage["restored_epoch"] == 2
    assert best.read_bytes() == (target/"weights/best.pt").read_bytes()
    assert (source/"results.csv").read_bytes() == b"torn unrelated CSV\x00\x00"
    assert len(list(csv.DictReader((target/"results.csv").open()))) == 2
    assert json.loads((target/"per-class-epochs.json").read_text()) == [{"epoch": 1}, {"epoch": 2}]
    with pytest.raises(ValueError, match="new run directory"):
        prepare_resume(cfg, {"evaluation_fingerprint": "same-data"}, checkpoint, source)
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_resume(cfg, {"evaluation_fingerprint": "changed-data"}, checkpoint, tmp_path/"unused")


def test_report_sync_failure_preserves_previous_snapshot(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path/"report.json"
    write_json(target, {"old": True})
    def fail(*args):
        raise OSError("unit sync failure")
    monkeypatch.setattr("siteguard.training.datasets.os.fsync", fail)
    with pytest.raises(OSError, match="sync failure"):
        write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))
