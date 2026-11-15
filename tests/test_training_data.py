import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from siteguard.training.datasets import build_dataset, validate_datasets
from siteguard.training.runner import validate_built_data


def test_reviewed_build_and_mutation_guard(tmp_path: Path) -> None:
    root = tmp_path/"raw"
    provenance = {}
    for seed, split in enumerate(("train", "val")):
        image = root/"images"/split/"image.png"
        label = root/"labels"/split/"image.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        Image.fromarray(np.random.default_rng(seed).integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(image)
        label.write_text("0 0.5 0.5 0.3 0.3\n1 0.5 0.2 0.1 0.1\n2 0.5 0.7 0.1 0.1\n")
        provenance[f"images/{split}/image.png"] = {"groups": [f"synthetic-test-{split}"],
                                                   "origin": "synthetic unit test fixture"}
    source_yaml = root/"data.yaml"
    source_yaml.write_text(yaml.safe_dump({"names": ["body", "hat", "head"],
                                         "train": "images/train", "val": "images/val"}))
    (root/"manifest.json").write_text(json.dumps({"name": "synthetic-test-fixture",
        "source_url": "local:test-only", "license": "CC0-1.0", "version": "test-v1",
        "archive_sha256": "test-only-not-a-real-dataset", "downloaded_at": "2026-11-15"}))
    provenance_file = tmp_path/"provenance.json"
    provenance_file.write_text(json.dumps(provenance))
    review_file = tmp_path/"review.json"
    config = tmp_path/"datasets.yaml"
    config.write_text(yaml.safe_dump({"sources": [{"name": "synthetic-test-fixture",
        "root": str(root), "yaml_file": str(source_yaml),
        "mapping": {"body": "person", "hat": "helmet", "head": "barehead"},
        "review_file": str(review_file), "provenance_file": str(provenance_file)}],
        "report": str(tmp_path/"report.json"), "output": str(tmp_path/"built")}))
    report = validate_datasets(config)
    assert not report["errors"] and not report["leakage"]
    with pytest.raises(ValueError, match="Review"):
        build_dataset(config)
    review_file.write_text(json.dumps({"reviewer": "synthetic-unit-test",
        "reviewed_at": "2026-11-15", "fingerprint": report["sources"][0]["fingerprint"],
        "semantics_verified": True, "canonical_annotations_complete": True,
        "sample_paths": ["synthetic-unit-test-only"]}))
    built_yaml = build_dataset(config)
    assert validate_built_data(built_yaml)["independence_proven"]
    label = next((tmp_path/"built"/"labels").rglob("*.txt"))
    label.write_text("0 .1 .1 .1 .1")
    with pytest.raises(ValueError, match="changed"):
        validate_built_data(built_yaml)
