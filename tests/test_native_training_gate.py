import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from siteguard.training.datasets import validate_datasets
from siteguard.training.runner import run_training, validate_source_data
from test_datasets import make_dataset


def test_native_source_keeps_names_checks_bytes_and_bounds_review_scope(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    source = cfg["sources"][0]
    root = Path(source["root"])
    Image.fromarray(np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
        root/"images/val/a.png")
    manifest = {"license": "CC-BY-4.0", "content_fingerprint": "synthetic-fixture-only"}
    (root/"manifest.json").write_text(json.dumps(manifest))
    source["label_schema"] = "source"
    review_file = tmp_path/"review.json"
    source["review_file"] = str(review_file)
    config.write_text(yaml.safe_dump(cfg))
    report = validate_datasets(config)
    review_file.write_text(json.dumps({"reviewer": "synthetic test", "kind": "agent_visual",
        "reviewed_at": "2026-11-15", "fingerprint": report["sources"][0]["fingerprint"],
        "semantics_verified": True, "label_ids_preserved": True,
        "sample_paths": ["synthetic-test-only"], "training_scope": "one_epoch_smoke_only"}))
    validate_datasets(config)
    data = yaml.safe_load((root/"data.yaml").read_text())
    data["path"] = str(root.resolve())
    local_data = tmp_path/"local.yaml"
    local_data.write_text(yaml.safe_dump(data))
    checked = validate_source_data(local_data, Path(cfg["report"]), smoke_only=True)
    assert checked["sources"][0]["classes"] == {"0": "nohelmet"}
    with pytest.raises(ValueError, match="one-epoch smoke only"):
        validate_source_data(local_data, Path(cfg["report"]), smoke_only=False)
    (root/"labels/train/a.txt").write_text("0 .5 .5 .2 .2")
    with pytest.raises(ValueError, match="changed"):
        validate_source_data(local_data, Path(cfg["report"]), smoke_only=True)


def test_smoke_cannot_start_long_run(tmp_path: Path) -> None:
    config = tmp_path/"train.yaml"
    config.write_text("stage: smoke\nepochs: 2\n")
    with pytest.raises(ValueError, match="exactly one epoch"):
        run_training(config)
