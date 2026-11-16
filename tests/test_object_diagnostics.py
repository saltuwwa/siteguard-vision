"""Synthetic annotations test diagnostic bookkeeping, never model accuracy."""
from pathlib import Path

import yaml
from PIL import Image

from siteguard.evaluation.objects import frame_diagnostics, validation_content_fingerprint


def test_native_classes_wrong_assignment_missed_head_and_duplicate(tmp_path: Path) -> None:
    images = tmp_path / "images" / "val"
    labels = tmp_path / "labels" / "val"
    predictions = tmp_path / "predictions"
    for folder in (images, labels, predictions):
        folder.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(images / "fixture.jpg")
    names = {i: f"class{i}" for i in range(12)}
    names.update({0: "helmet", 5: "Person", 6: "no_helmet"})
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "val": "images/val", "names": names}))
    (labels / "fixture.txt").write_text(
        "6 .2 .2 .1 .1\n6 .8 .2 .1 .1\n5 .5 .6 .3 .5\n"
    )
    (predictions / "fixture.txt").write_text(
        "0 .2 .2 .1 .1 .9\n0 .2 .2 .1 .1 .8\n5 .5 .6 .3 .5 .95\n"
        "6 .8 .2 .1 .1 .1\n"  # Below the chosen diagnostic threshold.
    )
    report = frame_diagnostics(data, predictions, tmp_path / "errors")
    matrix = report["confusion_rows_gt_columns_prediction"]
    assert len(matrix) == 13
    assert matrix[6][0] == 1
    assert matrix[6][12] == 1
    assert matrix[12][0] == 1
    assert matrix[5][5] == 1
    assert sum(map(sum, matrix)) == 4
    assert {e["kind"] for e in report["examples"][0]["errors"]} == {"wrong_class", "FP", "FN"}
    assert Path(report["examples"][0]["visualization"]).is_file()
    before = validation_content_fingerprint(data)
    (labels/"fixture.txt").write_text("6 .2 .2 .1 .1\n")
    assert before != validation_content_fingerprint(data)
