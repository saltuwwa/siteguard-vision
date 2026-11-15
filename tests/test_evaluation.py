import json
from pathlib import Path

import pytest

from siteguard.config.settings import RulesConfig
from siteguard.evaluation.events import evaluate_events
from siteguard.rules.temporal import ViolationEngine
from siteguard.tracking.association import HeadEvidence, HeadState


def test_source_disjoint_matching_duplicate_and_delay(tmp_path: Path) -> None:
    gt = {"schema_version": 1, "label_origin": "human_reviewed", "recordings": [
        {"source_id": "fixture", "camera_id": "cam", "source_groups": ["test-video"],
         "reviewed_intervals": [{"start": 0, "end": 3600}], "timestamp_basis": "decoder_pts",
         "reviewer": "synthetic-unit-test", "reviewed_at": "2026-11-15"}], "incidents": [
        {"incident_id": "i1", "camera_id": "cam", "source_id": "fixture", "person_id": "p1",
         "start": 0, "end": 2}]}
    ground = tmp_path/"gt.json"
    ground.write_text(json.dumps(gt))
    rules = ViolationEngine("cam", "session", "test-only", RulesConfig())
    for t in (0., .5, 1., 1.5):
        rules.observe(t, HeadEvidence(1, HeadState.BAREHEAD, .9))
    event = rules.finish()[0]
    duplicate = event.model_copy(update={"eventid": "duplicate"})
    pred = tmp_path/"pred.jsonl"
    pred.write_text(event.model_dump_json()+"\n"+duplicate.model_dump_json())
    mapping = tmp_path/"mapping.json"
    mapping.write_text(json.dumps({"cam/session/1": {"source_id": "fixture", "person_id": "p1"}}))
    report = evaluate_events(pred, ground, tmp_path/"report.json", mapping)
    assert report["event_precision"] == .5
    assert report["event_recall"] == 1.
    assert report["duplicate_event_rate"] == .5
    assert report["false_alerts_per_camera_hour"] == 1.
    assert report["detection_delay_seconds"]["p50"] == 1.5
    with pytest.raises(ValueError, match="leakage"):
        evaluate_events(pred, ground, tmp_path/"no.json", mapping, training_groups={"test-video"})


def test_unreviewed_data_rejected(tmp_path: Path) -> None:
    ground = tmp_path/"bad.json"
    ground.write_text('{"schema_version":1,"label_origin":"pseudo_labels","recordings":[],"incidents":[]}')
    with pytest.raises(ValueError):
        evaluate_events(tmp_path/"pred", ground, tmp_path/"out", tmp_path/"mapping")
