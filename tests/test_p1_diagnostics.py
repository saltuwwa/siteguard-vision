from pathlib import Path
import json
import math

import numpy as np
import pytest

from siteguard.config.settings import RulesConfig, RuntimeConfig
from siteguard.diagnostics import Timing, PipelineDiagnostics, gpu_memory_snapshot
from siteguard.inference.base import Detection
from siteguard.rules.temporal import State, ViolationEngine
from siteguard.tracking.association import AssociationDiagnostics, HeadEvidence, HeadState, associate_heads
from siteguard.tracking.bytetrack import PersonTrack
from siteguard.pipeline import analyze
from test_ingestion import create_video
from test_pipeline import FixtureDetector, FixtureTracker


@pytest.mark.parametrize("head,people,reason", [
    (Detection((30, 12, 50, 35), .4, "barehead"), [PersonTrack(1, (0, 0, 100, 120), .9)], "low_detection_confidence"),
    (Detection((30, 12, 35, 17), .9, "barehead"), [PersonTrack(1, (0, 0, 100, 120), .9)], "head_too_small"),
    (Detection((30, 12, 50, 35), .9, "barehead"), [PersonTrack(1, (0, 0, 100, 120), .4)], "person_confidence_too_low"),
    (Detection((30, 12, 50, 35), .9, "barehead"), [], "no_matching_person"),
    (Detection((50, 12, 30, 35), .9, "barehead"), [], "invalid_head_geometry"),
    (Detection((math.nan, 12, 50, 35), .9, "barehead"), [], "invalid_head_geometry"),
    (Detection((0, 12, 20, 35), .9, "barehead"), [], "head_clipped"),
])
def test_rejection_accounting(head: Detection, people: list[PersonTrack], reason: str) -> None:
    diagnostics = AssociationDiagnostics()
    result = associate_heads(people, [head], image_shape=(120, 160), diagnostics=diagnostics)
    assert diagnostics.attempts == 1
    assert diagnostics.accepted == 0
    assert diagnostics.rejected == {reason: 1}
    assert all(e.state == HeadState.UNKNOWN for e in result.values())


def test_conflicts_duplicates_and_missing_heads() -> None:
    people = [PersonTrack(1, (0, 0, 100, 150), .9)]
    bare = Detection((30, 12, 50, 35), .9, "barehead")
    audit = AssociationDiagnostics()
    result = associate_heads(people, [bare, Detection(bare.box, .95, "helmet")], diagnostics=audit)
    assert result[1].reason == "conflicting_head_predictions"
    assert audit.rejected == {"conflicting_head_predictions": 2}
    audit = AssociationDiagnostics()
    assert associate_heads(people, [bare, bare], diagnostics=audit)[1].state == HeadState.BAREHEAD
    assert audit.accepted == 1 and audit.rejected == {"duplicate_head_prediction": 1}
    assert associate_heads(people, [])[1].state == HeadState.UNKNOWN
    far = Detection((65, 12, 85, 35), .9, "barehead")
    assert associate_heads(people, [bare, far])[1].reason == "multiple_distinct_heads"
    audit = AssociationDiagnostics()
    nested = [Detection((30, 12, 50, 52), .95, "barehead"),
              Detection((30, 12, 50, 28), .9, "barehead")]
    assert associate_heads(people, nested, diagnostics=audit)[1].state == HeadState.BAREHEAD
    assert audit.accepted == 1 and audit.rejected == {"duplicate_head_prediction": 1}


def test_ambiguity_accounting_and_unique_assignment() -> None:
    people = [PersonTrack(1, (0, 0, 100, 150), .9), PersonTrack(2, (1, 0, 101, 150), .9)]
    audit = AssociationDiagnostics()
    result = associate_heads(people, [Detection((30, 12, 50, 35), .95, "barehead")], diagnostics=audit)
    assert all(e.state == HeadState.UNKNOWN for e in result.values())
    assert audit.rejected == {"ambiguous_person_association": 1}


def test_distant_third_candidate_keeps_its_own_head() -> None:
    people = [PersonTrack(1, (0, 0, 100, 300), .9), PersonTrack(2, (1, 0, 101, 300), .9),
              PersonTrack(3, (45, 0, 145, 300), .9)]
    result = associate_heads(people, [Detection((42, 12, 62, 35), .95, "barehead"),
                                     Detection((110, 12, 130, 35), .95, "helmet")])
    assert result[1].state == result[2].state == HeadState.UNKNOWN
    assert result[3].state == HeadState.HELMET


def test_confirmed_incident_survives_brief_track_absence_and_expires() -> None:
    rules = ViolationEngine("cam", "session", "fixture", RulesConfig())
    for t in (0., .5, 1., 1.5):
        rules.observe(t, HeadEvidence(1, HeadState.BAREHEAD, .9))
    assert not rules.tick(2.5, set())
    assert rules.tracks[1].state == State.CONFIRMED
    ended = rules.tick(6., set())
    assert len(ended) == 1 and ended[0].endreason == "track_lost"
    assert ended[0].endtimestamp == 1.5
    assert not rules.finish()
    with pytest.raises(ValueError):
        rules.observe(5., HeadEvidence(2))


def test_diagnostic_timing_memory_is_bounded() -> None:
    metric = Timing()
    for i in range(10000):
        metric.add(float(i))
    report = metric.report()
    assert report["count"] == 10000 and report["retained_samples"] == 2048
    assert report["max_seconds"] == 9999


def test_invalid_confidence_is_counted_without_an_invented_histogram_bin() -> None:
    diagnostics = PipelineDiagnostics()
    report = diagnostics.detections([Detection((0, 0, 20, 20), math.nan, "barehead")])
    assert report["barehead_detections"] == 1
    assert sum(report["confidence_histograms"]["barehead"]) == 0
    assert diagnostics.counts["invalid_detection_confidence"] == 1


def test_optional_gpu_telemetry_cannot_mask_an_inference_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace
    def fail(*args):
        raise RuntimeError("unit driver reset")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, memory_allocated=fail)))
    assert gpu_memory_snapshot("0") == {"device": 0, "error_type": "RuntimeError"}
    assert gpu_memory_snapshot("cpu") is None


def test_pipeline_counters_match_real_fixture_behavior(tmp_path: Path) -> None:
    source = tmp_path/"fixture.mp4"
    create_video(source)
    out = tmp_path/"out"
    summary = analyze(str(source), out, "fixture", RuntimeConfig(inference_fps=5,
        database_url=f"sqlite:///{tmp_path/'events.sqlite'}"), FixtureDetector(), tracker=FixtureTracker())
    report = json.loads((out/"diagnostics.json").read_text())
    assert report["person_detections"] == report["barehead_detections"] == 10
    assert report["accepted_head_associations"] == 10
    assert report["confirmed_events"] == 1 and summary["confirmed_events"] == 1
    assert report["ended_events"] == 1
    assert report["suspected_violation_observations"] > 0
    assert summary["performance"]["main_thread"]["temporal_rules"]["count"] == 10
    assert summary["performance"]["event_io_thread"]["evidence_capture"]["count"] == 1


def test_detector_failure_saves_audit_and_closes_event(tmp_path: Path) -> None:
    source = tmp_path/"fixture.mp4"
    create_video(source, frames=100)
    class FailingDetector(FixtureDetector):
        calls = 0

        def detect(self, image: np.ndarray) -> list[Detection]:
            self.calls += 1
            if self.calls == 11:
                raise RuntimeError("injected test failure")
            return super().detect(image)
    out = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="injected test failure"):
        analyze(str(source), out, "fixture", RuntimeConfig(inference_fps=5,
            database_url=f"sqlite:///{tmp_path/'events.sqlite'}"), FailingDetector(), tracker=FixtureTracker())
    report = json.loads((out/"metrics.json").read_text())
    assert report["status"] == "failed" and report["processed_frames"] == 10
    events = [json.loads(line) for line in (out/"events.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == "ended" and events[-1]["endreason"] == "processing_error"
    assert len({e["eventid"] for e in events}) == 1
    assert len(list((out/"evidence").rglob("*.jpg"))) == 1


def test_auto_device_uses_a_healthy_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from siteguard.inference.device import select_device
    monkeypatch.setattr("siteguard.inference.device.doctor", lambda: {"cuda_ok": False,
        "gpus": [{"index": 0, "compatible": False},
                 {"index": 1, "compatible": True, "fp16_ok": True}]})
    assert select_device("auto", True) == ("1", True)
    with pytest.raises(RuntimeError):
        select_device("0")
    with pytest.raises(ValueError):
        select_device("cpu", True)


def test_cooldown_does_not_display_a_bare_head_as_compliant() -> None:
    rules = ViolationEngine("cam", "session", "fixture", RulesConfig())
    for t in (0., .5, 1., 1.5):
        rules.observe(t, HeadEvidence(1, HeadState.BAREHEAD, .9))
    for t in (2., 2.5, 3.):
        rules.observe(t, HeadEvidence(1, HeadState.HELMET, .9))
    assert rules.tracks[1].state == State.CLEARED
    assert not rules.observe(3.5, HeadEvidence(1, HeadState.BAREHEAD, .9))
    assert rules.tracks[1].state == State.SUSPECT
    assert rules.tracks[1].decision_reason == "cooldown_active"


def test_brief_absence_cannot_complete_a_helmet_clearance_interval() -> None:
    rules = ViolationEngine("cam", "session", "fixture", RulesConfig())
    for t in (0., .5, 1., 1.5):
        rules.observe(t, HeadEvidence(1, HeadState.BAREHEAD, .9))
    rules.observe(2., HeadEvidence(1, HeadState.HELMET, .9))
    rules.tick(2.25, set())
    assert not rules.observe(2.5, HeadEvidence(1, HeadState.HELMET, .9))
    assert not rules.observe(3., HeadEvidence(1, HeadState.HELMET, .9))
    assert rules.tracks[1].event is not None
    assert rules.observe(3.5, HeadEvidence(1, HeadState.HELMET, .9))[0].status == "cleared"


def test_persistence_failure_remains_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path/"fixture.mp4"
    create_video(source)
    def fail(*args, **kwargs):
        raise OSError("unit evidence failure")
    monkeypatch.setattr("siteguard.events.sink.save_evidence", fail)
    out = tmp_path/"io-failed"
    with pytest.raises(RuntimeError, match="Event I/O failed"):
        analyze(str(source), out, "fixture", RuntimeConfig(inference_fps=5,
            database_url=f"sqlite:///{tmp_path/'events.sqlite'}"), FixtureDetector(), tracker=FixtureTracker())
    metrics = json.loads((out/"metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert metrics["cleanup_errors"]
    assert (out/"events.jsonl").read_text() == ""


def test_interruption_closes_incident_and_preserves_reports(tmp_path: Path) -> None:
    source = tmp_path/"fixture.mp4"
    create_video(source, frames=100)
    class InterruptingDetector(FixtureDetector):
        calls = 0

        def detect(self, image: np.ndarray) -> list[Detection]:
            self.calls += 1
            if self.calls == 11:
                raise KeyboardInterrupt
            return super().detect(image)
    out = tmp_path/"interrupted"
    report = analyze(str(source), out, "fixture", RuntimeConfig(inference_fps=5,
        database_url=f"sqlite:///{tmp_path/'events.sqlite'}"), InterruptingDetector(), tracker=FixtureTracker())
    assert report["failure"] == "Interrupted" and report["processed_frames"] == 10
    events = [json.loads(line) for line in (out/"events.jsonl").read_text().splitlines()]
    assert events[-1]["endreason"] == "processing_interrupted"
    assert events[-1]["endtimestamp"] <= 1.81


@pytest.mark.parametrize("reset", ["camera_cut_reset", "timestamp_discontinuity_reset"])
def test_cut_or_regression_starts_a_new_identity_scope(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, reset: str) -> None:
    from siteguard.ingestion.video import IngestionMetrics
    from test_resilience import frame

    class Reader:
        def __init__(self, *args):
            self.metrics = IngestionMetrics(source_fps=5)

        def __iter__(self):
            for i in range(8):
                timestamp = i*.5 if reset == "camera_cut_reset" or i < 4 else (i-4)*.5+.2
                f = frame("one-source-session", timestamp)
                if i >= 4 and reset == "camera_cut_reset":
                    f.image.fill(255)
                self.metrics.sampled += 1
                yield f

        def close(self):
            pass
    monkeypatch.setattr("siteguard.pipeline.VideoReader", Reader)
    out = tmp_path/reset
    metrics = analyze("test-only", out, "fixture", RuntimeConfig(inference_fps=5,
        database_url=f"sqlite:///{tmp_path/'events.sqlite'}"), FixtureDetector(), tracker=FixtureTracker())
    events = [json.loads(line) for line in (out/"events.jsonl").read_text().splitlines()]
    assert metrics["confirmed_events"] == 2
    assert len({e["streamsessionid"] for e in events}) == 2
    assert any(e["endreason"] == reset for e in events)
    assert json.loads((out/"diagnostics.json").read_text())["tracking_resets_by_reason"][reset] == 1
