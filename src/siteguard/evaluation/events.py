from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import linear_sum_assignment

from siteguard.config.settings import StrictModel
from siteguard.events.schema import ViolationEvent
from siteguard.training.datasets import write_json


class Interval(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> "Interval":
        if self.end <= self.start:
            raise ValueError("Interval must have positive duration")
        return self


class GroundTruthIncident(Interval):
    incident_id: str
    camera_id: str
    source_id: str
    person_id: str


class VideoRecording(StrictModel):
    source_id: str
    camera_id: str
    source_groups: list[str] = Field(min_length=1)
    reviewed_intervals: list[Interval] = Field(min_length=1)
    timestamp_basis: Literal["decoder_pts", "frame_index_fps", "receive_monotonic"]
    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)


class VideoObservation(StrictModel):
    source_id: str
    camera_id: str
    timestamp: float = Field(ge=0)
    person_id: str
    state: Literal["helmet", "barehead", "unknown"]
    occluded: bool
    visible: bool
    box: tuple[float, float, float, float] | None = None


class VideoGroundTruth(StrictModel):
    schema_version: Literal[1] = 1
    label_origin: Literal["human_reviewed"]
    recordings: list[VideoRecording] = Field(min_length=1)
    incidents: list[GroundTruthIncident]
    observations: list[VideoObservation] = Field(default_factory=list)
    training_source_groups: list[str] = Field(default_factory=list)


def temporal_iou(start: float, end: float, gt: GroundTruthIncident, tolerance: float) -> float:
    left = max(0, gt.start-tolerance)
    right = gt.end+tolerance
    intersection = max(0.0, min(end, right)-max(start, left))
    union = max(end, right)-min(start, left)
    return intersection/union if union > 0 else 0.0


def evaluate_events(predictions: Path, ground_truth: Path, output: Path,
                    identity_mapping: Path, min_iou: float = .1, tolerance: float = .5,
                    training_groups: set[str] | None = None) -> dict[str, Any]:
    if not (0 < min_iou <= 1) or tolerance < 0:
        raise ValueError("Invalid temporal matching thresholds")
    gt = VideoGroundTruth.model_validate_json(ground_truth.read_text())
    recordings = {(r.camera_id, r.source_id): r for r in gt.recordings}
    if len(recordings) != len(gt.recordings):
        raise ValueError("Duplicate camera/source recording")
    declared_training = set(gt.training_source_groups) | (training_groups or set())
    eval_groups = {g for r in gt.recordings for g in r.source_groups}
    if declared_training & eval_groups:
        raise ValueError("Training/evaluation source leakage")
    exposure = 0.0
    for recording in gt.recordings:
        intervals = sorted(recording.reviewed_intervals, key=lambda i: i.start)
        if any(a.end > b.start for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Reviewed intervals overlap; camera-hour denominator would double count")
        exposure += sum(i.end-i.start for i in intervals)
    for incident in gt.incidents:
        coverage = recordings.get((incident.camera_id, incident.source_id))
        if not coverage or not any(i.start <= incident.start < incident.end <= i.end
                                    for i in coverage.reviewed_intervals):
            raise ValueError("Ground-truth incident outside reviewed recording coverage")
    if len({i.incident_id for i in gt.incidents}) != len(gt.incidents):
        raise ValueError("Ground-truth incident IDs must be unique")
    mappings = json.loads(identity_mapping.read_text())
    latest: dict[str, ViolationEvent] = {}
    for line in predictions.read_text().splitlines():
        if line.strip():
            event = ViolationEvent.model_validate_json(line)
            latest[event.eventid] = event
    pred: list[tuple[ViolationEvent, str, str]] = []
    excluded = 0
    for event in latest.values():
        key = f"{event.cameraid}/{event.streamsessionid}/{event.trackid}"
        identity = mappings.get(key)
        if identity is None:
            raise ValueError(f"Missing manually reviewed source/person correspondence: {key}")
        source_id = identity["source_id"]
        coverage = recordings.get((event.cameraid, source_id))
        if coverage is None:
            raise ValueError("Predicted source not in ground-truth manifest")
        if event.endtimestamp is None:
            raise ValueError("Finish recorded-video analysis before evaluating open events")
        if not any(i.start <= event.starttimestamp <= event.endtimestamp <= i.end
                   and i.start <= event.confirmedtimestamp <= i.end
                   for i in coverage.reviewed_intervals):
            excluded += 1
            continue
        pred.append((event, source_id, str(identity["person_id"])))
    weights = np.zeros((len(pred), len(gt.incidents)), dtype=np.float64)
    for p, (event, source_id, person_id) in enumerate(pred):
        assert event.endtimestamp is not None
        for g, incident in enumerate(gt.incidents):
            if (event.cameraid, source_id, person_id) != (
                    incident.camera_id, incident.source_id, incident.person_id):
                continue
            overlap = temporal_iou(event.starttimestamp, event.endtimestamp, incident, tolerance)
            if overlap >= min_iou:
                weights[p, g] = 1+overlap
    pairs = []
    if weights.size:
        rows, cols = linear_sum_assignment(-weights)
        pairs = [(int(p), int(g)) for p, g in zip(rows, cols, strict=True) if weights[p, g] > 0]
    matched_preds = {p for p, _ in pairs}
    duplicate_count = sum(p not in matched_preds and bool(weights[p].any()) for p in range(len(pred)))
    tp, fp, fn = len(pairs), len(pred)-len(pairs), len(gt.incidents)-len(pairs)
    delays = [pred[p][0].confirmedtimestamp-gt.incidents[g].start for p, g in pairs]
    result = {"schema_version": 1, "label_origin": gt.label_origin, "true_positive": tp,
        "false_positive": fp, "false_negative": fn,
        "event_precision": tp/(tp+fp) if tp+fp else None,
        "event_recall": tp/(tp+fn) if tp+fn else None,
        "false_alerts_per_camera_hour": fp/(exposure/3600),
        "reviewed_camera_hours": exposure/3600,
        "missed_incident_rate": fn/len(gt.incidents) if gt.incidents else None,
        "duplicate_event_rate": duplicate_count/len(pred) if pred else None,
        "detection_delay_seconds": {"p50": float(np.percentile(delays, 50)) if delays else None,
                                    "p95": float(np.percentile(delays, 95)) if delays else None},
        "matches": [{"eventid": pred[p][0].eventid, "incident_id": gt.incidents[g].incident_id,
                     "delay_seconds": pred[p][0].confirmedtimestamp-gt.incidents[g].start}
                    for p, g in pairs],
        "excluded_unreviewed_events": excluded, "matching": {"min_temporal_iou": min_iou,
        "ground_truth_padding_seconds": tolerance, "identity_mapping_required": True},
        "source_independence": "declared human provenance; checked against supplied training groups"}
    write_json(output, result)
    return result
