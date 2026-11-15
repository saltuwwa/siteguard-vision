from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5

from siteguard.config.settings import RulesConfig
from siteguard.events.schema import SupportingObservation, ViolationEvent
from siteguard.tracking.association import HeadEvidence, HeadState


class State(StrEnum):
    UNKNOWN = "UNKNOWN"
    COMPLIANT = "COMPLIANT"
    SUSPECT = "SUSPECT"
    CONFIRMED = "CONFIRMED"
    CLEARED = "CLEARED"


@dataclass
class TrackState:
    state: State = State.UNKNOWN
    observations: deque[SupportingObservation] = field(default_factory=deque)
    last_timestamp: float | None = None
    last_barehead: float | None = None
    compliant_since: float | None = None
    cooldown_until: float = -1
    event: ViolationEvent | None = None


class ViolationEngine:
    def __init__(self, camera_id: str, session_id: str, model_version: str,
                 config: RulesConfig) -> None:
        self.camera_id = camera_id
        self.session_id = session_id
        self.model_version = model_version
        self.config = config
        self.tracks: dict[int, TrackState] = {}

    def _close(self, record: TrackState, reason: str, cleared: bool = False) -> ViolationEvent | None:
        event = record.event
        if event is None:
            return None
        event.endtimestamp = record.last_barehead
        event.status = "cleared" if cleared else "ended"
        event.endreason = reason
        record.event = None
        record.observations.clear()
        record.cooldown_until = (record.last_timestamp or 0) + self.config.cooldown_seconds
        record.state = State.CLEARED if cleared else State.UNKNOWN
        return event.model_copy(deep=True)

    def observe(self, timestamp: float, evidence: HeadEvidence) -> list[ViolationEvent]:
        record = self.tracks.setdefault(evidence.track_id, TrackState())
        if record.last_timestamp is not None and timestamp <= record.last_timestamp:
            raise ValueError("Per-track observations must have strictly increasing source timestamps")
        updates: list[ViolationEvent] = []
        if record.last_timestamp is not None and timestamp-record.last_timestamp > self.config.max_gap:
            record.observations.clear()
            record.compliant_since = None
        record.last_timestamp = timestamp
        while record.observations and timestamp-record.observations[0].timestamp > self.config.window_seconds:
            record.observations.popleft()
        visible_bare = evidence.state == HeadState.BAREHEAD and evidence.confidence >= self.config.min_confidence
        visible_helmet = evidence.state == HeadState.HELMET and evidence.confidence >= self.config.min_confidence
        if record.event and record.last_barehead is not None and (
                timestamp-record.last_barehead > self.config.window_seconds):
            closed = self._close(record, "evidence_lost")
            if closed:
                updates.append(closed)
        if visible_bare:
            record.compliant_since = None
            if record.last_barehead is not None and timestamp-record.last_barehead > self.config.max_gap:
                record.observations.clear()
            record.last_barehead = timestamp
            observation = SupportingObservation(timestamp=timestamp,
                confidence=evidence.confidence, head_box=evidence.box)
            record.observations.append(observation)
            if record.event:
                event = record.event
                event.observationcount += 1
                summary = event.confidencesummary
                summary["min"] = min(summary["min"], evidence.confidence)
                summary["max"] = max(summary["max"], evidence.confidence)
                summary["mean"] += (evidence.confidence-summary["mean"])/event.observationcount
                # Keep onset observations and most recent observations; count is uncapped.
                if len(event.supportingobservations) >= 128:
                    event.supportingobservations.pop(64)
                event.supportingobservations.append(observation)
                record.state = State.CONFIRMED
                updates.append(event.model_copy(deep=True))
            elif timestamp >= record.cooldown_until:
                record.state = State.SUSPECT
                first = record.observations[0].timestamp
                if (len(record.observations) >= self.config.min_observations
                        and timestamp-first >= self.config.min_duration):
                    confidences = [o.confidence for o in record.observations]
                    event = ViolationEvent(
                        eventid=str(uuid5(NAMESPACE_URL, f"{self.camera_id}/{self.session_id}/{evidence.track_id}/{first:.9f}")),
                        cameraid=self.camera_id, streamsessionid=self.session_id,
                        trackid=evidence.track_id, starttimestamp=first, confirmedtimestamp=timestamp,
                        modelversion=self.model_version,
                        confidencesummary={"min": min(confidences), "max": max(confidences),
                                           "mean": sum(confidences)/len(confidences)},
                        supportingobservations=list(record.observations), observationcount=len(confidences))
                    record.event = event
                    record.state = State.CONFIRMED
                    updates.append(event.model_copy(deep=True))
        elif visible_helmet:
            record.observations.clear()
            if record.compliant_since is None:
                record.compliant_since = timestamp
            if record.event:
                if timestamp-record.compliant_since >= self.config.clear_seconds:
                    closed = self._close(record, "visible_helmet", cleared=True)
                    if closed:
                        updates.append(closed)
            else:
                record.state = State.COMPLIANT
        else:
            record.compliant_since = None
            if not record.event:
                record.state = State.UNKNOWN
        return updates

    def tick(self, timestamp: float, visible_ids: set[int]) -> list[ViolationEvent]:
        updates = []
        for track_id, record in list(self.tracks.items()):
            if track_id in visible_ids:
                continue
            if record.last_timestamp is not None and timestamp-record.last_timestamp > self.config.max_gap:
                record.observations.clear()
                record.state = State.UNKNOWN
            if record.last_timestamp is not None and timestamp-record.last_timestamp > self.config.window_seconds:
                closed = self._close(record, "track_lost")
                if closed:
                    updates.append(closed)
                del self.tracks[track_id]
        return updates

    def finish(self, reason: str = "source_ended") -> list[ViolationEvent]:
        updates = []
        for record in self.tracks.values():
            closed = self._close(record, reason)
            if closed:
                updates.append(closed)
        self.tracks.clear()
        return updates
