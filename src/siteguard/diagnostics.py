"""Bounded runtime measurements, independent of ML libraries and API schemas."""
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterator

from siteguard.inference.base import Detection


@dataclass
class Timing:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=2048))

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        self.maximum = max(self.maximum, seconds)
        self.samples.append(seconds)

    def report(self) -> dict[str, Any]:
        values = sorted(self.samples)
        def quantile(q: float) -> float | None:
            if not values:
                return None
            index = (len(values)-1)*q
            lo, hi = math.floor(index), math.ceil(index)
            return values[lo]+(values[hi]-values[lo])*(index-lo)
        return {"count": self.count, "total_seconds": self.total,
                "max_seconds": self.maximum, "retained_samples": len(values),
                "p50_seconds": quantile(.5), "p95_seconds": quantile(.95),
                "percentile_scope": "last 2048 observations"}


class StageTimings:
    def __init__(self) -> None:
        self.stages: dict[str, Timing] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter()-started)

    def add(self, name: str, seconds: float) -> None:
        self.stages.setdefault(name, Timing()).add(seconds)

    def report(self) -> dict[str, Any]:
        return {name: values.report() for name, values in self.stages.items()}


class PipelineDiagnostics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.unknown_reasons: Counter[str] = Counter()
        self.resets: Counter[str] = Counter()
        self.confidence = {name: [0]*10 for name in ("person", "helmet", "barehead")}
        # Fixed names make zero activity distinguishable from missing instrumentation.
        for name in ("processed_frames", "person_detections", "helmet_detections",
                     "barehead_detections", "active_person_track_observations",
                     "head_association_attempts", "accepted_head_associations",
                     "rejected_head_associations", "unknown_worker_observations",
                     "suspected_violation_observations", "confirmed_events",
                     "cleared_events", "expired_events", "tracking_resets",
                     "ended_events", "expired_tracks", "cooldown_suppressed_observations",
                     "insufficient_temporal_evidence"):
            self.counts[name] = 0

    def detections(self, detections: list[Detection]) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        hist = {name: [0]*10 for name in self.confidence}
        for detection in detections:
            if detection.label in self.confidence:
                name = detection.label
                counts[f"{name}_detections"] += 1
                if not math.isfinite(detection.confidence) or not 0 <= detection.confidence <= 1:
                    counts["invalid_detection_confidence"] += 1
                    continue
                index = min(9, max(0, int(detection.confidence*10)))
                hist[name][index] += 1
                self.confidence[name][index] += 1
        self.counts.update(counts)
        return {**{f"{name}_detections": counts[f"{name}_detections"] for name in hist},
                "confidence_histograms": hist}

    def report(self) -> dict[str, Any]:
        return {"schema_version": 1, **dict(self.counts),
                "rejected_associations_by_reason": dict(self.rejections),
                "unknown_workers_by_reason": dict(self.unknown_reasons),
                "tracking_resets_by_reason": dict(self.resets),
                "confidence_histograms": self.confidence,
                "confidence_bin_edges": [i/10 for i in range(11)],
                "detection_scope": "adapter outputs after detector threshold; not pre-threshold logits",
                "observation_scope": "counts across processed frames, not unique people/incidents"}


def gpu_memory_snapshot(device: str) -> dict[str, Any] | None:
    index = device.removeprefix("cuda:")
    if not index.isdigit():
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return {"device": int(index), "error_type": "CUDAUnavailable"}
        return {"device": int(index), "allocated_bytes": torch.cuda.memory_allocated(int(index)),
                "reserved_bytes": torch.cuda.memory_reserved(int(index)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(int(index)),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(int(index)),
                "scope": "process allocator since last reset; includes model initialization"}
    except Exception as exc:
        # Optional telemetry must never erase the original inference failure report.
        return {"device": int(index), "error_type": type(exc).__name__}
