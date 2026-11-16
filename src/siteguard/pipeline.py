"""Single-worker orchestration; decoder runs in a bounded producer thread."""
import json
import time
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from siteguard.config.settings import RuntimeConfig
from siteguard.inference.base import Detector, Detection
from siteguard.ingestion.output import AnnotatedVideo
from siteguard.ingestion.video import VideoReader
from siteguard.training.datasets import write_json
from siteguard.training.datasets import sha256
from siteguard.tracking.association import associate_heads
from siteguard.tracking.bytetrack import ByteTracker, Tracker
from siteguard.rules.temporal import ViolationEngine
from siteguard.events.schema import ViolationEvent
from siteguard.events.sink import EventSink
from siteguard.storage.database import EventStore
from siteguard.inference.base import ImageArray
from siteguard.diagnostics import PipelineDiagnostics, StageTimings, Timing, gpu_memory_snapshot
from siteguard.tracking.association import AssociationDiagnostics
from siteguard.rules.temporal import State


def analyze(source: str, output: Path, camera_id: str, config: RuntimeConfig,
            detector: Detector, live: bool = False, max_frames: int | None = None,
            tracker: Tracker | None = None) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"camera_id": camera_id, "source_type": "rtsp" if live else "file",
                "max_frames": max_frames,
                "source_file_sha256": sha256(Path(source)) if not live and Path(source).is_file() else None,
                "model_version": detector.version,
                "effective_device": getattr(detector, "device", "injected-test-detector"),
                "effective_half": getattr(detector, "half", None),
                "configuration": config.model_dump(mode="json", exclude={"database_url"})}
    write_json(output/"run.json", metadata)
    reader = VideoReader(source, camera_id, config, live)
    video = AnnotatedVideo(output / "annotated.mp4", config.inference_fps)
    started = time.monotonic()
    diagnostics, performance, latency = PipelineDiagnostics(), StageTimings(), Timing()
    temporal_reasons: Counter[str] = Counter()
    event_end_reasons: Counter[str] = Counter()
    count = 0
    tracker = tracker or ByteTracker(config.inference_fps, config.track_buffer_seconds)
    engine: ViolationEngine | None = None
    decode_session: str | None = None
    last_timestamp: float | None = None
    previous_histogram: Any = None
    event_ids: set[str] = set()
    store = EventStore(config.database_url)
    confirmed_count = 0
    failure: str | None = None
    session_id = "not-started"
    with (output/"frames.jsonl").open("w", encoding="utf-8") as frames, \
            (output/"events.jsonl").open("w", encoding="utf-8") as event_log:
        sink = EventSink(store, output/"evidence", event_log, config.kafka_enabled)

        def persist(updates: list[ViolationEvent], image: ImageArray | None = None) -> None:
            nonlocal confirmed_count
            for event in updates:
                first = event.status == "confirmed" and event.eventid not in event_ids
                with performance.measure("event_enqueue"):
                    sink.submit(event, image if first else None)
                if first:
                    event_ids.add(event.eventid)
                    confirmed_count += 1
                    diagnostics.counts["confirmed_events"] += 1
                if event.status != "confirmed":
                    event_ids.discard(event.eventid)
                    diagnostics.counts[f"{event.status}_events"] += 1
                    event_end_reasons[event.endreason or event.status] += 1
                    if event.endreason in {"track_lost", "evidence_lost"}:
                        diagnostics.counts["expired_events"] += 1

        try:
            iterator = iter(reader)
            while True:
                try:
                    with performance.measure("frame_wait"):
                        frame = next(iterator)
                except StopIteration:
                    break
                stage_started = time.perf_counter()
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
                cv2.normalize(histogram, histogram)
                cut = previous_histogram is not None and cv2.compareHist(
                    histogram, previous_histogram, cv2.HISTCMP_BHATTACHARYYA) > config.cut_threshold
                gap = last_timestamp is not None and frame.source_timestamp-last_timestamp > config.track_buffer_seconds
                regression = last_timestamp is not None and frame.source_timestamp <= last_timestamp
                performance.add("scene_change", time.perf_counter()-stage_started)
                reset_reason = None
                if decode_session != frame.stream_session_id or cut or gap or regression:
                    reset_reason = ("source_started" if decode_session is None else
                        "source_reconnect_reset" if decode_session != frame.stream_session_id else
                        "timestamp_discontinuity_reset" if regression else
                        "camera_cut_reset" if cut else "timestamp_gap_reset")
                    if engine:
                        persist(engine.finish(reset_reason))
                    tracker.reset()
                    diagnostics.counts["tracking_resets"] += 1
                    diagnostics.resets[reset_reason] += 1
                    decode_session = frame.stream_session_id
                    engine = ViolationEngine(camera_id, uuid4().hex, detector.version, config.rules)
                    logging.getLogger("siteguard.pipeline").info("tracking_session_started", extra={
                        "cameraid": camera_id, "sessionid": engine.session_id,
                        "reason": reset_reason})
                previous_histogram = histogram
                last_timestamp = frame.source_timestamp
                with performance.measure("detector_inference"):
                    detections = detector.detect(frame.image)
                frame_diagnostic = diagnostics.detections(detections)
                with performance.measure("bytetrack"):
                    people = tracker.update(detections, frame.image)
                association = AssociationDiagnostics()
                with performance.measure("head_association"):
                    evidence = associate_heads(people, detections, config.rules.min_confidence,
                        config.rules.min_head_pixels, image_shape=frame.image.shape[:2],
                        diagnostics=association)
                diagnostics.counts["active_person_track_observations"] += len(people)
                diagnostics.counts["head_association_attempts"] += association.attempts
                diagnostics.counts["accepted_head_associations"] += association.accepted
                diagnostics.counts["rejected_head_associations"] += sum(association.rejected.values())
                diagnostics.rejections.update(association.rejected)
                assert engine is not None
                session_id = engine.session_id
                updates: list[ViolationEvent] = []
                before_ids = set(engine.tracks)
                with performance.measure("temporal_rules"):
                    for item in evidence.values():
                        updates.extend(engine.observe(frame.source_timestamp, item))
                    updates.extend(engine.tick(frame.source_timestamp, set(evidence)))
                retired = len(before_ids-set(engine.tracks))
                diagnostics.counts["expired_tracks"] += retired
                persist(updates, frame.image)
                unknown = Counter(item.reason for item in evidence.values() if item.state == "unknown")
                suspected = sum(engine.tracks[p.track_id].state == State.SUSPECT for p in people)
                decisions = Counter(engine.tracks[p.track_id].decision_reason for p in people)
                temporal_reasons.update(decisions)
                diagnostics.unknown_reasons.update(unknown)
                diagnostics.counts["unknown_worker_observations"] += sum(unknown.values())
                diagnostics.counts["suspected_violation_observations"] += suspected
                diagnostics.counts["insufficient_temporal_evidence"] += decisions["insufficient_temporal_evidence"]
                diagnostics.counts["cooldown_suppressed_observations"] += decisions["cooldown_active"]
                frame_diagnostic.update({"active_person_tracks": len(people),
                    "head_association_attempts": association.attempts,
                    "accepted_head_associations": association.accepted,
                    "rejected_associations_by_reason": dict(association.rejected),
                    "unknown_workers_by_reason": dict(unknown), "suspected_violations": suspected,
                    "temporal_decisions": dict(decisions),
                    "temporal_insufficient_evidence": decisions["insufficient_temporal_evidence"],
                    "confirmed_events": sum(e.status == "confirmed" and
                        e.confirmedtimestamp == frame.source_timestamp for e in updates),
                    "active_confirmed_incidents": len(event_ids),
                    "closed_events": sum(e.status != "confirmed" for e in updates),
                    "cleared_events": sum(e.status == "cleared" for e in updates),
                    "expired_events": sum(e.endreason in {"track_lost", "evidence_lost"} for e in updates),
                    "expired_tracks": retired,
                    "tracking_reset_reason": reset_reason})
                if reader.metrics.source_fps > 0:
                    video.fps = min(config.inference_fps, reader.metrics.source_fps)
                annotations = [Detection(p.box, p.confidence,
                    f"track:{p.track_id}:{engine.tracks[p.track_id].state}") for p in people]
                setter = getattr(video, "set_track_evidence", None)
                if setter:
                    setter(evidence)
                with performance.measure("annotation_and_encode"):
                    video.write(frame.image, detections+annotations,
                                f"{camera_id} t={frame.source_timestamp:.3f}s; automated")
                stage_started = time.perf_counter()
                frames.write(json.dumps({"camera_id": camera_id,
                    "stream_session_id": frame.stream_session_id, "frame_index": frame.index,
                    "source_timestamp": frame.source_timestamp,
                    "raw_decoder_timestamp": frame.decoder_timestamp,
                    "timestamp_basis": frame.timestamp_basis,
                    "tracking_session_id": engine.session_id,
                    "tracks": [asdict(p) for p in people],
                    "head_evidence": [asdict(e) for e in evidence.values()],
                    "diagnostics": frame_diagnostic,
                    "original_shape": frame.original_shape, "processed_shape": frame.image.shape[:2],
                    "decoded_at": frame.decoded_at.isoformat(),
                    "processed_unix": time.time(), "detections": [asdict(d) for d in detections]})+"\n")
                count += 1
                performance.add("frame_jsonl", time.perf_counter()-stage_started)
                diagnostics.counts["processed_frames"] += 1
                latency.add(time.monotonic()-frame.decoded_monotonic)
                if count % max(1, round(config.inference_fps)) == 0:
                    with performance.measure("camera_heartbeat"):
                        store.update_camera(camera_id, session_id, {"state": "running",
                            "modelversion": detector.version, "processed_frames": count,
                            "ingestion": asdict(reader.metrics)})
                if max_frames and count >= max_frames:
                    break
        except KeyboardInterrupt:
            failure = "Interrupted"
        except BaseException as exc:
            failure = type(exc).__name__
            raise
        finally:
            cleanup_errors: list[Exception] = []
            try:
                if engine:
                    persist(engine.finish("processing_interrupted" if failure == "Interrupted" else
                                          "processing_error" if failure else "source_ended"))
            except Exception as exc:
                cleanup_errors.append(exc)
            for component in (reader, video, sink):
                try:
                    with performance.measure("shutdown"):
                        component.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
            primary_failure = failure
            failure = failure or (type(cleanup_errors[0]).__name__ if cleanup_errors else None)
            try:
                with performance.measure("camera_heartbeat"):
                    store.update_camera(camera_id, session_id, {"state": "failed" if failure else "stopped",
                        "processed_frames": count, "failure": failure, "ingestion": asdict(reader.metrics)})
            except Exception as exc:
                cleanup_errors.append(exc)
                failure = failure or type(exc).__name__
            gpu = gpu_memory_snapshot(str(getattr(detector, "device", "cpu")))
            duration = time.monotonic()-started
            latency_report = latency.report()
            summary = {"processed_frames": count, "elapsed_seconds": duration,
                "processed_fps": count/duration if duration else 0,
                "ingestion": asdict(reader.metrics), "model_version": detector.version,
                "confirmed_events": confirmed_count,
                "latency_seconds": {"basis": "decode-to-output; excludes camera/network and async event commit",
                    "sample_count": latency.count, "p50": latency_report["p50_seconds"],
                    "p95": latency_report["p95_seconds"], "retained_samples": len(latency.samples)},
                "dropped_frame_rate": reader.metrics.dropped/reader.metrics.sampled if reader.metrics.sampled else 0,
                "kafka": "outbox_pending_external_publisher" if config.kafka_enabled else "disabled",
                "accuracy_metrics": None, "accuracy_note": "No ground truth provided",
                "status": "failed" if failure else "completed", "failure": failure,
                "stopped_after_frame_limit": bool(max_frames and count >= max_frames),
                "unprocessed_sampled_frames": max(0, reader.metrics.sampled-reader.metrics.dropped-count),
                "cleanup_errors": [type(exc).__name__ for exc in cleanup_errors],
                "performance": {"main_thread": performance.report(),
                    "decoder_thread": reader.performance.report() if hasattr(reader, "performance") else {},
                    "video_output": video.performance.report() if hasattr(video, "performance") else {},
                    "event_io_thread": sink.performance.report(), "gpu_memory": gpu,
                    "sampling_target_fps": config.inference_fps,
                    "meets_sampling_target": count/duration >= config.inference_fps if duration and count else None,
                    "scope": "includes cold inference; concurrent thread totals overlap"}}
            report = diagnostics.report()
            report["temporal_reasons"] = dict(temporal_reasons)
            report["event_end_reasons"] = dict(event_end_reasons)
            write_json(output/"diagnostics.json", report)
            write_json(output/"summary.json", summary)
            write_json(output/"metrics.json", summary)
            if cleanup_errors and primary_failure is None:
                raise cleanup_errors[0]
    return summary
