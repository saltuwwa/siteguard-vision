"""Single-worker orchestration; decoder runs in a bounded producer thread."""
import json
import time
import logging
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


def analyze(source: str, output: Path, camera_id: str, config: RuntimeConfig,
            detector: Detector, live: bool = False, max_frames: int | None = None,
            tracker: Tracker | None = None) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"camera_id": camera_id, "source_type": "rtsp" if live else "file",
                "source_file_sha256": sha256(Path(source)) if not live and Path(source).is_file() else None,
                "model_version": detector.version,
                "effective_device": getattr(detector, "device", "injected-test-detector"),
                "effective_half": getattr(detector, "half", None),
                "configuration": config.model_dump(mode="json", exclude={"database_url"})}
    write_json(output/"run.json", metadata)
    reader = VideoReader(source, camera_id, config, live)
    video = AnnotatedVideo(output / "annotated.mp4", config.inference_fps)
    started = time.monotonic()
    count = 0
    tracker = tracker or ByteTracker(config.inference_fps, config.track_buffer_seconds)
    engine: ViolationEngine | None = None
    decode_session: str | None = None
    last_timestamp: float | None = None
    previous_histogram: Any = None
    event_ids: set[str] = set()
    store = EventStore(config.database_url)
    latencies: list[float] = []
    failure: str | None = None
    session_id = "not-started"
    with (output/"frames.jsonl").open("w", encoding="utf-8") as frames, \
            (output/"events.jsonl").open("w", encoding="utf-8") as event_log:
        sink = EventSink(store, output/"evidence", event_log, config.kafka_enabled)

        def persist(updates: list[ViolationEvent], image: ImageArray | None = None) -> None:
            for event in updates:
                first = event.eventid not in event_ids
                event_ids.add(event.eventid)
                sink.submit(event, image if first else None)

        try:
            for frame in reader:
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
                cv2.normalize(histogram, histogram)
                cut = previous_histogram is not None and cv2.compareHist(
                    histogram, previous_histogram, cv2.HISTCMP_BHATTACHARYYA) > config.cut_threshold
                gap = last_timestamp is not None and frame.source_timestamp-last_timestamp > config.track_buffer_seconds
                if decode_session != frame.stream_session_id or cut or gap:
                    if engine:
                        persist(engine.finish("camera_cut" if cut else "source_session_reset"))
                    tracker.reset()
                    decode_session = frame.stream_session_id
                    engine = ViolationEngine(camera_id, uuid4().hex, detector.version, config.rules)
                    logging.getLogger("siteguard.pipeline").info("tracking_session_started", extra={
                        "cameraid": camera_id, "sessionid": engine.session_id,
                        "reason": "camera_cut" if cut else "new_source_or_gap"})
                previous_histogram = histogram
                last_timestamp = frame.source_timestamp
                detections = detector.detect(frame.image)
                people = tracker.update(detections, frame.image)
                evidence = associate_heads(people, detections, config.rules.min_confidence,
                    config.rules.min_head_pixels, image_shape=frame.image.shape[:2])
                assert engine is not None
                session_id = engine.session_id
                for item in evidence.values():
                    persist(engine.observe(frame.source_timestamp, item), frame.image)
                persist(engine.tick(frame.source_timestamp, set(evidence)))
                if reader.metrics.source_fps > 0:
                    video.fps = min(config.inference_fps, reader.metrics.source_fps)
                annotations = [Detection(p.box, p.confidence,
                    f"track:{p.track_id}:{engine.tracks[p.track_id].state}") for p in people]
                video.write(frame.image, detections+annotations, f"{camera_id} t={frame.source_timestamp:.3f}s")
                frames.write(json.dumps({"camera_id": camera_id,
                    "stream_session_id": frame.stream_session_id, "frame_index": frame.index,
                    "source_timestamp": frame.source_timestamp,
                    "raw_decoder_timestamp": frame.decoder_timestamp,
                    "timestamp_basis": frame.timestamp_basis,
                    "tracking_session_id": engine.session_id,
                    "tracks": [asdict(p) for p in people],
                    "head_evidence": [asdict(e) for e in evidence.values()],
                    "decoded_at": frame.decoded_at.isoformat(),
                    "processed_unix": time.time(), "detections": [asdict(d) for d in detections]})+"\n")
                count += 1
                latency = time.monotonic()-frame.decoded_monotonic
                latencies.append(latency)
                if len(latencies) > 100000:
                    latencies.pop(0)
                if count % max(1, round(config.inference_fps)) == 0:
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
            try:
                if engine:
                    persist(engine.finish("processing_error" if failure else "source_ended"))
            finally:
                try:
                    reader.close()
                finally:
                    video.close()
                    sink.close()
                    store.update_camera(camera_id, session_id, {"state": "failed" if failure else "stopped",
                        "processed_frames": count, "failure": failure, "ingestion": asdict(reader.metrics)})
    duration = time.monotonic() - started
    summary = {"processed_frames": count, "elapsed_seconds": duration,
               "processed_fps": count/duration if duration else 0,
               "ingestion": asdict(reader.metrics), "model_version": detector.version,
               "confirmed_events": len(event_ids),
               "latency_seconds": {"basis": "decode-to-output (excludes camera/network latency)",
                   "sample_count": len(latencies),
                   "p50": sorted(latencies)[len(latencies)//2] if latencies else None,
                   "p95": sorted(latencies)[min(len(latencies)-1, int(len(latencies)*.95))] if latencies else None},
               "dropped_frame_rate": reader.metrics.dropped/reader.metrics.sampled if reader.metrics.sampled else 0,
               "kafka": "outbox_pending_external_publisher" if config.kafka_enabled else "disabled",
               "accuracy_metrics": None, "accuracy_note": "No ground truth provided"}
    write_json(output/"summary.json", summary)
    write_json(output/"metrics.json", summary)
    return summary
