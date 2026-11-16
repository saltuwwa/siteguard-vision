from __future__ import annotations

import math
import os
import queue
import threading
import time
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2

from siteguard.config.settings import RuntimeConfig
from siteguard.inference.base import ImageArray
from siteguard.diagnostics import StageTimings


class MediaError(RuntimeError):
    pass


class TimestampSampler:
    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("Sampling FPS must be positive")
        self.period = 1.0 / fps
        self.next_at: float | None = None
        self.last: float | None = None

    def accept(self, timestamp: float) -> bool:
        if not math.isfinite(timestamp) or (self.last is not None and timestamp < self.last):
            raise ValueError("Nonmonotonic or invalid source timestamp")
        self.last = timestamp
        if self.next_at is None:
            self.next_at = timestamp
        if timestamp + 1e-7 < self.next_at:
            return False
        periods = math.floor((timestamp - self.next_at + 1e-7) / self.period) + 1
        self.next_at += periods * self.period
        return True


@dataclass(frozen=True)
class Frame:
    image: ImageArray
    source_timestamp: float
    decoded_at: datetime
    decoded_monotonic: float
    camera_id: str
    stream_session_id: str
    index: int
    timestamp_basis: str
    decoder_timestamp: float | None = None
    original_shape: tuple[int, int] | None = None


@dataclass
class IngestionMetrics:
    decoded: int = 0
    sampled: int = 0
    sampling_skipped: int = 0
    dropped: int = 0
    reconnects: int = 0
    source_fps: float = 0.0
    timestamp_fallbacks: int = 0
    errors: list[str] = field(default_factory=list)


class VideoReader:
    """Decode on one thread; bounded queue blocks offline and drops explicitly live.

    OpenCV FFmpeg exposes file PTS via POS_MSEC. RTSP PTS is retained when monotonic;
    if unavailable, timestamp_basis explicitly identifies receive-monotonic fallback.
    RTSP timestamps are session relative, not camera UTC or RTP/NTP timestamps.
    """
    def __init__(self, source: str, camera_id: str, config: RuntimeConfig,
                 live: bool = False) -> None:
        if not live and not Path(source).is_file():
            raise FileNotFoundError("Input file does not exist")
        self.source = source
        self.camera_id = camera_id
        self.config = config
        self.live = live
        self.metrics = IngestionMetrics()
        self.queue: queue.Queue[Frame] = queue.Queue(config.queue_size)
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None
        self.performance = StageTimings()

    def _open(self) -> cv2.VideoCapture:
        if self.live:
            # FFmpeg capture options are process global: one transport per worker process.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.config.rtsp_transport}|fflags;nobuffer")
            os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
            cv2.setLogLevel(0)
        capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.read_timeout_ms])
        if not capture.isOpened():
            capture.release()
            raise MediaError("Cannot open video source (source URL intentionally omitted)")
        return capture

    def _enqueue(self, frame: Frame) -> None:
        if not self.live:
            while not self.stop_event.is_set():
                try:
                    self.queue.put(frame, timeout=0.1)
                    return
                except queue.Full:
                    continue
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.metrics.dropped += 1
            if self.config.live_drop_policy == "drop_oldest":
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(frame)

    def _decode_session(self) -> None:
        capture = self._open()
        sampler = TimestampSampler(self.config.inference_fps)
        session = uuid4().hex
        started = time.monotonic()
        previous = -1.0
        index = 0
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.metrics.source_fps = fps if math.isfinite(fps) else 0.0
            while not self.stop_event.is_set():
                with self.performance.measure("decode_read"):
                    ok, image = capture.read()
                if not ok:
                    if self.live:
                        raise MediaError("Live decoder disconnected")
                    if index == 0 or (expected > 0 and index < expected):
                        raise MediaError(f"Unexpected decode termination at frame {index}/{expected}")
                    return
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                raw_pts = pts if math.isfinite(pts) else None
                basis = "decoder_pts"
                if math.isfinite(pts) and pts < previous:
                    raise MediaError("Source timestamp regressed; refusing to bridge identities")
                if not math.isfinite(pts) or pts == previous:
                    if self.live:
                        pts = time.monotonic() - started
                        basis = "receive_monotonic"
                    elif math.isfinite(fps) and fps > 0:
                        pts = index / fps
                        basis = "frame_index_fps"
                    else:
                        raise MediaError("Neither valid timestamps nor source FPS available")
                    self.metrics.timestamp_fallbacks += 1
                if pts <= previous:
                    raise MediaError("Decoder timestamp discontinuity; session reset required")
                previous = pts
                self.metrics.decoded += 1
                if sampler.accept(pts):
                    height, width = image.shape[:2]
                    original_shape = (height, width)
                    if self.config.max_width is not None and width > self.config.max_width:
                        new_height = max(2, round(height * self.config.max_width / width))
                        with self.performance.measure("decode_resize"):
                            image = cv2.resize(image, (self.config.max_width, new_height),
                                               interpolation=cv2.INTER_AREA)
                    self.metrics.sampled += 1
                    self._enqueue(Frame(image, pts, datetime.now(UTC), time.monotonic(),
                                        self.camera_id, session, index, basis, raw_pts, original_shape))
                else:
                    self.metrics.sampling_skipped += 1
                index += 1
        finally:
            capture.release()

    def _run(self) -> None:
        failures = 0
        try:
            while not self.stop_event.is_set():
                try:
                    self._decode_session()
                    return
                except MediaError as exc:
                    self.metrics.errors.append(str(exc))
                    failures += 1
                    if not self.live or failures > self.config.reconnect_attempts:
                        raise
                    self.metrics.reconnects += 1
                    logging.getLogger("siteguard.ingestion").warning("source_reconnecting", extra={
                        "cameraid": self.camera_id, "attempt": failures})
                    # Old frames must not survive a reconnect.
                    while True:
                        try:
                            self.queue.get_nowait()
                            self.metrics.dropped += 1
                        except queue.Empty:
                            break
                    self.stop_event.wait(min(30, self.config.reconnect_delay * 2**min(failures-1, 5)))
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    def __iter__(self) -> Iterator[Frame]:
        if self.thread is not None:
            raise RuntimeError("VideoReader instances are single use")
        self.thread = threading.Thread(target=self._run, name="video-decoder", daemon=True)
        self.thread.start()
        try:
            while not (self.done.is_set() and self.queue.empty()):
                if self.stop_event.is_set():
                    break
                try:
                    yield self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            if self.error:
                raise self.error
        finally:
            self.close()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=(self.config.read_timeout_ms+self.config.open_timeout_ms)/1000 + 2)
            if self.thread.is_alive():
                raise MediaError("Decoder did not stop within configured timeout")
