from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from siteguard.config.settings import RuntimeConfig
from siteguard.ingestion.video import Frame, MediaError, VideoReader
from siteguard.pipeline import analyze
from test_pipeline import FixtureDetector, FixtureTracker


def frame(session: str, timestamp: float) -> Frame:
    import time
    return Frame(np.zeros((120, 160, 3), dtype=np.uint8), timestamp, datetime.now(UTC),
                 time.monotonic(), "cam", session, round(timestamp*5), "decoder_pts")


@pytest.mark.parametrize("policy,expected", [("drop_oldest", 2.), ("drop_newest", 1.)])
def test_live_drop_policy(policy: str, expected: float) -> None:
    config = RuntimeConfig.model_validate({"queue_size": 1, "live_drop_policy": policy})
    reader = VideoReader("rtsp://private.invalid/ignored", "cam", config, live=True)
    reader._enqueue(frame("one", 1.))
    reader._enqueue(frame("one", 2.))
    assert reader.queue.get_nowait().source_timestamp == expected
    assert reader.metrics.dropped == 1


def test_reconnect_budget_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = VideoReader("rtsp://private.invalid", "cam", RuntimeConfig(
        reconnect_attempts=2, reconnect_delay=.001), live=True)
    def disconnect() -> None:
        raise MediaError("test disconnect (no URL)")
    monkeypatch.setattr(reader, "_decode_session", disconnect)
    with pytest.raises(MediaError):
        list(reader)
    assert reader.metrics.reconnects == 2
    assert reader.thread is not None and not reader.thread.is_alive()


def test_sessions_reset_temporal_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class SessionReader:
        def __init__(self, *args: Any) -> None:
            from siteguard.ingestion.video import IngestionMetrics
            self.metrics = IngestionMetrics(source_fps=5)

        def __iter__(self) -> Any:
            for session in ("session-a", "session-b"):
                # Each session too short to confirm; combining them would create a false event.
                for t in (0., .25, .5, .75, 1.):
                    self.metrics.decoded += 1
                    self.metrics.sampled += 1
                    yield frame(session, t)

        def close(self) -> None:
            pass

    monkeypatch.setattr("siteguard.pipeline.VideoReader", SessionReader)
    summary = analyze("unused", tmp_path/"output", "cam",
        RuntimeConfig(database_url=f"sqlite:///{tmp_path/'db.sqlite'}"),
        FixtureDetector(), tracker=FixtureTracker())
    assert summary["confirmed_events"] == 0
