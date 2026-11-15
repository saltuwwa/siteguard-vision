from pathlib import Path
import json

import cv2

from siteguard.config.settings import RuntimeConfig
from siteguard.inference.base import Detection, ImageArray
from siteguard.pipeline import analyze
from siteguard.tracking.bytetrack import PersonTrack
from test_ingestion import create_video


class FixtureDetector:
    """Deterministic software fixture. Not evidence of real detector accuracy."""
    version = "synthetic-test-only"

    def detect(self, image: ImageArray) -> list[Detection]:
        return [Detection((30, 10, 80, 115), 0.95, "person"),
                Detection((45, 12, 65, 32), 0.9, "barehead")]


class FixtureTracker:
    def update(self, detections: list[Detection], image: ImageArray) -> list[PersonTrack]:
        return [PersonTrack(1, (30, 10, 80, 115), .95)]

    def reset(self) -> None:
        pass


def test_mp4_inference_output(tmp_path: Path) -> None:
    path = tmp_path/"tiny.mp4"
    create_video(path)
    output = tmp_path/"run"
    settings = RuntimeConfig(inference_fps=5, database_url=f"sqlite:///{tmp_path/'events.sqlite'}")
    summary = analyze(str(path), output, "test", settings, FixtureDetector(),
                      tracker=FixtureTracker())
    assert summary["processed_frames"] == 10
    assert summary["accuracy_metrics"] is None
    assert summary["confirmed_events"] == 1
    assert json.loads((output/"metrics.json").read_text()) == summary
    assert len(list((output/"evidence").rglob("*.jpg"))) == 1
    from siteguard.storage.database import EventStore
    total, events = EventStore(settings.database_url).list_events()
    assert total == 1 and events[0].status == "ended"
    capture = cv2.VideoCapture(str(output/"annotated.mp4"))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    capture.release()
