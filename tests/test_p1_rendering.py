from pathlib import Path

import cv2
import numpy as np
import pytest

from siteguard.inference.base import Detection
from siteguard.ingestion.output import AnnotatedVideo
from siteguard.tracking.association import HeadEvidence


def test_detection_corners_and_confirmed_track_are_distinct(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []
    labels = []

    class Writer:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return True

        def write(self, canvas):
            captured.append(canvas.copy())

        def release(self):
            pass
    original_text = cv2.putText

    def text(image, label, *args):
        labels.append(label)
        return original_text(image, label, *args)
    monkeypatch.setattr("siteguard.ingestion.output.cv2.VideoWriter", Writer)
    monkeypatch.setattr("siteguard.ingestion.output.cv2.putText", text)
    output = AnnotatedVideo(tmp_path/"unit.mp4", 8)
    output.set_track_evidence({1: HeadEvidence(1)})
    image = np.zeros((121, 161, 3), dtype=np.uint8)
    output.write(image, [Detection((30, 45, 55, 65), .9, "barehead"),
                         Detection((10, 40, 150, 110), .9, "track:1:CONFIRMED")])
    canvas = captured[-1]
    assert canvas.shape[:2] == (122, 162)  # Pad, do not resample coordinates.
    assert tuple(canvas[65, 30]) == (0, 150, 255)
    assert tuple(canvas[65, 42]) == (0, 0, 0)  # A detection has corners, not a full body box.
    assert tuple(canvas[110, 10]) == (20, 20, 240)
    assert "worker #1 head=unknown event=CONFIRMED" in labels
    assert not image.any()
    output.write(np.zeros((80, 200, 3), dtype=np.uint8), [])
    assert not captured[-1][75:].any()  # Aspect change is padded rather than stretched.
    output.close()
