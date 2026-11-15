from dataclasses import dataclass
import inspect
import os
from types import SimpleNamespace
from typing import Protocol

import numpy as np

from siteguard.inference.base import Box, Detection, ImageArray


@dataclass(frozen=True)
class PersonTrack:
    track_id: int
    box: Box
    confidence: float


class Tracker(Protocol):
    def update(self, detections: list[Detection], image: ImageArray) -> list[PersonTrack]: ...

    def reset(self) -> None: ...


class ByteTracker:
    """Use the pinned Ultralytics ByteTrack implementation, persons only.

    Tracker IDs are local to the pipeline's camera/session. No tracker object is
    shared across workers. Time discontinuities are reset by the pipeline.
    """
    def __init__(self, fps: float, buffer_seconds: float = 2.0) -> None:
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics.trackers.byte_tracker import BYTETracker

        uses_frame_rate = "frame_rate" in inspect.signature(BYTETracker).parameters
        arguments = SimpleNamespace(
            track_high_thresh=0.4, track_low_thresh=0.1, new_track_thresh=0.5,
            track_buffer=max(1, round((30 if uses_frame_rate else fps)*buffer_seconds)),
            match_thresh=0.8, fuse_score=True)
        self.impl = (BYTETracker(arguments, frame_rate=max(1, round(fps)))
                     if uses_frame_rate else BYTETracker(arguments))

    def update(self, detections: list[Detection], image: ImageArray) -> list[PersonTrack]:
        from ultralytics.engine.results import Boxes

        people = [d for d in detections if d.label == "person"]
        data = np.asarray([[*d.box, d.confidence, 0] for d in people], dtype=np.float32).reshape(-1, 6)
        tracks = self.impl.update(Boxes(data, image.shape[:2]), img=image)
        return [PersonTrack(int(row[4]), (float(row[0]), float(row[1]), float(row[2]), float(row[3])), float(row[5]))
                for row in tracks]

    def reset(self) -> None:
        self.impl.reset()
