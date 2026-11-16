from pathlib import Path

import cv2
import time

from siteguard.inference.base import Detection, ImageArray
from siteguard.tracking.association import HeadEvidence
from siteguard.diagnostics import StageTimings


class AnnotatedVideo:
    """Sampled CFR output; frame JSONL retains actual timestamps for VFR/jitter auditing."""
    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.head_evidence: dict[int, HeadEvidence] = {}
        self.performance = StageTimings()

    def set_track_evidence(self, evidence: dict[int, HeadEvidence]) -> None:
        self.head_evidence = evidence

    def write(self, image: ImageArray, detections: list[Detection], text: str = "") -> None:
        started = time.perf_counter()
        canvas = image.copy()
        height, width = canvas.shape[:2]
        # mp4v requires even dimensions.
        canvas = cv2.copyMakeBorder(canvas, 0, height % 2, 0, width % 2,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (canvas.shape[1], canvas.shape[0]))
            self.size = (canvas.shape[1], canvas.shape[0])
            if not self.writer.isOpened():
                raise RuntimeError("MP4 encoder could not open output")
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.box)
            if detection.label.startswith("track:"):
                _, identifier, state = detection.label.split(":", 2)
                colors = {"CONFIRMED": (20, 20, 240), "SUSPECT": (0, 190, 255),
                          "COMPLIANT": (30, 200, 30), "CLEARED": (230, 150, 30)}
                color = colors.get(state, (170, 170, 170))
                head = self.head_evidence.get(int(identifier))
                visible = head.state.value if head else "unknown"
                label = f"worker #{identifier} head={visible} event={state}"
                thickness = 4 if state == "CONFIRMED" else 2
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            else:
                color = (0, 150, 255) if detection.label == "barehead" else (40, 210, 40)
                label = f"det:{detection.label} {detection.confidence:.2f}"
                # Corners distinguish raw detections from full tracked-body boxes.
                for x, dx in ((x1, 1), (x2, -1)):
                    for y, dy in ((y1, 1), (y2, -1)):
                        cv2.line(canvas, (x, y), (x+dx*7, y), color, 1)
                        cv2.line(canvas, (x, y), (x, y+dy*7), color, 1)
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
            label_x = min(max(0, x1), max(0, canvas.shape[1]-text_width-2))
            label_y = min(canvas.shape[0]-3, max(32, y1))
            cv2.putText(canvas, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(canvas, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if self.size != (canvas.shape[1], canvas.shape[0]):
            scale = min(self.size[0]/canvas.shape[1], self.size[1]/canvas.shape[0])
            fitted = cv2.resize(canvas, (max(1, round(canvas.shape[1]*scale)),
                                        max(1, round(canvas.shape[0]*scale))))
            canvas = cv2.copyMakeBorder(fitted, 0, self.size[1]-fitted.shape[0],
                0, self.size[0]-fitted.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.performance.add("annotation_render", time.perf_counter()-started)
        with self.performance.measure("video_encode"):
            self.writer.write(canvas)

    def close(self) -> None:
        if self.writer:
            self.writer.release()
