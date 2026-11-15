from pathlib import Path

import cv2

from siteguard.inference.base import Detection, ImageArray


class AnnotatedVideo:
    """Sampled CFR output; frame JSONL retains actual timestamps for VFR/jitter auditing."""
    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None

    def write(self, image: ImageArray, detections: list[Detection], text: str = "") -> None:
        canvas = image.copy()
        height, width = canvas.shape[:2]
        # mp4v requires even dimensions.
        canvas = cv2.resize(canvas, (width-width % 2, height-height % 2))
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (canvas.shape[1], canvas.shape[0]))
            self.size = (canvas.shape[1], canvas.shape[0])
            if not self.writer.isOpened():
                raise RuntimeError("MP4 encoder could not open output")
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.box)
            color = (0, 0, 255) if detection.label == "barehead" else (0, 200, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{detection.label} {detection.confidence:.2f}", (x1, max(15, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(canvas, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if self.size != (canvas.shape[1], canvas.shape[0]):
            canvas = cv2.resize(canvas, self.size)
        self.writer.write(canvas)

    def close(self) -> None:
        if self.writer:
            self.writer.release()
