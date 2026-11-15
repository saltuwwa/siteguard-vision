import hashlib
import os
from pathlib import Path

from siteguard.config.settings import RuntimeConfig
from siteguard.inference.base import Detection, ImageArray
from siteguard.inference.device import select_device
from siteguard.inference.runtime import require_ultralytics


class YoloDetector:
    def __init__(self, config: RuntimeConfig) -> None:
        os.environ["YOLO_AUTOINSTALL"] = "false"
        installed = require_ultralytics()
        from ultralytics import YOLO
        path = Path(config.model)
        if not path.is_file():
            raise FileNotFoundError("PPE checkpoint missing; train first. COCO weights cannot detect bare heads")
        self.model = YOLO(str(path))
        names = set(self.model.names.values())
        if not set(config.model_mapping).issubset(names):
            raise ValueError(f"Checkpoint classes {sorted(names)} do not match configured source names")
        if set(config.model_mapping.values()) != {"person", "helmet", "barehead"}:
            raise ValueError("Explicit person/helmet/barehead mappings required")
        self.config = config
        self.device, self.half = select_device(config.device, config.half)
        self.ready = False
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.version = f"ultralytics-{installed}:sha256:{digest}"

    def detect(self, image: ImageArray) -> list[Detection]:
        result = self.model.predict(source=image, conf=self.config.confidence,
            imgsz=self.config.imgsz, device=self.device, half=self.half, verbose=False)[0]
        expected_device = "cpu" if self.device == "cpu" else f"cuda:{self.device}"
        actual_device = str(self.model.predictor.device)
        if actual_device != expected_device:
            self.ready = False
            raise RuntimeError(f"Inference device mismatch: requested {expected_device}, got {actual_device}")
        self.ready = True
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist(), strict=True):
            label = self.config.model_mapping.get(result.names[int(cls)])
            if label:
                detections.append(Detection((box[0], box[1], box[2], box[3]), float(confidence), label))
        return detections
