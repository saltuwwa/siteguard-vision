import hashlib
import os
import math
import time
from pathlib import Path

from siteguard.config.settings import RuntimeConfig
from siteguard.inference.base import Detection, ImageArray
from siteguard.inference.device import select_device
from siteguard.inference.runtime import require_ultralytics


class YoloDetector:
    def __init__(self, config: RuntimeConfig) -> None:
        initialized_at = time.perf_counter()
        os.environ["YOLO_AUTOINSTALL"] = "false"
        installed = require_ultralytics()
        from ultralytics import YOLO
        path = Path(config.model)
        self.ready = False
        if not path.is_file():
            raise FileNotFoundError("PPE checkpoint missing; train first. COCO weights cannot detect bare heads")
        if path.suffix.lower() != ".pt":
            raise ValueError("This adapter supports trusted PyTorch .pt detection checkpoints")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if config.model_sha256 and digest != config.model_sha256.lower():
            raise ValueError("Checkpoint SHA256 differs from the configured fingerprint")
        try:
            self.model = YOLO(str(path), task="detect")
        except Exception as exc:
            raise ValueError("Unable to load a supported PPE detection checkpoint") from exc
        if self.model.task != "detect":
            raise ValueError("Object detection checkpoint required")
        names = set(self.model.names.values())
        if not set(config.model_mapping).issubset(names):
            raise ValueError(f"Checkpoint classes {sorted(names)} do not match configured source names")
        if set(config.model_mapping.values()) != {"person", "helmet", "barehead"}:
            raise ValueError("Explicit person/helmet/barehead mappings required")
        self.config = config
        self.device, self.half = select_device(config.device, config.half)
        from ultralytics.cfg import DEFAULT_CFG_DICT
        self.precision_args = ({"quantize": 16 if self.half else 32}
                               if "quantize" in DEFAULT_CFG_DICT else {"half": self.half})
        self.version = f"ultralytics-{installed}:sha256:{digest}"
        self.initialization_seconds = time.perf_counter()-initialized_at

    def detect(self, image: ImageArray) -> list[Detection]:
        # A previous success must not mask a later predictor/device failure.
        self.ready = False
        if image.dtype.name != "uint8" or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("Detector input must be a nonempty uint8 BGR image")
        result = self.model.predict(source=image, conf=self.config.confidence,
            imgsz=self.config.imgsz, device=self.device, **self.precision_args, verbose=False)[0]
        expected_device = "cpu" if self.device == "cpu" else f"cuda:{self.device}"
        actual_device = str(self.model.predictor.device)
        if actual_device != expected_device:
            self.ready = False
            raise RuntimeError(f"Inference device mismatch: requested {expected_device}, got {actual_device}")
        if bool(self.model.predictor.model.fp16) != self.half:
            raise RuntimeError("Inference precision differs from the validated configuration")
        detections: list[Detection] = []
        if result.boxes is None:
            self.ready = True
            return detections
        for box, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist(), strict=True):
            label = self.config.model_mapping.get(result.names[int(cls)])
            if label:
                if (not all(math.isfinite(v) for v in (*box, confidence))
                        or not 0 <= confidence <= 1 or box[2] <= box[0] or box[3] <= box[1]):
                    raise RuntimeError("Detector returned invalid mapped geometry or confidence")
                detections.append(Detection((box[0], box[1], box[2], box[3]), float(confidence), label))
        self.ready = True
        return detections
