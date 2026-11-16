"""Genuine fixed-checkpoint inference ablation; originals and labels stay untouched."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from siteguard.evaluation.objects import frame_diagnostics, validation_content_fingerprint
from siteguard.inference.device import select_device
from siteguard.inference.runtime import require_ultralytics
from siteguard.training.datasets import class_names, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Ablation output must be new")
    require_ultralytics()
    device, _ = select_device(args.device, False)
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    model = YOLO(str(args.model))
    data = yaml.safe_load(args.data.read_text())
    if model.names != class_names(data):
        raise ValueError("Checkpoint/data class mapping mismatch")
    report = {"scope": "paired per-image development object diagnostic, not video or event accuracy",
        "validation_content_sha256": validation_content_fingerprint(args.data),
        "model_sha256": sha256(args.model), "data_yaml_sha256": sha256(args.data),
        "imgsz": 640, "device": device, "half": False, "batch": 1,
        "confidence": .001, "diagnostic_confidence": .25, "diagnostic_iou": .5,
        "runs": []}
    for limit, name, interpolation in ((None, "native", None),
            (1280, "width1280-linear", cv2.INTER_LINEAR),
            (1280, "width1280-area", cv2.INTER_AREA)):
        folder = args.output/name
        labels = folder/"labels"
        labels.mkdir(parents=True)
        changed = 0
        started = time.perf_counter()
        for image_path in sorted((Path(data["path"])/data["val"]).glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Validation image decode failed")
            h, w = image.shape[:2]
            if limit and w > limit:
                image = cv2.resize(image, (limit, max(2, round(h*limit/w))), interpolation=interpolation)
                changed += 1
            precision = {"quantize": 32} if "quantize" in DEFAULT_CFG_DICT else {"half": False}
            result = model.predict(image, device=device, imgsz=640, conf=.001,
                iou=.7, max_det=300, verbose=False, **precision)[0]
            expected = "cpu" if device == "cpu" else "cuda:"+device
            if str(model.predictor.device) != expected or model.predictor.model.fp16:
                raise RuntimeError("Ablation device or precision mismatch")
            result.save_txt(str(labels/f"{image_path.stem}.txt"), save_conf=True)
        elapsed = time.perf_counter()-started
        diagnostic = frame_diagnostics(args.data, labels, folder/"errors", max_examples=8)
        matrix = np.asarray(diagnostic["confusion_rows_gt_columns_prediction"])
        counts = {}
        for name in ("helmet", "no_helmet"):
            cls = diagnostic["classes"].index(name)
            counts[name] = {"tp": int(matrix[cls, cls]),
                "fn": int(matrix[cls].sum()-matrix[cls, cls]),
                "fp": int(matrix[:, cls].sum()-matrix[cls, cls])}
        report["runs"].append({"max_width": limit, "interpolation": interpolation, "resized_images": changed,
            "wall_seconds_including_image_io": elapsed, "fixed_counts": counts})
    if validation_content_fingerprint(args.data) != report["validation_content_sha256"]:
        raise ValueError("Validation bytes changed during preprocessing comparison")
    write_json(args.output/"comparison.json", report)


if __name__ == "__main__":
    main()
