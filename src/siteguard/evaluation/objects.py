"""Auditable FP/FN examples at one chosen IoU/confidence operating point.

mAP remains the official Ultralytics validation metric, not this diagnostic.
"""
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from siteguard.training.datasets import CANONICAL, write_json


def iou(a: list[float], b: list[float]) -> float:
    def corners(box: list[float]) -> tuple[float, float, float, float]:
        x, y, w, h = box
        return x-w/2, y-h/2, x+w/2, y+h/2
    x1, y1, x2, y2 = corners(a)
    u1, v1, u2, v2 = corners(b)
    intersection = max(0., min(x2, u2)-max(x1, u1))*max(0., min(y2, v2)-max(y1, v1))
    union = a[2]*a[3]+b[2]*b[3]-intersection
    return intersection/union if union > 0 else 0.


def frame_diagnostics(data_path: Path, predicted_labels: Path, output: Path,
                      split: str = "val", threshold: float = .25, match_iou: float = .5) -> dict[str, Any]:
    data = yaml.safe_load(data_path.read_text())
    if data["names"] != CANONICAL:
        raise ValueError("Expected canonical built dataset")
    root = Path(data["path"])
    records = []
    confusion = np.zeros((4, 4), dtype=np.int64)  # row GT, col prediction; 3=background
    for path in sorted((root/data[split]).rglob("*")):
        if not path.is_file():
            continue
        label = root/"labels"/split/f"{path.stem}.txt"
        truth = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        predicted_file = predicted_labels/f"{path.stem}.txt"
        prediction = [list(map(float, line.split())) for line in predicted_file.read_text().splitlines()
                      if line.strip()] if predicted_file.exists() else []
        if any(len(p) != 6 for p in prediction):
            raise ValueError("Predictions require YOLO class x y w h confidence")
        prediction = [p for p in prediction if p[5] >= threshold]
        weights = np.zeros((len(truth), len(prediction)))
        for g, gt in enumerate(truth):
            for p, pred in enumerate(prediction):
                overlap = iou(gt[1:5], pred[1:5])
                if overlap >= match_iou:
                    weights[g, p] = 1+overlap
        pairs = []
        if weights.size:
            rows, cols = linear_sum_assignment(-weights)
            pairs = [(int(g), int(p)) for g, p in zip(rows, cols, strict=True) if weights[g, p] > 0]
        gt_matched, pred_matched = {g for g, _ in pairs}, {p for _, p in pairs}
        errors = []
        for g, p in pairs:
            confusion[int(truth[g][0]), int(prediction[p][0])] += 1
            if truth[g][0] != prediction[p][0]:
                errors.append({"kind": "wrong_class", "gt": truth[g], "prediction": prediction[p]})
        for g, gt in enumerate(truth):
            if g not in gt_matched:
                confusion[int(gt[0]), 3] += 1
                errors.append({"kind": "FN", "gt": gt})
        for p, pred in enumerate(prediction):
            if p not in pred_matched:
                confusion[3, int(pred[0])] += 1
                errors.append({"kind": "FP", "prediction": pred})
        if errors:
            output.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as im:
                canvas = im.convert("RGB")
                draw = ImageDraw.Draw(canvas)
                for rows, color in ((truth, "lime"), (prediction, "red")):
                    for row in rows:
                        cls, x, y, w, h = row[:5]
                        box = [(x-w/2)*im.width, (y-h/2)*im.height,
                               (x+w/2)*im.width, (y+h/2)*im.height]
                        draw.rectangle(box, outline=color, width=2)
                        draw.text(box[:2], CANONICAL[int(cls)], fill=color)
                canvas.save(output/f"{path.stem}.jpg")
            records.append({"image": str(path), "errors": errors})
    report = {"confidence": threshold, "iou": match_iou, "classes": [*CANONICAL, "background"],
              "confusion_rows_gt_columns_prediction": confusion.tolist(), "examples": records}
    write_json(output/"diagnostics.json", report)
    return report
