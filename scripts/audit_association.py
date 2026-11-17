"""Postprocessing sensitivity using genuine saved predictions, not tracking truth."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment

from siteguard.inference.base import Detection
from siteguard.tracking.association import AssociationDiagnostics, associate_heads
from siteguard.tracking.bytetrack import PersonTrack
from siteguard.training.datasets import class_names, write_json
from siteguard.evaluation.objects import iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-confidence", type=float, default=.6)
    parser.add_argument("--person-confidence", type=float, default=None)
    args = parser.parse_args()
    data = yaml.safe_load(args.data.read_text())
    names, root = class_names(data), Path(data["path"])
    report = {"scope": "fixed saved genuine detector predictions; per-image person candidates, NOT tracking/association accuracy",
              "detector_threshold": .15, "head_confidence": args.head_confidence,
              "head_and_person_confidence": (args.head_confidence if args.person_confidence in
                                              (None, args.head_confidence) else None),
              "person_confidence": (args.head_confidence if args.person_confidence is None
                                    else args.person_confidence),
              "minimum_head_size_basis": "12 pixels in decoded image, NOT model input pixels",
              "counterfactual_limit": "coordinate rescaling only; does not rerun detector after video resize",
              "runs": []}
    for width_limit in (1280, None):
        audit, unknown = AssociationDiagnostics(), Counter()
        totals = Counter()
        bare_quality = Counter()
        for path in sorted((root/data["val"]).glob("*.jpg")):
            with Image.open(path) as im:
                w, h = im.size
            scale = min(1., width_limit/w) if width_limit else 1.
            w, h = round(w*scale), round(h*scale)
            detections = []
            prediction = args.predictions/f"{path.stem}.txt"
            if not prediction.exists():
                continue
            mapping = {"Person": "person", "helmet": "helmet", "no_helmet": "barehead"}
            for line in prediction.read_text().splitlines():
                cls, x, y, bw, bh, conf = map(float, line.split())
                name = mapping.get(names[int(cls)])
                if name is not None and conf >= .15:
                    detections.append(Detection(((x-bw/2)*w, (y-bh/2)*h,
                        (x+bw/2)*w, (y+bh/2)*h), conf, name))
            people = [PersonTrack(i+1, d.box, d.confidence) for i, d in enumerate(detections) if d.label == "person"]
            evidence = associate_heads(people, detections, min_confidence=args.head_confidence,
                image_shape=(h, w), diagnostics=audit,
                min_person_confidence=args.person_confidence)
            bareheads = [d for d in detections if d.label == "barehead"]
            truths = [list(map(float, line.split())) for line in
                      (root/"labels/val"/f"{path.stem}.txt").read_text().splitlines() if line.strip()]
            truths = [r[1:5] for r in truths if names[int(r[0])] == "no_helmet"]
            matrix = np.zeros((len(truths), len(bareheads)))
            for g, truth in enumerate(truths):
                for p, head in enumerate(bareheads):
                    x1, y1, x2, y2 = head.box
                    box = [(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]
                    overlap = iou(truth, box)
                    matrix[g, p] = 1+overlap if overlap >= .5 else 0
            matched = set()
            if matrix.size:
                rows, cols = linear_sum_assignment(-matrix)
                matched = {int(p) for g, p in zip(rows, cols, strict=True) if matrix[g, p] > 0}
            accepted_boxes = {e.box for e in evidence.values() if e.state == "barehead"}
            for index, head in enumerate(bareheads):
                prefix = "annotation_tp" if index in matched else "annotation_fp"
                bare_quality[prefix] += 1
                if head.box in accepted_boxes:
                    bare_quality[prefix+"_accepted_to_some_person"] += 1
                else:
                    bare_quality[prefix+"_not_accepted"] += 1
                if head.confidence < args.head_confidence:
                    bare_quality[prefix+"_below_association_confidence"] += 1
                elif min(head.box[2]-head.box[0], head.box[3]-head.box[1]) < 12:
                    bare_quality[prefix+"_below_size_after_confidence"] += 1
            totals.update(d.label for d in detections)
            unknown.update(e.reason for e in evidence.values() if e.state == "unknown")
        report["runs"].append({"max_width": width_limit, "detections": dict(totals),
            "head_attempts": audit.attempts, "accepted": audit.accepted,
            "accepted_by_class": dict(audit.accepted_by_class),
            "rejected_by_class": {k: dict(v) for k, v in audit.rejected_by_class.items()},
            "barehead_annotation_relative_iou50": dict(bare_quality),
            "rejected_by_reason": dict(audit.rejected), "unknown_workers": dict(unknown)})
    write_json(args.output, report)


if __name__ == "__main__":
    main()
