"""Compare saved real evaluations; reject differing validation protocols."""
import argparse
import json
from pathlib import Path

from PIL import Image
import yaml

from siteguard.training.datasets import write_json
from siteguard.evaluation.objects import validation_content_fingerprint


def head_size_recall(analysis: Path, data_path: Path) -> dict:
    data = yaml.safe_load(data_path.read_text())
    root = Path(data["path"])
    diagnostic = json.loads((analysis/"errors/diagnostics.json").read_text())
    cls = diagnostic["classes"].index("no_helmet")
    errors = {Path(r["image"]).stem: {tuple(e["gt"]) for e in r["errors"] if "gt" in e}
              for r in diagnostic["examples"]}
    bins = {name: {"ground_truth": 0, "correct": 0, "missed_or_wrong": 0}
            for name in ("under8", "8to16", "16to32", "32plus")}
    for image in (root/data["val"]).glob("*.jpg"):
        with Image.open(image) as im:
            width, height = im.size
        for line in (root/"labels/val"/f"{image.stem}.txt").read_text().splitlines():
            row = tuple(map(float, line.split()))
            if int(row[0]) != cls:
                continue
            short = min(row[3]*width, row[4]*height)*640/max(width, height)
            group = "under8" if short < 8 else "8to16" if short < 16 else "16to32" if short < 32 else "32plus"
            bins[group]["ground_truth"] += 1
            missed = row in errors.get(image.stem, set())
            bins[group]["missed_or_wrong" if missed else "correct"] += 1
    return {k: {**v, "recall": v["correct"]/v["ground_truth"] if v["ground_truth"] else None}
            for k, v in bins.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    # Historical aliases stay compatible with the original smoke/baseline runs.
    parser.add_argument("--reference", "--smoke", dest="smoke", type=Path, required=True)
    parser.add_argument("--candidate", "--baseline", dest="baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-resolution-change", action="store_true")
    parser.add_argument("--require-same-checkpoint", action="store_true",
                        help="For inference ablations, reject different model weights")
    args = parser.parse_args()
    a, b = [json.loads((p/"metrics.json").read_text()) for p in (args.smoke, args.baseline)]
    if args.require_same_checkpoint and a["checkpoint_sha256"] != b["checkpoint_sha256"]:
        raise ValueError("Inference ablation requires identical checkpoint hashes")
    content_a, content_b = a.get("validation_content_sha256"), b.get("validation_content_sha256")
    if content_a or content_b or args.allow_resolution_change:
        if not content_a or content_a != content_b or content_a != validation_content_fingerprint(args.data):
            raise ValueError("Evaluation content hashes differ/missing; re-evaluate both checkpoints")
    for key in ("protocol", "data_yaml_sha256", "ultralytics_version", "confusion_classes"):
        if key == "data_yaml_sha256" and content_a and content_b:
            continue  # Absolute root changes are safe only with equal content/class hashes.
        left, right = a[key], b[key]
        if key == "protocol" and args.allow_resolution_change:
            left = {k: v for k, v in left.items() if k != "imgsz"}
            right = {k: v for k, v in right.items() if k != "imgsz"}
        if left != right:
            raise ValueError(f"Evaluation settings differ: {key}")
    # head_size_recall projects SOURCE boxes to640 independently of the model's
    # inference size. A960->1280 comparison uses the same source bins as640->960.
    comparison = {"scope": "same official development validation; not independent test",
        "reference_protocol": a["protocol"], "candidate_protocol": b["protocol"],
        "size_bin_basis": "same source boxes projected to long-side640 for both models",
        "validation_content_sha256": content_a,
        "same_checkpoint": a["checkpoint_sha256"] == b["checkpoint_sha256"],
        "interpretation": ("same-checkpoint inference resolution ablation" if
            a["checkpoint_sha256"] == b["checkpoint_sha256"] else
            "different checkpoints; training confounders require separate disclosure"),
        "content_integrity": "verified" if content_a else "not recorded in legacy evaluations",
        "protocol": b["protocol"], "smoke_checkpoint_sha256": a["checkpoint_sha256"],
        "baseline_checkpoint_sha256": b["checkpoint_sha256"],
        "results": {k: {"smoke": a["results"][k], "baseline": b["results"][k],
                        "delta": b["results"][k]-a["results"][k]} for k in a["results"]},
        "classwise": [], "fixed_threshold_counts": {}, "wrong_no_helmet_as_helmet": {},
        "no_helmet_size_recall": {}}
    for ca, cb in zip(a["classwise"], b["classwise"], strict=True):
        if ca["class"] != cb["class"]:
            raise ValueError("Class ordering differs")
        comparison["classwise"].append({"class": ca["class"], **{k: {
            "smoke": ca[k], "baseline": cb[k], "delta": cb[k]-ca[k]}
            for k in ("precision", "recall", "ap50", "ap50_95")}})
    for name, metrics, folder in (("smoke", a, args.smoke), ("baseline", b, args.baseline)):
        comparison["fixed_threshold_counts"][name] = metrics["focus_fixed_counts"]
        comparison["wrong_no_helmet_as_helmet"][name] = metrics["no_helmet_to_helmet"]
        comparison["no_helmet_size_recall"][name] = head_size_recall(folder, args.data)
    write_json(args.output, comparison)
    print(args.output)


if __name__ == "__main__":
    main()
