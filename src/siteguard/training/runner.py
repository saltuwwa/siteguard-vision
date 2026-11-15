from __future__ import annotations

import importlib.metadata
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from siteguard.config.settings import StrictModel, load_config
from siteguard.inference.device import doctor, select_device
from siteguard.inference.runtime import require_ultralytics
from siteguard.training.datasets import CANONICAL, class_names, sha256, write_json


class TrainConfig(StrictModel):
    stage: Literal["A", "B", "comparison", "smoke"] = "A"
    model: str = "yolo26s.pt"
    data: Path = Path("data/built/baseline-a-v1/data.yaml")
    data_schema: Literal["canonical", "source"] = "canonical"
    validation_report: Path | None = None
    project: Path = Path("runs/train")
    name: str = "stage-a-seed42"
    epochs: int = Field(100, ge=1)
    batch: int = Field(8, ge=1)
    imgsz: int = Field(640, ge=32)
    device: str = "auto"
    workers: int = Field(4, ge=0)
    seed: int = 42
    amp: bool = False
    deterministic: bool = True
    patience: int = Field(30, ge=0)
    resume: Path | None = None
    tensorboard: bool = False
    reference_experiment: Path | None = None
    augmentation: dict[str, float] = Field(default_factory=lambda: {
        "hsv_h": .015, "hsv_s": .7, "hsv_v": .4, "degrees": 0., "translate": .1,
        "scale": .5, "fliplr": .5, "flipud": 0., "mosaic": 1., "mixup": 0.})


def validate_built_data(data_path: Path, split: str = "val") -> dict[str, Any]:
    data = yaml.safe_load(data_path.read_text())
    if data.get("names") != CANONICAL or "download" in data:
        raise ValueError("Use siteguard dataset build; canonical names and no download hooks required")
    root = Path(data["path"])
    report_path = root / "build-report.json"
    report = json.loads(report_path.read_text())
    if report["errors"] or report["leakage"] or not all(s["reviewed"] for s in report["sources"]):
        raise ValueError("Built data did not pass integrity/review gates")
    if split not in data or not (root / data[split]).is_dir():
        raise ValueError(f"Missing {split} images")
    if split == "test" and not report["independence_proven"]:
        raise ValueError("Cannot claim independent test without source provenance")
    inventory_path = root / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    for relative, checksum in inventory.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != checksum:
            raise ValueError(f"Built dataset changed since validation: {relative}")
    return report


def validate_source_data(data_path: Path, report_path: Path, smoke_only: bool) -> dict[str, Any]:
    """Bind native labels to the actual audited bytes, without canonical projection.

    Agent review never claims independent human annotation review. A baseline
    can proceed under an explicit user authorization recorded in the review.
    """
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if "download" in data or not Path(data["path"]).is_absolute():
        raise ValueError("Native source training needs a local resolved root and no download hooks")
    root = Path(data["path"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["errors"] or report["leakage"] or len(report["sources"]) != 1:
        raise ValueError("Native dataset failed validation/leakage checks")
    source = report["sources"][0]
    if source.get("label_schema") != "source" or not source["reviewed"]:
        raise ValueError("Native label preservation and visual review are required")
    if {int(k): v for k, v in source["classes"].items()} != class_names(data):
        raise ValueError("Training class names differ from the audited source")
    review = source["review"]
    if not smoke_only and (review.get("training_scope") != "baseline" or not (
            review.get("kind") == "human" or review.get("authorized_by_user") is True)):
        raise ValueError("Source spot-check authorizes one-epoch smoke only; baseline needs explicit authorization")
    manifest = source["manifest"]
    if manifest.get("license") not in {"CC-BY-4.0", "CC0-1.0", "Public-Domain"}:
        raise ValueError("Source license not approved")
    if not manifest.get("archive_sha256") and not manifest.get("content_fingerprint"):
        raise ValueError("Source requires an archive checksum or explicit extracted-content fingerprint")
    for split in ("train", "val"):
        if split not in data or not (root/data[split]).resolve().is_relative_to(root):
            raise ValueError("Invalid local split")
        expected_images = {Path(r["path"]).resolve() for r in source["records"] if r["split"] == split}
        actual_images = {p.resolve() for p in (root/data[split]).rglob("*") if p.is_file()
                         and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
        if not expected_images or actual_images != expected_images:
            raise ValueError("Training split differs from audited image inventory")
    for record in source["records"]:
        for key, hash_key in (("path", "sha256"), ("label", "label_sha256")):
            path = Path(record[key]).resolve()
            if not path.is_relative_to(root) or sha256(path) != record[hash_key]:
                raise ValueError(f"Audited source changed: {record['relative_path']}")
    report["evaluation_fingerprint"] = hashlib.sha256(json.dumps(
        [(r["relative_path"], r["sha256"], r["label_sha256"]) for r in source["records"]
         if r["split"] == "val"], sort_keys=True).encode()).hexdigest()
    return report


def run_training(config_path: Path) -> Path:
    cfg = load_config(config_path, TrainConfig)
    if cfg.stage == "smoke" and cfg.epochs != 1:
        raise ValueError("Smoke stage is restricted to exactly one epoch")
    if cfg.data_schema == "source":
        if cfg.validation_report is None:
            raise ValueError("Native source training requires validation_report")
        report = validate_source_data(cfg.data, cfg.validation_report, cfg.stage == "smoke")
    else:
        report = validate_built_data(cfg.data)
    source_names = {s["name"] for s in report["sources"]}
    if cfg.stage == "A" and (source_names != {"construction-ppe"} or cfg.model != "yolo26s.pt"):
        raise ValueError("Stage A requires Construction-PPE only and yolo26s.pt")
    if cfg.stage == "B" and (source_names != {"construction-ppe", "hard-hat-workers"}
                              or cfg.model != "yolo26s.pt"):
        raise ValueError("Stage B requires verified Construction-PPE + Hard Hat Workers and yolo26s.pt")
    if cfg.model not in {"yolo26s.pt", "yolo26m.pt"} and not Path(cfg.model).is_file():
        raise ValueError("Expected official yolo26s.pt/yolo26m.pt or a local checkpoint")
    require_ultralytics()
    if cfg.stage in {"B", "comparison"}:
        if cfg.reference_experiment is None:
            raise ValueError("Stage B/comparison requires reference_experiment from completed baseline A/B")
        reference = json.loads(cfg.reference_experiment.read_text())
        if reference["status"] != "completed" or reference["evaluation_fingerprint"] != report["evaluation_fingerprint"]:
            raise ValueError("Reference run incomplete or evaluation sets differ")
        for key in ("epochs", "batch", "imgsz", "seed", "amp", "augmentation"):
            if reference["config"][key] != cfg.model_dump(mode="json")[key]:
                raise ValueError(f"Controlled comparison setting differs: {key}")
    from ultralytics import YOLO, settings

    device, amp = select_device(cfg.device, cfg.amp)
    if cfg.tensorboard:
        importlib.metadata.version("tensorboard")
    # Ultralytics settings are global: training runs are separate CLI processes.
    options = {"tensorboard": cfg.tensorboard, "mlflow": False, "wandb": False,
               "sync": False, "hub": False, "clearml": False, "comet": False, "neptune": False}
    settings.update({k: v for k, v in options.items() if k in settings})
    experiment_dir = cfg.project / cfg.name
    if experiment_dir.exists() and not cfg.resume:
        raise ValueError("Experiment exists; choose a new run name or --resume checkpoint via config")
    checkpoint = cfg.resume or cfg.model
    if cfg.resume and not cfg.resume.is_file():
        raise FileNotFoundError("Resume checkpoint missing")
    model = YOLO(str(checkpoint))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "config": cfg.model_dump(mode="json"),
        "dataset_report_sha256": sha256(cfg.validation_report if cfg.data_schema == "source"
            else Path(yaml.safe_load(cfg.data.read_text())["path"])/"build-report.json"),
        "data_yaml_sha256": sha256(cfg.data), "hardware": doctor(), "platform": platform.platform(),
        "initial_checkpoint_sha256": sha256(Path(model.ckpt_path)),
        "evaluation_fingerprint": report["evaluation_fingerprint"], "status": "running",
        "ultralytics_version": importlib.metadata.version("ultralytics"),
        "class_names": class_names(yaml.safe_load(cfg.data.read_text())),
        "dataset_independence_proven": report["independence_proven"],
        "review_scope": report["sources"][0].get("review", {})}
    write_json(experiment_dir/"experiment.json", metadata)
    import torch
    epoch_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(int(device))

    def record_epoch(trainer: Any) -> None:
        metrics = trainer.validator.metrics
        names = trainer.data["names"]
        row = {"epoch": int(trainer.epoch)+1, "elapsed_seconds": time.perf_counter()-started,
               "phase": "epoch" if trainer.validator.training else "final_best_validation",
               "classwise": [{"class_id": int(c), "class": names[int(c)],
                   "precision": float(p), "recall": float(r), "ap50": float(a50), "ap50_95": float(ap)}
                   for c, p, r, a50, ap in zip(metrics.box.ap_class_index, metrics.box.p,
                       metrics.box.r, metrics.box.ap50, metrics.box.ap, strict=True)]}
        if device != "cpu":
            row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(int(device))
            row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(int(device))
        if not trainer.validator.training:
            # Ultralytics invokes this callback again after final best.pt validation
            # and temporarily increments epoch. Preserve it separately, never as
            # an extra training epoch.
            row["epochs_completed"] = int(trainer.epoch)
            row.pop("epoch")
            write_json(experiment_dir/"final-best-validation.json", row)
            return
        epoch_records.append(row)
        write_json(experiment_dir/"per-class-epochs.json", epoch_records)

    model.add_callback("on_fit_epoch_end", record_epoch)
    try:
        arguments: dict[str, Any] = {"data": str(cfg.data.resolve()), "epochs": cfg.epochs,
            "batch": cfg.batch, "imgsz": cfg.imgsz, "device": device, "workers": cfg.workers,
            "seed": cfg.seed, "amp": amp, "deterministic": cfg.deterministic, "patience": cfg.patience,
            "project": str(cfg.project.resolve()), "name": cfg.name, "exist_ok": True, "save": True,
            "plots": True, **cfg.augmentation}
        if cfg.resume:
            # Ultralytics restores the checkpoint's training arguments on resume.
            model.train(resume=str(cfg.resume), device=device)
        else:
            model.train(**arguments)
        save_dir = Path(model.trainer.save_dir)
        metadata["resolved_train_args"] = vars(model.trainer.args)
        metadata["weights"] = {name: {"path": str(save_dir/"weights"/f"{name}.pt"),
            "sha256": sha256(save_dir/"weights"/f"{name}.pt")} for name in ("best", "last")}
        metadata["results_csv"] = str(save_dir/"results.csv")
        import csv
        with (save_dir/"results.csv").open(encoding="utf-8", newline="") as metrics_stream:
            metadata["epoch_metrics"] = [{k.strip(): float(v) for k, v in row.items()}
                for row in csv.DictReader(metrics_stream)]
        metadata["actual_training_device"] = str(model.trainer.device)
        metadata["epochs_completed"] = int(model.trainer.epoch)+1
        metadata["stop_reason"] = "early_stopping" if metadata["epochs_completed"] < cfg.epochs else "epoch_limit"
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        raise
    finally:
        metadata["training_seconds"] = time.perf_counter()-started
        if device != "cpu":
            metadata["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(int(device))
            metadata["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(int(device))
        write_json(experiment_dir/"experiment.json", metadata)
    return experiment_dir


def validate_checkpoint(model_path: Path, data_path: Path, output: Path,
                        split: str = "val", device: str = "auto", imgsz: int = 640) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    report = validate_built_data(data_path, split)
    require_ultralytics()
    from ultralytics import YOLO

    selected, _ = select_device(device)
    model = YOLO(str(model_path))
    result = model.val(data=str(data_path.resolve()), split=split, device=selected,
        imgsz=imgsz, plots=True, save_txt=True, save_conf=True,
        project=str(output.parent), name=output.name, exist_ok=False)
    metrics = {"schema_version": 1, "split": split,
        "interpretation": "development validation (not unbiased test)" if split == "val" else "independent test",
        "model_sha256": sha256(model_path), "data_sha256": sha256(data_path),
        "precision": float(result.box.mp), "recall": float(result.box.mr),
        "map50": float(result.box.map50), "map50_95": float(result.box.map),
        "classwise": [{"class": model.names[int(cls)], "precision": float(p), "recall": float(r),
                       "ap50": float(ap50), "ap50_95": float(ap)}
            for cls, p, r, ap50, ap in zip(result.box.ap_class_index, result.box.p, result.box.r,
                result.box.ap50, result.box.ap, strict=True)],
        "confusion_matrix": result.confusion_matrix.matrix.tolist(),
        "dataset_independence_proven": report["independence_proven"],
        "artifacts": str(result.save_dir), "hardware": doctor()}
    write_json(Path(result.save_dir)/"object-metrics.json", metrics)
    return metrics


def export_checkpoint(model_path: Path, format: str, device: str, half: bool, imgsz: int) -> str:
    require_ultralytics()
    from ultralytics import YOLO

    selected, fp16 = select_device(device, half)
    if format not in {"onnx", "engine"}:
        raise ValueError("Only ONNX or explicitly verified TensorRT engine exports supported")
    if format == "engine":
        if selected == "cpu":
            raise ValueError("TensorRT requires validated NVIDIA CUDA")
        import tensorrt as trt
        if tuple(map(int, trt.__version__.split(".")[:2])) < (10, 8):
            raise ValueError("Blackwell requires TensorRT >=10.8; verify installed GPU/runtime support")
    return str(YOLO(str(model_path)).export(format=format, device=selected, half=fp16, imgsz=imgsz))
