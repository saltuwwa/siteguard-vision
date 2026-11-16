"""Actual forward/backward memory probe; no optimizer steps or saved weights."""
import argparse
from pathlib import Path
import time

from siteguard.config.settings import load_config
from siteguard.inference.device import select_device
from siteguard.inference.runtime import require_ultralytics
from siteguard.training.datasets import write_json
from siteguard.training.runner import TrainConfig, validate_source_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, TrainConfig)
    if args.output.exists():
        raise ValueError("Choose a new probe output")
    if cfg.validation_report is None:
        raise ValueError("Audited source data required")
    validate_source_data(cfg.data, cfg.validation_report, False)
    require_ultralytics()
    device, _ = select_device(cfg.device, False)
    if device == "cpu":
        raise ValueError("Memory probe requires validated CUDA")
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer

    trainer = DetectionTrainer(overrides={"model": cfg.model, "data": str(cfg.data.resolve()),
        "device": device, "imgsz": cfg.imgsz, "batch": cfg.batch, "epochs": cfg.epochs,
        "workers": 0, "amp": False, "seed": cfg.seed, "deterministic": cfg.deterministic,
        "project": str(args.output.parent.resolve()), "name": args.output.stem,
        "plots": False, "save": False, **cfg.augmentation})
    trainer._setup_train()
    torch.cuda.reset_peak_memory_stats(int(device))
    started = time.perf_counter()
    batches = 0
    for batch in trainer.train_loader:
        trainer.model.zero_grad(set_to_none=True)
        batch = trainer.preprocess_batch(batch)
        loss, _ = trainer.model(batch)
        loss.sum().backward()
        torch.cuda.synchronize(int(device))
        batches += 1
        if batches == 3:
            break
    write_json(args.output, {"scope": "three genuine training batches; no optimizer steps",
        "device": device, "batch": cfg.batch, "imgsz": cfg.imgsz, "amp": False,
        "batches": batches, "seconds": time.perf_counter()-started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(int(device)),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(int(device)),
        "free_total_bytes": list(torch.cuda.mem_get_info(int(device))),
        "note": "Full training adds optimizer state; retain headroom. Baseline weights unchanged."})


if __name__ == "__main__":
    main()
