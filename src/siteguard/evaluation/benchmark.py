import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from siteguard.config.settings import RuntimeConfig
from siteguard.inference.device import doctor
from siteguard.inference.yolo import YoloDetector
from siteguard.ingestion.video import VideoReader
from siteguard.training.datasets import write_json


def gpu_sample() -> list[dict[str, str]] | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
        return [dict(zip(("index", "memory_used_mib", "utilization_percent"),
                         [p.strip() for p in line.split(",")], strict=True))
                for line in result.stdout.strip().splitlines()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def benchmark(source: Path, model: str, output: Path, config: RuntimeConfig,
               max_frames: int = 100, include_fp16: bool = True) -> dict[str, Any]:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    hardware = doctor()
    precisions = [False, True] if hardware["cuda_ok"] and include_fp16 and config.device != "cpu" else [False]
    runs = []
    for half in precisions:
        settings = config.model_copy(update={"model": model, "half": half})
        detector = YoloDetector(settings)
        reader = VideoReader(str(source), "benchmark", settings)
        times = []
        samples = []
        try:
            for frame in reader:
                if not times:
                    for _ in range(3):
                        detector.detect(frame.image)
                started = time.perf_counter()
                detector.detect(frame.image)
                times.append(time.perf_counter()-started)
                if len(times) % 10 == 0:
                    samples.append(gpu_sample())
                if len(times) >= max_frames:
                    break
        finally:
            reader.close()
        if not times:
            raise ValueError("Benchmark source produced no frames")
        runs.append({"precision": "fp16" if half else "fp32", "device": detector.device,
            "model_version": detector.version, "frames": len(times), "warmup_iterations": 3,
            "inference_fps": len(times)/sum(times), "inference_ms_p50": float(np.percentile(times, 50)*1000),
            "inference_ms_p95": float(np.percentile(times, 95)*1000), "gpu_samples": samples,
            "ingestion": vars(reader.metrics)})
    report = {"schema_version": 1, "hardware": hardware, "runs": runs,
        "scope": "Detector-only latency; end-to-end metrics are in analyze summary.json",
        "source": source.name, "accuracy_metrics": None,
        "fp16_comparison": "measured" if len(runs) == 2 else "unavailable or disabled"}
    write_json(output, report)
    return report
