import os
from pathlib import Path

import numpy as np
import pytest

from siteguard.config.settings import RuntimeConfig
from siteguard.inference.base import Detection
from siteguard.inference.device import doctor
from siteguard.tracking.bytetrack import ByteTracker


def test_real_bytetrack_adapter_if_installed() -> None:
    pytest.importorskip("ultralytics")
    pytest.importorskip("lap")
    tracker = ByteTracker(5)
    image = np.zeros((300, 200, 3), dtype=np.uint8)
    first_id = None
    for i in range(5):
        people = tracker.update([Detection((20+i, 10, 120+i, 290), .95, "person")], image)
        assert len(people) == 1
        first_id = first_id or people[0].track_id
        assert first_id == people[0].track_id
    tracker.update([], image)
    tracker.reset()
    assert len(tracker.update([Detection((20, 10, 120, 290), .95, "person")], image)) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not os.getenv("SITEGUARD_GPU_MODEL"), reason="Set SITEGUARD_GPU_MODEL to real PPE weights")
def test_genuine_gpu_smoke() -> None:
    from siteguard.inference.yolo import YoloDetector
    assert doctor()["cuda_ok"], "GPU requested but architecture/tensor tests failed"
    model = Path(os.environ["SITEGUARD_GPU_MODEL"])
    assert model.is_file()
    settings = RuntimeConfig(model=str(model), device="0")
    if os.getenv("SITEGUARD_GPU_MAPPING"):
        import json
        settings.model_mapping = json.loads(Path(os.environ["SITEGUARD_GPU_MAPPING"]).read_text())
    detector = YoloDetector(settings)
    assert not detector.ready
    assert isinstance(detector.detect(np.zeros((640, 640, 3), dtype=np.uint8)), list)
    assert detector.ready
