from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from siteguard.config.settings import RuntimeConfig


@pytest.fixture
def stub_yolo(monkeypatch: pytest.MonkeyPatch):
    # Only unit-tests checkpoint validation/failure paths. No simulated runtime detections.
    pytest.importorskip("ultralytics")
    class Model:
        task = "detect"
        names = {0: "person", 1: "helmet", 2: "barehead"}
        predictor = SimpleNamespace(device="cpu", model=SimpleNamespace(fp16=False))
        fail = False

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            if self.fail:
                raise RuntimeError("test predictor failure")
            return [SimpleNamespace(boxes=None)]
    monkeypatch.setattr("ultralytics.YOLO", Model)
    return Model


def test_checkpoint_and_readiness_fail_closed(tmp_path: Path, stub_yolo) -> None:
    from siteguard.inference.yolo import YoloDetector
    weights = tmp_path/"test.pt"
    weights.write_bytes(b"synthetic-test-checkpoint-not-real-weights")
    with pytest.raises(ValueError, match="SHA256"):
        YoloDetector(RuntimeConfig(model=str(weights), device="cpu", model_sha256="0"*64))
    unsupported = tmp_path/"unsupported.bin"
    unsupported.touch()
    with pytest.raises(ValueError, match=".pt"):
        YoloDetector(RuntimeConfig(model=str(unsupported), device="cpu"))
    detector = YoloDetector(RuntimeConfig(model=str(weights), device="cpu"))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector.detect(image) == [] and detector.ready
    detector.model.fail = True
    with pytest.raises(RuntimeError, match="predictor failure"):
        detector.detect(image)
    assert not detector.ready
    detector.model.fail = False
    detector.model.predictor.device = "cuda:0"
    with pytest.raises(RuntimeError, match="device mismatch"):
        detector.detect(image)
    assert not detector.ready


def test_unreadable_checkpoint_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ultralytics")
    from siteguard.inference.yolo import YoloDetector
    def broken(*args, **kwargs):
        raise EOFError("unit test")
    monkeypatch.setattr("ultralytics.YOLO", broken)
    weights = tmp_path/"bad.pt"
    weights.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Unable to load"):
        YoloDetector(RuntimeConfig(model=str(weights), device="cpu"))
