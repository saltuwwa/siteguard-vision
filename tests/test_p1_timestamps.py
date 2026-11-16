from pathlib import Path

import cv2
import numpy as np
import pytest

from siteguard.config.settings import RuntimeConfig
from siteguard.ingestion.video import MediaError, VideoReader


@pytest.mark.parametrize("timestamps,regresses", [([0., .04, .071, .125], False),
                                                 ([0., .1, .05], True)])
def test_vfr_kept_and_regression_not_fabricated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              timestamps: list[float], regresses: bool) -> None:
    class Capture:
        index = 0
        released = False

        def read(self):
            if self.index == len(timestamps):
                return False, None
            self.index += 1
            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_FRAME_COUNT: len(timestamps),
                    cv2.CAP_PROP_POS_MSEC: timestamps[max(0, self.index-1)]*1000}[prop]

        def release(self):
            self.released = True
    source = tmp_path/"unit-fixture.mp4"
    source.touch()
    reader = VideoReader(str(source), "fixture", RuntimeConfig(inference_fps=30, max_width=None))
    capture = Capture()
    monkeypatch.setattr(reader, "_open", lambda: capture)
    if regresses:
        with pytest.raises(MediaError, match="regressed"):
            list(reader)
    else:
        frames = list(reader)
        assert [f.source_timestamp for f in frames] == timestamps
        assert all(f.image.shape[:2] == f.original_shape == (120, 160) for f in frames)
        assert reader.metrics.timestamp_fallbacks == 0
    assert capture.released and reader.thread is not None and not reader.thread.is_alive()
