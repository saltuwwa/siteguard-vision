from pathlib import Path

import cv2
import numpy as np
import pytest

from siteguard.config.settings import RuntimeConfig
from siteguard.ingestion.video import MediaError, TimestampSampler, VideoReader


def create_video(path: Path, fps: int = 25, frames: int = 50) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    for i in range(frames):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (30+i % 10, 10), (80+i % 10, 110), (100, 100, 100), -1)
        writer.write(image)
    writer.release()


@pytest.mark.parametrize("input_fps", [15, 25, 30, 60])
def test_sampling(input_fps: int) -> None:
    sampler = TimestampSampler(5)
    assert sum(sampler.accept(i/input_fps) for i in range(input_fps*2)) == 10


def test_eof_and_no_queue_drops(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    create_video(path)
    reader = VideoReader(str(path), "test", RuntimeConfig(inference_fps=5, queue_size=1))
    frames = list(reader)
    assert len(frames) == 10
    assert reader.metrics.decoded == 50
    assert reader.metrics.dropped == 0
    assert reader.done.is_set()
    assert reader.thread is not None and not reader.thread.is_alive()
    assert all(b.source_timestamp > a.source_timestamp for a, b in zip(frames, frames[1:]))


def test_corrupt_media_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"corrupt")
    with pytest.raises(MediaError):
        list(VideoReader(str(path), "test", RuntimeConfig()))
