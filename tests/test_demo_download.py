"""Network/media failure bookkeeping; synthetic bytes are never demo footage."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[1]/"scripts/download_construction_demo.py"
    spec = importlib.util.spec_from_file_location("demo_downloader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_network_failure_has_no_finished_video(tmp_path, monkeypatch, downloader):
    monkeypatch.setattr(downloader.shutil, "which", lambda _: "unit-ffprobe")

    def fail(*args, **kwargs):
        raise urllib.error.URLError("unit DNS failure")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        downloader.download(tmp_path)
    report = json.loads((tmp_path/"provenance.json").read_text())
    assert report["status"] == "failed" and report["ground_truth"] is None
    assert not list(tmp_path.glob("*.mp4"))


def test_existing_source_is_not_overwritten(tmp_path, downloader):
    target = tmp_path/"source.mp4"
    target.write_bytes(b"unit original")
    with pytest.raises(FileExistsError):
        downloader.download(tmp_path)
    assert target.read_bytes() == b"unit original"


@pytest.mark.parametrize("valid", [True, False])
def test_only_probed_media_is_finalized(tmp_path, monkeypatch, downloader, valid):
    # Mock bytes and probe output exercise persistence, not decoder verification.
    payload = b"unit fake media; never used as real footage"
    monkeypatch.setattr(downloader.shutil, "which", lambda _: "unit-ffprobe")
    monkeypatch.setattr(downloader.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(payload))
    media = {"streams": [{"width": 640, "height": 480}], "format": {"duration": "1"}}
    monkeypatch.setattr(downloader.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps(media if valid else {})))
    if valid:
        assert downloader.download(tmp_path).read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="video stream"):
            downloader.download(tmp_path)
        assert not (tmp_path/"source.mp4").exists()
    report = json.loads((tmp_path/"provenance.json").read_text())
    assert report["status"] == ("downloaded" if valid else "failed")
    assert report["visual_review"].startswith("pending")
    assert not (tmp_path/"source.partial.mp4").exists()
