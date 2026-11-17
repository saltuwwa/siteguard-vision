"""Fetch one public demo candidate; never labels it verified CCTV or ground truth.

Run from repository root. Source media and provenance stay in gitignored data/.
No authentication, model acquisition, GPU work or system changes are performed.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


SOURCE = "https://www.pexels.com/video/aerial-view-of-construction-crew-at-work-35631532/"
DOWNLOAD = "https://www.pexels.com/download/video/35631532/"
LICENSE = "https://www.pexels.com/license/"


def download(destination: Path) -> Path:
    target = destination / "source.mp4"
    if target.exists():
        raise FileExistsError("Source already exists; preserve it or choose a new destination")
    probe = shutil.which("ffprobe")
    if probe is None:
        raise RuntimeError("FFprobe is required to verify the downloaded media")
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / "source.partial.mp4"
    if partial.exists():
        raise FileExistsError("Partial download exists; inspect it or choose a new destination")
    report = {
        "source_url": SOURCE,
        "download_url": DOWNLOAD,
        "license_url": LICENSE,
        "license": "Pexels License, not CC0",
        "catalog_author": "SAI GON CONG TY CP SAN XUAT - THUONG MAI",
        "catalog_duration_seconds": 37,
        "purpose": "demo/evaluation candidate; excluded from training",
        "ground_truth": None,
        "visual_review": "pending; footage content and suitability unverified",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "started",
    }
    try:
        digest = hashlib.sha256()
        size = 0
        request = urllib.request.Request(DOWNLOAD, headers={"User-Agent": "SiteGuard/0.1"})
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("xb") as stream:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 500 * 1024**2:
                    raise ValueError("Demo transfer exceeds 500 MiB limit")
                stream.write(block)
                digest.update(block)
            stream.flush()
            os.fsync(stream.fileno())
        result = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
             "-of", "json", str(partial)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        media = json.loads(result.stdout)
        streams = media.get("streams", [])
        if not streams or streams[0].get("width", 0) < 32 or streams[0].get("height", 0) < 32:
            raise ValueError("Downloaded file has no usable video stream")
        if float(media.get("format", {}).get("duration", 0)) <= 0:
            raise ValueError("Downloaded file has no positive duration")
        partial.replace(target)
        report.update(status="downloaded", sha256=digest.hexdigest(), bytes=size, media=media)
        return target
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        partial.unlink(missing_ok=True)
        raise
    finally:
        (destination / "provenance.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path("data/demo/pexels-35631532"))
    args = parser.parse_args()
    print(download(args.destination))


if __name__ == "__main__":
    main()
