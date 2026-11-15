"""Install an intact, compatible cached CUDA wheel pair into the active virtualenv.

Offline recovery only: cached files are checked against their original wheel RECORD.
Repacked wheel hashes are local hashes, not claimed upstream distribution hashes.
This never edits the cache or the global Python installation.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
import sysconfig
import zipfile
from datetime import UTC, datetime
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any


def repack(source: Path, output: Path, name: str, version: str) -> tuple[Path, dict[str, Any]]:
    infos = list(source.glob(f"{name}-*.dist-info"))
    if len(infos) != 1:
        raise ValueError(f"Expected one {name} dist-info in {source}")
    info = infos[0]
    metadata = Parser().parsestr((info/"METADATA").read_text(encoding="utf-8"))
    if metadata["Name"] != name or metadata["Version"] != version:
        raise ValueError(f"Expected {name}=={version}, got {metadata['Name']}=={metadata['Version']}")
    tags = Parser().parsestr((info/"WHEEL").read_text(encoding="utf-8")).get_all("Tag", [])
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    expected_tag = f"{tag}-{tag}-{platform_tag}"
    if expected_tag not in tags:
        raise ValueError(f"Cached wheel tags {tags} do not include {expected_tag}")
    record_bytes = (info/"RECORD").read_bytes()
    entries = list(csv.reader(io.StringIO(record_bytes.decode("utf-8"))))
    wheel = output / f"{name}-{version}-{expected_tag}.whl"
    temporary = wheel.with_suffix(".whl.part")
    files = total = 0
    output.mkdir(parents=True, exist_ok=True)
    root = source.resolve()
    normalized_entries = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zipped:
        for relative, encoded_hash, size in entries:
            relative = relative.replace("\\", "/")
            part = PurePosixPath(relative)
            path = source / relative
            if part.is_absolute() or ".." in part.parts or not path.resolve().is_relative_to(root):
                raise ValueError("Unsafe cached RECORD path")
            if not path.is_file():
                raise ValueError(f"Cached wheel is incomplete: {relative}")
            if size and path.stat().st_size != int(size):
                raise ValueError(f"Cached file size mismatch: {relative}")
            if encoded_hash:
                algorithm, expected = encoded_hash.split("=", 1)
                if algorithm != "sha256":
                    raise ValueError(f"Unsupported RECORD hash: {algorithm}")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, algorithm).digest()
                actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                if actual != expected:
                    raise ValueError(f"Cached file hash mismatch: {relative}")
            elif relative != f"{info.name}/RECORD":
                raise ValueError(f"Unhashed cache entry: {relative}")
            if relative != f"{info.name}/RECORD":
                zipped.write(path, relative)
            normalized_entries.append((relative, encoded_hash, size))
            files += 1
            total += path.stat().st_size
            if files % 500 == 0:
                print(f"Validated/repacked {name}: {files} files, {total} bytes", flush=True)
        normalized_record = io.StringIO(newline="")
        csv.writer(normalized_record, lineterminator="\n").writerows(normalized_entries)
        zipped.writestr(f"{info.name}/RECORD", normalized_record.getvalue())
    temporary.replace(wheel)
    with wheel.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return wheel, {"name": name, "version": version, "tag": expected_tag,
        "validated_files": files, "uncompressed_bytes": total, "cache_source": str(source.resolve()),
        "wheel": str(wheel.resolve()), "local_repacked_wheel_sha256": checksum,
        "integrity_basis": "original per-file wheel RECORD; no upstream archive hash available"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-cache", type=Path, required=True)
    parser.add_argument("--torchvision-cache", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, default=Path(".uv-cache/offline-wheels"))
    parser.add_argument("--report", type=Path, default=Path("runs/p0/cuda-recovery.json"))
    parser.add_argument("--reuse-validated", action="store_true",
                        help="Reuse previously validated repacks only when their full hash still matches")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run with the project virtualenv Python; global installation is forbidden")
    packages = [("torch", "2.8.0+cu128", args.torch_cache),
                ("torchvision", "0.23.0+cu128", args.torchvision_cache)]
    report: dict[str, Any] = {"started_at": datetime.now(UTC).isoformat(),
                             "python": sys.executable, "packages": [], "status": "repacking"}
    previous = json.loads(args.report.read_text()) if args.reuse_validated and args.report.exists() else {}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        wheels = []
        for name, version, source in packages:
            old = next((r for r in previous.get("packages", []) if r["name"] == name
                        and r["version"] == version and r["cache_source"] == str(source.resolve())), None)
            if old is not None and Path(old["wheel"]).is_file():
                wheel = Path(old["wheel"])
                with wheel.open("rb") as stream:
                    actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_hash != old["local_repacked_wheel_sha256"]:
                    raise ValueError("Previously validated local wheel hash changed")
                record = old
                print(f"Reusing hash-verified {wheel.name}", flush=True)
            else:
                wheel, record = repack(source, args.wheel_dir, name, version)
            wheels.append(str(wheel))
            report["packages"].append(record)
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        "--force-reinstall", *wheels], check=True)
        report["installed_versions"] = {name: importlib.metadata.version(name) for name, _, _ in packages}
        report["status"] = "installed_requires_doctor"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
