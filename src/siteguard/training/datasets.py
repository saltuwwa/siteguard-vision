"""Explicit dataset semantics, immutable raw imports, and leakage auditing.

No dataset YAML download hooks are executed. Relative paths resolve from repository cwd.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree as ET

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.fft import dctn
from pydantic import Field

from siteguard.config.settings import StrictModel, load_config

CANONICAL = ["person", "helmet", "barehead"]
# Five-decimal YOLO exports can move a reconstructed edge by <= 7.5e-6.
# Components still must be normalized; tolerance applies only to box corners.
CORNER_TOLERANCE = 1e-5
CONSTRUCTION_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
)
SOURCES = {
    "construction-ppe": "https://docs.ultralytics.com/datasets/detect/construction-ppe/",
    "hard-hat-workers": "https://universe.roboflow.com/joseph-nelson/hard-hat-workers",
    "shwd": "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class DatasetSource(StrictModel):
    name: str
    root: Path
    yaml_file: Path
    mapping: dict[str, str | None]
    enabled: bool = True
    review_file: Path | None = None
    provenance_file: Path | None = None
    import_manifest: Path | None = None
    label_schema: Literal["canonical", "source"] = "canonical"
    # An independent source must never contribute training data.
    role: str = "development"
    include_splits: list[str] = Field(default_factory=lambda: ["train", "val", "test"])


class DatasetConfig(StrictModel):
    sources: list[DatasetSource]
    output: Path = Path("data/built/baseline-a")
    report: Path = Path("runs/dataset-report.json")
    near_duplicate_distance: int = Field(4, ge=0, le=16)
    commercial: bool = True
    # Explicit acknowledgement; reports still state that independence is unproven.
    allow_unknown_provenance: bool = False


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        if sum(item.file_size for item in zipped.infolist()) > 50 * 1024**3:
            raise ValueError("Archive exceeds 50 GiB extraction limit")
        for item in zipped.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
        zipped.extractall(destination)


def download_dataset(
    name: str, destination: Path, archive: Path | None = None, version: str | None = None,
    license_id: str | None = None, accept_license: bool = False,
) -> Path:
    if name not in SOURCES:
        raise ValueError("Unsupported source. SH17 is deliberately excluded.")
    target = destination / name
    if (target / "manifest.json").exists():
        raise ValueError("Raw import exists; use a new --destination for a different version")
    if name != "construction-ppe" and not (version and license_id and accept_license):
        raise ValueError("Manual/versioned source requires --version, --license-id and --accept-license")
    if name == "shwd" and archive is None:
        raise ValueError("Obtain original VOC archive manually and supply --archive")
    if name == "hard-hat-workers" and archive is None:
        # Authenticated access is only attempted after the user's explicit license action.
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise ValueError("Export a versioned YOLO archive manually, or set ROBOFLOW_API_KEY")
        if not version or not version.isdigit():
            raise ValueError("Roboflow version must be an integer")
        url = f"https://api.roboflow.com/joseph-nelson/hard-hat-workers/{version}/yolov8?api_key={key}"
        try:
            for attempt in range(12):
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                if payload.get("export", {}).get("link"):
                    url = payload["export"]["link"]
                    break
                if attempt == 11:
                    raise RuntimeError("Export not ready")
                time.sleep(5)
        except Exception:
            raise RuntimeError("Roboflow export request failed; credentials/URL suppressed") from None
    else:
        url = CONSTRUCTION_URL
    target.mkdir(parents=True, exist_ok=True)
    saved = target / "raw.zip"
    if archive:
        shutil.copyfile(archive, saved)
    else:
        try:
            with urllib.request.urlopen(url, timeout=120) as response, saved.open("wb") as out:
                shutil.copyfileobj(response, out)
        except Exception:
            raise RuntimeError("Dataset transfer failed; retry or use --archive (URL suppressed)") from None
    safe_extract(saved, target / "extracted")
    manifest = {
        "name": name, "source_url": SOURCES[name],
        "download_url": CONSTRUCTION_URL if name == "construction-ppe" else "manual/versioned export",
        "license": license_id or "CC-BY-4.0", "version": version or f"sha256:{sha256(saved)}",
        "downloaded_at": datetime.now(UTC).isoformat(), "archive_sha256": sha256(saved),
        "provenance": "mixed real/generated" if name == "construction-ppe" else "unknown",
        "license_accepted": accept_license, "annotation_semantics_verified": False,
    }
    write_json(target / "manifest.json", manifest)
    return target


def phash(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    frequencies = dctn(gray, type=2, norm="ortho")[:8, :8].flatten()
    bits = frequencies > np.median(frequencies[1:])
    return int("".join("1" if bit else "0" for bit in bits), 2)


def class_names(data: dict[str, Any]) -> dict[int, str]:
    names = data["names"]
    result = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if set(result) != set(range(len(result))) or len(set(result.values())) != len(result):
        raise ValueError("Class indices must be contiguous with unique names")
    return result


def source_fingerprint(source: DatasetSource, records: list[dict[str, Any]]) -> str:
    contents = {"mapping": source.mapping, "label_schema": source.label_schema,
                "yaml_sha256": sha256(source.yaml_file),
                "records": [(r["relative_path"], r["sha256"], r["label_sha256"])
                            for r in records]}
    return hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()


def inspect_source(source: DatasetSource, samples_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    rounded_boundary_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"name": source.name, "errors": errors, "warnings": warnings,
                              "records": records, "reviewed": False}
    try:
        data = yaml.safe_load(source.yaml_file.read_text(encoding="utf-8"))
        names = class_names(data)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"Cannot inspect dataset YAML: {exc}")
        return report
    report["classes"] = names
    report["included_splits"] = source.include_splits
    report["mapping"] = source.mapping
    report["label_schema"] = source.label_schema
    if set(source.mapping) != set(names.values()):
        errors.append("Mapping must explicitly cover every class by exact name (null = excluded)")
    if any(v not in CANONICAL and v is not None for v in source.mapping.values()):
        errors.append("Only person/helmet/barehead or null mappings permitted; unknown is not a class")
    manifest_path = source.import_manifest or source.root.parent.parent / "manifest.json"
    # Explicit manifest preferred because archive roots vary by version.
    if source.import_manifest is None:
        for parent in [source.root, *source.root.parents]:
            if (parent / "manifest.json").exists():
                manifest_path = parent / "manifest.json"
                break
    if manifest_path.exists():
        report["manifest"] = json.loads(manifest_path.read_text())
    else:
        errors.append("Missing import manifest with source URL, license, version and archive checksum")
    provenance: dict[str, Any] = {}
    if source.provenance_file and source.provenance_file.exists():
        provenance = json.loads(source.provenance_file.read_text())
    counts: dict[str, Any] = {}
    samples_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        locations = data[split] if isinstance(data[split], list) else [data[split]]
        images: list[Path] = []
        for location in locations:
            # Roboflow exports commonly contain ../train/images despite placing
            # train/images beside data.yaml. Normalize only this known adapter.
            normalized = str(location)
            if source.name == "hard-hat-workers":
                while normalized.startswith("../"):
                    normalized = normalized[3:]
                if normalized != str(location):
                    warnings.append(f"Roboflow export path normalized: {location} -> {normalized}")
            directory = source.root / normalized
            if not directory.resolve().is_relative_to(source.root.resolve()):
                errors.append(f"Split path outside source root: {location}")
                continue
            if not directory.is_dir():
                errors.append(f"Expected image directory: {directory}")
                continue
            images.extend(sorted(p for p in directory.rglob("*")
                                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}))
        by_class: Counter[str] = Counter()
        class_images: Counter[str] = Counter()
        for index, path in enumerate(images):
            relative = path.relative_to(source.root).as_posix()
            parts = list(path.relative_to(source.root).parts)
            if "images" not in parts:
                errors.append(f"No images path component: {relative}")
                continue
            parts[parts.index("images")] = "labels"
            label = (source.root / Path(*parts)).with_suffix(".txt")
            boxes: list[tuple[str, list[float]]] = []
            if not label.exists():
                errors.append(f"Missing annotation: {relative}")
                continue
            try:
                for number, line in enumerate(label.read_text().splitlines(), 1):
                    fields = line.split()
                    if not fields:
                        continue
                    if len(fields) != 5:
                        raise ValueError(f"line {number}: expected YOLO detection class x y w h")
                    cls = int(fields[0])
                    box = list(map(float, fields[1:]))
                    x, y, w, h = box
                    if cls not in names or not all(math.isfinite(v) for v in box):
                        raise ValueError("invalid class or nonfinite coordinate")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                            and -CORNER_TOLERANCE <= x-w/2 <= x+w/2 <= 1+CORNER_TOLERANCE
                            and -CORNER_TOLERANCE <= y-h/2 <= y+h/2 <= 1+CORNER_TOLERANCE):
                        raise ValueError("box outside normalized image bounds")
                    overflow = max(0., -(x-w/2), -(y-h/2), x+w/2-1, y+h/2-1)
                    if overflow > 0:
                        rounded_boundary_rows.append({"image": relative, "line": number,
                                                      "corner_overflow": overflow})
                    boxes.append((names[cls], box))
                with Image.open(path) as im:
                    im.load()
                    perceptual = phash(im)
                    if index < 12:
                        canvas = im.convert("RGB")
                        draw = ImageDraw.Draw(canvas)
                        for cls_name, (x, y, w, h) in boxes:
                            xy = [(x-w/2)*im.width, (y-h/2)*im.height,
                                  (x+w/2)*im.width, (y+h/2)*im.height]
                            draw.rectangle(xy, outline="red", width=2)
                            label_text = cls_name if source.label_schema == "source" else f"{cls_name} -> {source.mapping.get(cls_name)}"
                            draw.text(xy[:2], label_text, fill="red")
                        canvas.thumbnail((1280, 1280))
                        canvas.save(samples_dir / f"{source.name}-{split}-{index:03}.jpg")
                for cls_name in {cls_name for cls_name, _ in boxes}:
                    class_images[cls_name] += 1
                by_class.update(cls_name for cls_name, _ in boxes)
                if relative in seen:
                    errors.append(f"Repeated file across splits: {relative}")
                seen.add(relative)
                records.append({"source": source.name, "split": split,
                    "path": str(path.resolve()), "relative_path": relative,
                    "label": str(label.resolve()), "label_sha256": sha256(label),
                    "sha256": sha256(path), "phash": f"{perceptual:016x}",
                    "provenance": provenance.get(relative, {}), "boxes": boxes})
            except (OSError, ValueError) as exc:
                errors.append(f"Invalid annotation/image {relative}: {exc}")
        counts[split] = {"images": len(images), "objects_per_class": dict(by_class),
                         "images_per_class": dict(class_images)}
    report["counts"] = counts
    report["annotation_diagnostics"] = {
        "missing": sum(e.startswith("Missing annotation") for e in errors),
        "invalid_or_corrupt": sum(e.startswith("Invalid annotation/image") for e in errors)}
    report["rounded_boundary_rows"] = rounded_boundary_rows
    report["corner_tolerance"] = CORNER_TOLERANCE
    if rounded_boundary_rows:
        warnings.append(f"{len(rounded_boundary_rows)} rounded box edges within {CORNER_TOLERANCE}; labels unchanged")
    valid_labels = {str(Path(r["label"]).resolve()) for r in records}
    label_directories = [source.root/"labels"]
    if source.name == "hard-hat-workers":
        label_directories.extend(source.root/s/"labels" for s in ("train", "valid", "test"))
    for label_path in (p for directory in label_directories for p in directory.rglob("*.txt")):
        if str(label_path.resolve()) not in valid_labels:
            warnings.append(f"Orphan or invalid label: {label_path.relative_to(source.root)}")
    if source.name == "construction-ppe":
        actual = {split: value["images"] for split, value in counts.items()}
        report["matches_documented_2026_11_release"] = (
            len(names) == 12 and actual == {"train": 800, "val": 200})
        if not report["matches_documented_2026_11_release"]:
            warnings.append("Archive differs from documented 1,000-image/12-class November 2026 release")
    if not records:
        errors.append("No valid labeled images")
    fingerprint = source_fingerprint(source, records)
    report["fingerprint"] = fingerprint
    if source.review_file and source.review_file.exists():
        review = json.loads(source.review_file.read_text())
        report["review"] = review
        base_review = bool(review.get("reviewer") and review.get("reviewed_at")
            and review.get("fingerprint") == fingerprint and review.get("semantics_verified") is True
            and review.get("sample_paths"))
        report["reviewed"] = base_review and (
            review.get("canonical_annotations_complete") is True if source.label_schema == "canonical"
            else review.get("label_ids_preserved") is True
                 and review.get("training_scope") in {"one_epoch_smoke_only", "baseline"})
    if not report["reviewed"]:
        warnings.append("Annotation sample review required; training not approved")
    if any(not r["provenance"].get("groups") for r in records):
        warnings.append("Original video/session/camera/location groups unavailable; independence unproven")
    return report


def validate_datasets(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path, DatasetConfig)
    sources = [s for s in cfg.sources if s.enabled]
    reports = [inspect_source(s, cfg.report.parent / "annotation-samples") for s in sources]
    all_records = [r for report in reports for r in report["records"]]
    leaks: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    # BK tree, Hamming distance: avoids an all-pairs scan on larger collections.
    tree: dict[str, Any] = {}
    for record in all_records:
        value = int(record["phash"], 16)
        stack = [tree] if tree else []
        while stack:
            node = stack.pop()
            distance = (value ^ node["value"]).bit_count()
            if distance <= cfg.near_duplicate_distance:
                for other in node["records"]:
                    pair = {"a": other["path"], "b": record["path"], "distance": distance,
                            "exact": other["sha256"] == record["sha256"]}
                    duplicates.append(pair)
                    if record["split"] != other["split"]:
                        leaks.append({**pair, "reason": "duplicate across splits"})
            stack.extend(child for key, child in node["children"].items()
                         if distance-cfg.near_duplicate_distance <= key <= distance+cfg.near_duplicate_distance)
        if not tree:
            tree.update(value=value, records=[record], children={})
        else:
            node = tree
            while True:
                distance = (value ^ node["value"]).bit_count()
                if distance == 0:
                    node["records"].append(record)
                    break
                if distance not in node["children"]:
                    node["children"][distance] = {"value": value, "records": [record], "children": {}}
                    break
                node = node["children"][distance]
    groups: dict[str, tuple[str, str]] = {}
    for record in all_records:
        for group in record["provenance"].get("groups", []):
            if group in groups and groups[group][0] != record["split"]:
                leaks.append({"a": groups[group][1], "b": record["path"],
                              "reason": f"shared provenance group: {group}"})
            groups[group] = (record["split"], record["path"])
    result = {"schema_version": 1, "generated_at": datetime.now(UTC).isoformat(),
              "sources": reports, "suspected_duplicates": duplicates, "leakage": leaks,
              "errors": [e for r in reports for e in r["errors"]],
              "independence_proven": bool(all_records) and not leaks and all(
                  r["provenance"].get("groups") for r in all_records)}
    write_json(cfg.report, result)
    return result


def build_dataset(config_path: Path) -> Path:
    cfg = load_config(config_path, DatasetConfig)
    if any(s.enabled and s.label_schema == "source" for s in cfg.sources):
        raise ValueError("Source schema preserves native labels: use training data_schema=source with its validation report")
    report = validate_datasets(config_path)
    if report["errors"] or report["leakage"]:
        raise ValueError("Dataset errors/leakage found; inspect machine-readable report")
    if not report["independence_proven"] and not cfg.allow_unknown_provenance:
        raise ValueError("Independence unproven. Supply provenance or explicitly acknowledge in config")
    if not all(s["reviewed"] for s in report["sources"]):
        raise ValueError("Review labeled samples and record fingerprint before building")
    if cfg.output.exists():
        raise ValueError("Build output exists; use a new versioned output directory")
    concept_counts: Counter[str] = Counter()
    for source, spec in zip(report["sources"], [s for s in cfg.sources if s.enabled], strict=True):
        for record in source["records"]:
            if record["split"] == "train" and "train" in spec.include_splits:
                concept_counts.update(mapped for name, _ in record["boxes"]
                                      if (mapped := spec.mapping.get(name)) is not None)
    if any(not concept_counts[name] for name in CANONICAL):
        raise ValueError("Training build needs positive person, helmet and barehead annotations")
    for source in report["sources"]:
        manifest = source["manifest"]
        if not all(manifest.get(key) for key in
                   ("source_url", "license", "version", "archive_sha256", "downloaded_at")):
            raise ValueError("Incomplete dataset provenance manifest")
        if cfg.commercial and manifest["license"] not in {"CC-BY-4.0", "CC0-1.0", "Public-Domain"}:
            raise ValueError("Commercial build requires verified permitted dataset license")
    counts: Counter[str] = Counter()
    for source, spec in zip(report["sources"], [s for s in cfg.sources if s.enabled], strict=True):
        for record in source["records"]:
            split = record["split"]
            if split not in spec.include_splits:
                continue
            if split == "test" and spec.role != "independent":
                raise ValueError("Test must come from an explicitly independent source")
            if spec.role == "independent" and split != "test":
                raise ValueError("Independent evaluation source may only supply test images")
            image_path = Path(record["path"])
            identifier = hashlib.sha256(f"{spec.name}/{record['relative_path']}".encode()).hexdigest()[:24]
            image_dest = cfg.output / "images" / split / (identifier + image_path.suffix.lower())
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, image_dest)
            label_dest = cfg.output / "labels" / split / f"{identifier}.txt"
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for cls_name, box in record["boxes"]:
                mapped = spec.mapping.get(cls_name)
                if mapped:
                    lines.append(f"{CANONICAL.index(mapped)} " + " ".join(map(str, box)))
            label_dest.write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[split] += 1
    data = {"path": str(cfg.output.resolve()), "names": CANONICAL,
            **{split: f"images/{split}" for split in counts}}
    if not counts["train"] or not counts["val"]:
        raise ValueError("Training build requires nonempty official train and val splits")
    output_yaml = cfg.output / "data.yaml"
    output_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    inventory = {p.relative_to(cfg.output).as_posix(): sha256(p)
                 for directory in ("images", "labels") for p in sorted((cfg.output/directory).rglob("*"))
                 if p.is_file()}
    write_json(cfg.output/"inventory.json", inventory)
    report["evaluation_fingerprint"] = hashlib.sha256(json.dumps({k: v for k, v in inventory.items()
        if k.startswith(("images/val/", "labels/val/", "images/test/", "labels/test/"))},
        sort_keys=True).encode()).hexdigest()
    write_json(cfg.output / "build-report.json", report)
    shutil.copyfile(config_path, cfg.output / "build-config.yaml")
    return output_yaml


def convert_shwd(root: Path, output: Path) -> Path:
    """VOC -> YOLO, preserving SHWD's negative-head `person` semantics.

    Conversion does NOT authorize training: licensing and sample review are still required.
    The raw names remain distinct; the source `person` class is never a body detection.
    """
    if output.exists():
        raise ValueError("Conversion output already exists")
    count = 0
    for split_file in sorted((root / "ImageSets" / "Main").glob("*.txt")):
        if split_file.stem not in {"train", "val", "test"}:
            continue
        for identifier in split_file.read_text().split():
            if Path(identifier).name != identifier:
                raise ValueError("Invalid VOC image ID")
            annotation = ET.parse(root / "Annotations" / f"{identifier}.xml").getroot()
            width = float(annotation.findtext("size/width", "0"))
            height = float(annotation.findtext("size/height", "0"))
            if width <= 0 or height <= 0:
                raise ValueError("Invalid VOC dimensions")
            lines = []
            for obj in annotation.findall("object"):
                name = obj.findtext("name")
                if name not in {"hat", "person"}:
                    raise ValueError(f"Unexpected SHWD class: {name}")
                coords = [float(obj.findtext(f"bndbox/{key}", "nan"))
                          for key in ("xmin", "ymin", "xmax", "ymax")]
                x1, y1, x2, y2 = coords
                if not (1 <= x1 < x2 <= width and 1 <= y1 < y2 <= height):
                    raise ValueError("Invalid VOC box (expected 1-based inclusive coordinates)")
                # VOC inclusive -> zero-based half-open.
                x1 -= 1
                y1 -= 1
                box = [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]
                lines.append(f"{0 if name == 'hat' else 1} " + " ".join(map(str, box)))
            image = root / "JPEGImages" / f"{identifier}.jpg"
            dest = output / "images" / split_file.stem / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, dest)
            label = output / "labels" / split_file.stem / f"{identifier}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + "\n")
            count += 1
    if not count:
        raise ValueError("No VOC split files found; supply reviewed train/val/test ID lists")
    path = output / "data.yaml"
    path.write_text(yaml.safe_dump({"names": ["hat", "negative_head_person"],
        **{s: f"images/{s}" for s in ("train", "val", "test") if (output/"images"/s).exists()}}))
    return path
