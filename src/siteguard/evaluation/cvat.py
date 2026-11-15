"""CVAT video XML import; human review is mandatory before event evaluation."""
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from siteguard.training.datasets import write_json


def import_cvat(xml_path: Path, output: Path, camera_id: str, source_id: str,
                fps: float, timestamp_offset: float = 0.0) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("Frame-to-source timestamp conversion requires actual recording FPS")
    root = ET.parse(xml_path).getroot()
    observations = []
    for track in root.findall("track"):
        if track.get("label") != "person":
            raise ValueError("Expected person tracks with helmet_state attribute")
        for box in track.findall("box"):
            state = {a.get("name"): a.text for a in box.findall("attribute")}.get("helmet_state")
            if state not in {"helmet", "barehead", "unknown"}:
                raise ValueError("Every box requires reviewed helmet_state: helmet/barehead/unknown")
            occluded = box.get("occluded") == "1"
            visible = box.get("outside") != "1"
            if occluded or not visible:
                state = "unknown"
            observations.append({"camera_id": camera_id, "source_id": source_id,
                "person_id": str(track.get("id")),
                "timestamp": timestamp_offset+int(box.attrib["frame"])/fps,
                "state": state, "occluded": occluded, "visible": visible,
                "box": [float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]})
    if not observations:
        raise ValueError("No video track boxes found")
    # Intentionally not accepted by VideoGroundTruth until a human supplies coverage,
    # provenance, reviewer and incident intervals. Never pseudo-label incident truth.
    imported = {"schema_version": 1, "label_origin": "cvat_import_requires_review",
                "recordings": [], "incidents": [], "observations": observations,
                "import": {"fps": fps, "timestamp_offset": timestamp_offset,
                           "note": "Review interpolation, timestamps and ground-truth intervals manually"}}
    write_json(output, imported)
    return imported
