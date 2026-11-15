import hashlib
import os
from pathlib import Path

import cv2

from siteguard.events.schema import ViolationEvent
from siteguard.inference.base import ImageArray


def save_evidence(root: Path, event: ViolationEvent, image: ImageArray) -> str:
    camera = hashlib.sha256(event.cameraid.encode()).hexdigest()[:16]
    session = hashlib.sha256(event.streamsessionid.encode()).hexdigest()[:16]
    # Event IDs are generated UUIDs, but hash all external components defensively.
    identifier = hashlib.sha256(event.eventid.encode()).hexdigest()
    path = root / camera / session / f"{identifier}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Evidence image encoding failed")
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    return str(path.resolve())
