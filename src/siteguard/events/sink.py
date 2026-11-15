import queue
import threading
from pathlib import Path
from typing import TextIO

from siteguard.events.schema import ViolationEvent
from siteguard.inference.base import ImageArray
from siteguard.storage.database import EventStore
from siteguard.storage.evidence import save_evidence


class EventSink:
    """Bounded event I/O thread. Backpressure or explicit failure, never silent loss."""
    def __init__(self, store: EventStore, evidence_root: Path, log: TextIO,
                 publish: bool = False) -> None:
        self.store = store
        self.root = evidence_root
        self.log = log
        self.publish = publish
        self.queue: queue.Queue[tuple[ViolationEvent, ImageArray | None] | None] = queue.Queue(32)
        self.error: Exception | None = None
        self.paths: dict[str, str] = {}
        self.thread = threading.Thread(target=self._run, name="event-io", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                event, image = item
                path = self.paths.get(event.eventid)
                if path is None:
                    if image is None:
                        raise RuntimeError("First confirmation requires image evidence")
                    path = save_evidence(self.root, event, image)
                    self.paths[event.eventid] = path
                event.evidenceimagepath = path
                self.store.save(event, self.publish)
                self.log.write(event.model_dump_json()+"\n")
                self.log.flush()
                if event.status != "confirmed":
                    self.paths.pop(event.eventid, None)
        except Exception as exc:
            self.error = exc

    def _put(self, item: tuple[ViolationEvent, ImageArray | None] | None) -> None:
        while True:
            if self.error:
                raise RuntimeError("Event I/O failed") from self.error
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def submit(self, event: ViolationEvent, image: ImageArray | None) -> None:
        self._put((event.model_copy(deep=True), image.copy() if image is not None else None))

    def close(self) -> None:
        self._put(None)
        self.thread.join()
        if self.error:
            raise RuntimeError("Event I/O failed") from self.error
