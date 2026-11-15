"""Optional outbox drainer. The video worker never contacts Kafka."""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from siteguard.storage.database import EventStore


class EventPublisher(Protocol):
    def publish(self, event_id: str, payload: str) -> None: ...


class KafkaPublisher:
    def __init__(self, configuration: dict[str, Any], topic: str = "siteguard.violations.v1",
                 delivery_timeout: float = 10.0) -> None:
        from confluent_kafka import Producer

        self.producer = Producer({"enable.idempotence": True, "acks": "all",
            "message.timeout.ms": round(delivery_timeout*1000),
            "socket.timeout.ms": round(delivery_timeout*1000),
            **configuration})
        self.topic = topic
        self.timeout = delivery_timeout

    def publish(self, event_id: str, payload: str) -> None:
        data = json.loads(payload)
        if data.get("schema_version") != 1 or data.get("event_type") != "helmet_violation.confirmed":
            raise ValueError("Only schema v1 confirmed violations may be published")
        if any(k in data for k in ("frame", "image", "raw_frame", "supportingobservations")):
            raise ValueError("Broker payload must contain compact metadata and evidence references")
        delivered: list[bool] = []

        def callback(error: Any, message: Any) -> None:
            delivered.append(error is None)

        self.producer.produce(self.topic, key=event_id.encode(), value=payload.encode(), on_delivery=callback)
        remaining = self.producer.flush(self.timeout)
        if remaining or delivered != [True]:
            raise RuntimeError("Kafka delivery failed or timed out (broker details suppressed)")


def drain_outbox(store: EventStore, publisher: EventPublisher, limit: int = 100,
                 max_attempts: int = 5) -> dict[str, int]:
    if not 1 <= limit <= 10000 or not 1 <= max_attempts <= 20:
        raise ValueError("Invalid bounded publication limits")
    delivered = failed = 0
    for row in store.pending(limit):
        try:
            publisher.publish(row["eventid"], row["payload"])
        except Exception as exc:
            # Persist a safe error category, never SASL config, credentials or broker URL.
            store.publication_result(row["eventid"], type(exc).__name__, max_attempts)
            failed += 1
            logging.getLogger("siteguard.kafka").error("outbox_delivery_failed", extra={
                "eventid": row["eventid"], "error_type": type(exc).__name__})
        else:
            store.publication_result(row["eventid"], None, max_attempts)
            delivered += 1
    return {"delivered": delivered, "failed": failed, **store.publication_metrics()}
