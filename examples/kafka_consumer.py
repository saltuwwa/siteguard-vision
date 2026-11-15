"""Minimal at-least-once consumer: deduplicate in the same transaction as side effects.

Set KAFKA_BOOTSTRAP_SERVERS and optionally KAFKA_CONFIG_JSON for SASL/TLS.
This example persists metadata only; it never downloads evidence.
"""
import json
import os
import sqlite3

from confluent_kafka import Consumer


def main() -> None:
    database = sqlite3.connect("consumer.sqlite")
    database.execute("CREATE TABLE IF NOT EXISTS received(eventid TEXT PRIMARY KEY,payload TEXT NOT NULL)")
    configuration = json.loads(os.getenv("KAFKA_CONFIG_JSON", "{}"))
    consumer = Consumer({"bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        "group.id": "siteguard-example", "enable.auto.commit": False,
        "auto.offset.reset": "earliest", **configuration})
    consumer.subscribe(["siteguard.violations.v1"])
    try:
        while True:
            message = consumer.poll(1.)
            if message is None:
                continue
            if message.error():
                raise RuntimeError("Consumer poll failed")
            raw = message.value().decode()
            payload = json.loads(raw)
            if payload["schema_version"] != 1:
                raise ValueError("Unsupported schema; message remains uncommitted")
            with database:
                database.execute("INSERT INTO received VALUES(?,?) ON CONFLICT(eventid) DO NOTHING",
                                 (payload["eventid"], raw))
                # Additional database side effects must occur in this same transaction.
            consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
        database.close()


if __name__ == "__main__":
    main()
