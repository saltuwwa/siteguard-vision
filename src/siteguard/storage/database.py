from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from siteguard.events.schema import ViolationEvent


class EventStore:
    """One short-lived connection per transaction; SQLite WAL or optional psycopg.

    Schema is versioned in metadata. Use one publisher per database; broker-side
    idempotency is not a substitute for eventid-based consumer deduplication.
    """
    def __init__(self, url: str) -> None:
        self.url = url
        self.sqlite = url.startswith("sqlite:///")
        if self.sqlite:
            self.path = url.removeprefix("sqlite:///")
            if self.path == ":memory:":
                raise ValueError("Use a temporary SQLite file; connections are transaction scoped")
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        elif not url.startswith("postgresql://"):
            raise ValueError("Expected sqlite:///path or postgresql:// URL")
        with self.connection() as conn:
            statements = [
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS events (eventid TEXT PRIMARY KEY, cameraid TEXT NOT NULL, "
                "createdat TEXT NOT NULL, payload TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS events_camera_created ON events(cameraid, createdat)",
                "CREATE TABLE IF NOT EXISTS cameras (cameraid TEXT PRIMARY KEY, sessionid TEXT NOT NULL, "
                "lastseen TEXT NOT NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS outbox (eventid TEXT PRIMARY KEY REFERENCES events(eventid), "
                "payload TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, "
                "nextattempt DOUBLE PRECISION NOT NULL DEFAULT 0, lasterror TEXT)",
            ]
            for statement in statements:
                self.execute(conn, statement)
            self.execute(conn, "INSERT INTO metadata(key,value) VALUES('schema_version','1') "
                              "ON CONFLICT(key) DO NOTHING")
            version = self.execute(conn, "SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if version["value"] != "1":
                raise RuntimeError("Unsupported database schema version")

    @contextmanager
    def connection(self) -> Iterator[Any]:
        if self.sqlite:
            conn: Any = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
        else:
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(self.url, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, conn: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
        return conn.execute(sql if self.sqlite else sql.replace("?", "%s"), params)

    def save(self, event: ViolationEvent, publish: bool = False) -> None:
        with self.connection() as conn:
            self.execute(conn, "INSERT INTO events(eventid,cameraid,createdat,payload) VALUES(?,?,?,?) "
                "ON CONFLICT(eventid) DO UPDATE SET payload=excluded.payload",
                (event.eventid, event.cameraid, event.createdat.isoformat(), event.model_dump_json()))
            if publish:
                payload = event.model_dump(mode="json", exclude={"supportingobservations"})
                payload["event_type"] = "helmet_violation.confirmed"
                self.execute(conn, "INSERT INTO outbox(eventid,payload,state) VALUES(?,?,'pending') "
                    "ON CONFLICT(eventid) DO NOTHING", (event.eventid, json.dumps(payload)))

    def get(self, event_id: str) -> ViolationEvent | None:
        with self.connection() as conn:
            row = self.execute(conn, "SELECT payload FROM events WHERE eventid=?", (event_id,)).fetchone()
            return ViolationEvent.model_validate_json(row["payload"]) if row else None

    def list_events(self, limit: int = 50, offset: int = 0,
                    camera_id: str | None = None) -> tuple[int, list[ViolationEvent]]:
        clause = " WHERE cameraid=?" if camera_id else ""
        params: tuple[Any, ...] = (camera_id,) if camera_id else ()
        with self.connection() as conn:
            total = self.execute(conn, "SELECT COUNT(*) AS n FROM events"+clause, params).fetchone()["n"]
            rows = self.execute(conn, "SELECT payload FROM events"+clause+
                " ORDER BY createdat DESC, eventid LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
            return total, [ViolationEvent.model_validate_json(row["payload"]) for row in rows]

    def update_camera(self, camera_id: str, session_id: str, metrics: dict[str, Any]) -> None:
        with self.connection() as conn:
            self.execute(conn, "INSERT INTO cameras(cameraid,sessionid,lastseen,payload) VALUES(?,?,?,?) "
                "ON CONFLICT(cameraid) DO UPDATE SET sessionid=excluded.sessionid, "
                "lastseen=excluded.lastseen,payload=excluded.payload",
                (camera_id, session_id, datetime.now(UTC).isoformat(), json.dumps(metrics)))

    def cameras(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return [{"cameraid": r["cameraid"], "streamsessionid": r["sessionid"],
                "lastseen": r["lastseen"], "metrics": json.loads(r["payload"])}
                for r in self.execute(conn, "SELECT * FROM cameras ORDER BY cameraid").fetchall()]

    def pending(self, limit: int = 100, now: float | None = None) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = self.execute(conn, "SELECT * FROM outbox WHERE state IN ('pending','retry') "
                "AND nextattempt<=? ORDER BY nextattempt,eventid LIMIT ?",
                (time.time() if now is None else now, limit)).fetchall()
            return [dict(row) for row in rows]

    def publication_result(self, event_id: str, error: str | None, max_attempts: int = 5) -> None:
        with self.connection() as conn:
            row = self.execute(conn, "SELECT attempts FROM outbox WHERE eventid=?", (event_id,)).fetchone()
            if row is None:
                raise KeyError(event_id)
            attempts = row["attempts"]+1
            state = "delivered" if error is None else ("dead" if attempts >= max_attempts else "retry")
            self.execute(conn, "UPDATE outbox SET state=?,attempts=?,nextattempt=?,lasterror=? WHERE eventid=?",
                (state, attempts, time.time()+min(300, 2**attempts), error, event_id))

    def replay_dead(self) -> int:
        with self.connection() as conn:
            cursor = self.execute(conn, "UPDATE outbox SET state='pending',attempts=0,nextattempt=0 "
                                        "WHERE state='dead'")
            return int(cursor.rowcount)

    def publication_metrics(self) -> dict[str, int]:
        with self.connection() as conn:
            return {row["state"]: row["n"] for row in self.execute(conn,
                "SELECT state,COUNT(*) AS n FROM outbox GROUP BY state").fetchall()}

    def cleanup(self, before: datetime, evidence_root: Path) -> int:
        # Only closed incidents; pending/dead events must remain recoverable.
        with self.connection() as conn:
            rows = self.execute(conn, "SELECT e.eventid,e.payload FROM events e LEFT JOIN outbox o "
                "ON e.eventid=o.eventid WHERE e.createdat<? AND (o.state IS NULL OR o.state='delivered')",
                (before.isoformat(),)).fetchall()
            count = 0
            for row in rows:
                event = ViolationEvent.model_validate_json(row["payload"])
                if event.status == "confirmed":
                    continue
                paths = [Path(p).resolve() for p in (event.evidenceimagepath, event.evidencevideopath) if p]
                if any(not p.is_relative_to(evidence_root.resolve()) for p in paths):
                    raise ValueError("Evidence path outside retention root")
                for path in paths:
                    path.unlink(missing_ok=True)
                self.execute(conn, "DELETE FROM outbox WHERE eventid=?", (event.eventid,))
                self.execute(conn, "DELETE FROM events WHERE eventid=?", (event.eventid,))
                count += 1
            return count
