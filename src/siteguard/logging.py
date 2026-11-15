"""Small structured logger with an explicit safe field allowlist."""
import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"timestamp": datetime.now(UTC).isoformat(), "level": record.levelname,
                   "message": record.getMessage()}
        for field in ("cameraid", "sessionid", "eventid", "reason", "attempt", "error_type"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload)


def configure_logging() -> None:
    logger = logging.getLogger("siteguard")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
