from datetime import UTC, datetime, timedelta
from pathlib import Path
import os

import pytest

from fastapi.testclient import TestClient

from siteguard.api.app import create_app
from siteguard.config.settings import RulesConfig
from siteguard.rules.temporal import ViolationEngine
from siteguard.storage.database import EventStore
from siteguard.tracking.association import HeadEvidence, HeadState


def test_persistence_dedup_api_and_retention(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path/'events.sqlite'}")
    rules = ViolationEngine("cam", "session", "test-only", RulesConfig())
    for t in (0., .5, 1., 1.5):
        events = rules.observe(t, HeadEvidence(1, HeadState.BAREHEAD, .9))
    event = events[0]
    store.save(event, publish=True)
    store.save(event, publish=True)
    assert len(store.pending()) == 1
    assert store.list_events()[0] == 1
    assert store.cleanup(datetime.now(UTC)+timedelta(days=1), tmp_path) == 0
    event = rules.finish()[0]
    store.save(event)
    store.publication_result(event.eventid, None)
    assert store.cleanup(datetime.now(UTC)+timedelta(days=1), tmp_path) == 1


@pytest.mark.skipif(os.name == "nt" and not os.getenv("SITEGUARD_SOCKET_TESTS"),
                    reason="Windows sandbox socketpair unavailable; run in Linux/WSL CI or set SITEGUARD_SOCKET_TESTS=1")
def test_http_api(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path/'api.sqlite'}")
    rules = ViolationEngine("cam", "session", "test-only", RulesConfig())
    for t in (0., .5, 1., 1.5):
        events = rules.observe(t, HeadEvidence(1, HeadState.BAREHEAD, .9))
    event = events[0]
    store.save(event)
    client = TestClient(create_app(store, token="test-secret"))
    assert client.get("/health").status_code == 200
    assert client.get("/events").status_code == 401
    headers = {"Authorization": "Bearer test-secret"}
    page = client.get("/events", headers=headers).json()
    assert page["total"] == 1
    assert client.get(f"/events/{event.eventid}", headers=headers).json()["reviewstatus"] == "pending"
    assert client.get("/events/missing", headers=headers).json()["error"]["code"] == 404
    assert client.get("/events?limit=201", headers=headers).status_code == 422
    assert client.get("/ready", headers=headers).status_code == 200


@pytest.mark.skipif(os.name == "nt" and not os.getenv("SITEGUARD_SOCKET_TESTS"),
                    reason="Windows sandbox socketpair unavailable; Linux CI covers HTTP")
def test_no_token_denies_access(tmp_path: Path) -> None:
    client = TestClient(create_app(EventStore(f"sqlite:///{tmp_path/'db.sqlite'}"), token=""))
    assert client.get("/cameras").status_code == 503
