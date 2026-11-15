from pathlib import Path

from siteguard.config.settings import RulesConfig
from siteguard.events.kafka import drain_outbox
from siteguard.rules.temporal import ViolationEngine
from siteguard.storage.database import EventStore
from siteguard.tracking.association import HeadEvidence, HeadState


class Publisher:
    def __init__(self) -> None:
        self.fail = True
        self.keys: list[str] = []

    def publish(self, event_id: str, payload: str) -> None:
        if self.fail:
            raise RuntimeError("fixture broker failure")
        self.keys.append(event_id)


def test_failed_publication_persists_replays_and_deduplicates(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path/'db.sqlite'}")
    rules = ViolationEngine("cam", "session", "test", RulesConfig())
    for timestamp in (0., .5, 1., 1.5):
        events = rules.observe(timestamp, HeadEvidence(1, HeadState.BAREHEAD, .9))
    event = events[0]
    store.save(event, True)
    store.save(event, True)
    publisher = Publisher()
    result = drain_outbox(store, publisher, max_attempts=1)
    assert result["failed"] == 1 and result["dead"] == 1
    # Reopen to verify persistence across a worker/publisher restart.
    restarted = EventStore(store.url)
    assert restarted.publication_metrics()["dead"] == 1
    assert restarted.replay_dead() == 1
    publisher.fail = False
    assert drain_outbox(restarted, publisher)["delivered"] == 1
    assert publisher.keys == [event.eventid]
    assert restarted.get(event.eventid) is not None
    assert not restarted.pending()


def test_no_kafka_mode_has_no_outbox(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path/'db.sqlite'}")
    rules = ViolationEngine("cam", "session", "test", RulesConfig())
    for t in (0., .5, 1., 1.5):
        events = rules.observe(t, HeadEvidence(1, HeadState.BAREHEAD, .9))
    store.save(events[0])
    assert store.publication_metrics() == {}
