import pytest

from siteguard.config.settings import RulesConfig
from siteguard.rules.temporal import State, ViolationEngine
from siteguard.tracking.association import HeadEvidence, HeadState, associate_heads
from siteguard.tracking.bytetrack import PersonTrack
from siteguard.inference.base import Detection


def engine() -> ViolationEngine:
    return ViolationEngine("cam", "session", "test-only", RulesConfig())


def head(state: HeadState = HeadState.BAREHEAD) -> HeadEvidence:
    return HeadEvidence(1, state, .9, (10, 10, 30, 30))


def test_unknown_never_confirms() -> None:
    rules = engine()
    for t in range(20):
        assert not rules.observe(float(t), head(HeadState.UNKNOWN))
    assert rules.tracks[1].state == State.UNKNOWN


def test_sustained_confirmation_dedup_clear_cooldown() -> None:
    rules = engine()
    events = []
    for i in range(12):
        events.extend(rules.observe(i*.25, head()))
    assert len({e.eventid for e in events}) == 1
    assert rules.tracks[1].state == State.CONFIRMED
    assert events[0].confirmedtimestamp == 1.5
    for t in (3., 3.5, 4., 4.5):
        cleared = rules.observe(t, head(HeadState.HELMET))
        if cleared:
            assert cleared[0].status == "cleared"
    for t in (5., 5.5, 6., 6.5):
        assert not rules.observe(t, head())


def test_missing_frames_and_jitter() -> None:
    rules = engine()
    for t in (0., .21, .44, 2., 2.23, 2.45):
        assert not rules.observe(t, head())
    with pytest.raises(ValueError):
        rules.observe(2.4, head())
    rules.finish("reconnect")
    assert not rules.tracks


def test_association_conflict_ambiguous_absence() -> None:
    people = [PersonTrack(1, (0, 0, 100, 300), .9)]
    bare = Detection((30, 10, 60, 45), .9, "barehead")
    helmet = Detection(bare.box, .95, "helmet")
    assert associate_heads(people, [bare])[1].state == HeadState.BAREHEAD
    assert associate_heads(people, [bare, helmet])[1].state == HeadState.UNKNOWN
    assert associate_heads(people, [])[1].state == HeadState.UNKNOWN
    people.append(PersonTrack(2, (1, 0, 101, 300), .9))
    result = associate_heads(people, [helmet])
    assert all(e.state == HeadState.UNKNOWN for e in result.values())


def test_lost_track_closes_and_scope_differs() -> None:
    rules = engine()
    for t in (0., .5, 1., 1.5):
        events = rules.observe(t, head())
    original = events[0]
    ended = rules.tick(6., set())
    assert ended[0].eventid == original.eventid
    assert ended[0].status == "ended"
    other = ViolationEngine("cam", "another-session", "test-only", RulesConfig())
    for t in (0., .5, 1., 1.5):
        events = other.observe(t, head())
    assert events[0].eventid != original.eventid


def test_clearance_requires_contiguous_visible_support() -> None:
    rules = engine()
    for t in (0., .5, 1., 1.5):
        rules.observe(t, head())
    assert not rules.observe(2., head(HeadState.HELMET))
    # Two isolated compliant observations must not demonstrate sustained compliance.
    assert not rules.observe(3.2, head(HeadState.HELMET))
    assert rules.tracks[1].event is not None
    assert not rules.observe(3.7, head(HeadState.HELMET))
    closed = rules.observe(4.2, head(HeadState.HELMET))
    assert len(closed) == 1 and closed[0].status == "cleared"
