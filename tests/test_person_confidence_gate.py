import math

import pytest

from siteguard.config.settings import RulesConfig
from siteguard.inference.base import Detection
from siteguard.tracking.association import AssociationDiagnostics, HeadState, associate_heads
from siteguard.tracking.bytetrack import PersonTrack


def test_separate_body_gate_preserves_head_confidence_and_default_behavior():
    people = [PersonTrack(1, (10., 10., 100., 200.), .55)]
    head = Detection((35., 15., 65., 45.), .95, "barehead")
    audit = AssociationDiagnostics()
    assert associate_heads(people, [head], diagnostics=audit)[1].state == HeadState.UNKNOWN
    assert audit.rejected["person_confidence_too_low"] == 1
    assert RulesConfig().min_person_confidence is None
    observed = associate_heads(people, [head], min_person_confidence=.5)[1]
    assert observed.state == HeadState.BAREHEAD and observed.confidence == .95
    weak_head = Detection(head.box, .59, "barehead")
    audit = AssociationDiagnostics()
    assert associate_heads(people, [weak_head], min_person_confidence=.5,
                           diagnostics=audit)[1].state == HeadState.UNKNOWN
    assert audit.rejected["low_detection_confidence"] == 1


def test_additional_body_candidate_can_expose_ambiguity_instead_of_assigning_wrong_worker():
    people = [PersonTrack(1, (10., 10., 100., 200.), .95),
              PersonTrack(2, (10., 10., 100., 200.), .55)]
    head = Detection((35., 15., 65., 45.), .95, "barehead")
    audit = AssociationDiagnostics()
    result = associate_heads(people, [head], min_person_confidence=.5, diagnostics=audit)
    assert all(e.state == HeadState.UNKNOWN for e in result.values())
    assert audit.rejected["ambiguous_person_association"] == 1
    assert audit.accepted == 0
    conflict = [head, Detection(head.box, .95, "helmet")]
    assert associate_heads(people[:1], conflict, min_person_confidence=.5)[1].state == HeadState.UNKNOWN


@pytest.mark.parametrize("threshold", [-.1, 1.1, math.nan, math.inf])
def test_independent_association_entrypoint_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="finite"):
        associate_heads([], [], min_person_confidence=threshold)
