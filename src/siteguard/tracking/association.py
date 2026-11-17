from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
import math

from siteguard.inference.base import Box, Detection
from siteguard.tracking.bytetrack import PersonTrack


class HeadState(StrEnum):
    UNKNOWN = "unknown"
    HELMET = "helmet"
    BAREHEAD = "barehead"


@dataclass(frozen=True)
class HeadEvidence:
    track_id: int
    state: HeadState = HeadState.UNKNOWN
    confidence: float = 0.0
    box: Box | None = None
    reason: str = "no_visible_head"


@dataclass
class AssociationDiagnostics:
    attempts: int = 0
    accepted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    rejected_by_class: dict[str, Counter[str]] = field(default_factory=dict)
    accepted_by_class: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str, count: int = 1, label: str | None = None) -> None:
        self.rejected[reason] += count
        if label:
            self.rejected_by_class.setdefault(label, Counter())[reason] += count


def candidate_score(person: Box, head: Box) -> float | None:
    x1, y1, x2, y2 = person
    hx1, hy1, hx2, hy2 = head
    width, height = x2-x1, y2-y1
    hw, hh = hx2-hx1, hy2-hy1
    if (not all(math.isfinite(v) for v in (*person, *head))
            or min(width, height, hw, hh) <= 0 or hw > width*0.8 or hh > height*0.4):
        return None
    cx, cy = (hx1+hx2)/2, (hy1+hy2)/2
    if not (x1 <= cx <= x2 and y1-height*0.05 <= cy <= y1+height*0.38):
        return None
    intersection = max(0, min(hx2, x2)-max(hx1, x1))*max(0, min(hy2, y1+height*.4)-max(hy1, y1-height*.05))
    containment = intersection/(hw*hh)
    if containment < 0.8:
        return None
    return containment - abs(cx-(x1+x2)/2)/width*0.3 - abs(cy-(y1+height*.12))/height*0.3


def associate_heads(people: list[PersonTrack], detections: list[Detection],
                    min_confidence: float = 0.6, min_pixels: int = 12,
                    ambiguity_margin: float = 0.12,
                    image_shape: tuple[int, int] | None = None,
                    diagnostics: AssociationDiagnostics | None = None, *,
                    min_person_confidence: float | None = None) -> dict[int, HeadEvidence]:
    person_threshold = min_confidence if min_person_confidence is None else min_person_confidence
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value in (min_confidence, person_threshold)):
        raise ValueError("Association confidence thresholds must be finite and within[0,1]")
    audit = diagnostics if diagnostics is not None else AssociationDiagnostics()
    result = {p.track_id: HeadEvidence(p.track_id) for p in people}
    assigned: dict[int, list[Detection]] = {}
    ambiguous: set[int] = set()
    for head in detections:
        if head.label not in {"helmet", "barehead"}:
            continue
        audit.attempts += 1
        x1, y1, x2, y2 = head.box
        if not all(math.isfinite(v) for v in (*head.box, head.confidence)) or x2 <= x1 or y2 <= y1:
            audit.reject("invalid_head_geometry", label=head.label)
            continue
        if head.confidence < min_confidence:
            audit.reject("low_detection_confidence", label=head.label)
            continue
        if min(x2-x1, y2-y1) < min_pixels:
            audit.reject("head_too_small", label=head.label)
            continue
        if image_shape and (x1 <= 1 or y1 <= 1 or x2 >= image_shape[1]-1 or y2 >= image_shape[0]-1):
            audit.reject("head_clipped", label=head.label)
            continue
        candidates = []
        low_person_candidate = False
        for person in people:
            score = candidate_score(person.box, head.box)
            if not math.isfinite(person.confidence) or person.confidence < person_threshold:
                low_person_candidate |= score is not None
                continue
            if score is not None:
                candidates.append((score, person.track_id))
        candidates.sort(reverse=True)
        if not candidates:
            audit.reject("person_confidence_too_low" if low_person_candidate else "no_matching_person",
                         label=head.label)
            continue
        if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < ambiguity_margin:
            # A distant third candidate must not poison that person's own valid head.
            ambiguous.update(track_id for score, track_id in candidates
                             if candidates[0][0]-score < ambiguity_margin)
            audit.reject("ambiguous_person_association", label=head.label)
            continue
        assigned.setdefault(candidates[0][1], []).append(head)
    for track_id, heads in assigned.items():
        reason: str | None = None
        if track_id in ambiguous:
            reason = "ambiguous_person_association"
        elif len({h.label for h in heads}) != 1:
            reason = "conflicting_head_predictions"
        elif len(heads) > 1:
            best = max(heads, key=lambda h: h.confidence)
            for other in heads:
                a, b = best.box, other.box
                intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(
                    0., min(a[3], b[3])-max(a[1], b[1]))
                area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
                union = area_a+area_b-intersection
                # Native no_helmet predictions can localize scalp and full head
                # separately. Nested same-class boxes are one head even if IoU
                # is small; two spatially separate heads remain unsafe.
                duplicate = union > 0 and (intersection/union >= .5 or
                    intersection/min(area_a, area_b) >= .8)
                if not duplicate:
                    reason = "multiple_distinct_heads"
                    break
        if reason:
            for head in heads:
                audit.reject(reason, label=head.label)
            result[track_id] = HeadEvidence(track_id, reason=reason)
            continue
        best = max(heads, key=lambda h: h.confidence)
        audit.accepted += 1
        audit.accepted_by_class[best.label] += 1
        if len(heads) > 1:
            audit.reject("duplicate_head_prediction", len(heads)-1, label=best.label)
        result[track_id] = HeadEvidence(track_id, HeadState(best.label), best.confidence,
                                        best.box, "visible_associated_head")
    for track_id in ambiguous:
        result[track_id] = HeadEvidence(track_id, reason="ambiguous_person_association")
    return result
