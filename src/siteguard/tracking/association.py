from dataclasses import dataclass
from enum import StrEnum

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


def candidate_score(person: Box, head: Box) -> float | None:
    x1, y1, x2, y2 = person
    hx1, hy1, hx2, hy2 = head
    width, height = x2-x1, y2-y1
    hw, hh = hx2-hx1, hy2-hy1
    if min(width, height, hw, hh) <= 0 or hw > width*0.8 or hh > height*0.4:
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
                    image_shape: tuple[int, int] | None = None) -> dict[int, HeadEvidence]:
    result = {p.track_id: HeadEvidence(p.track_id) for p in people}
    assigned: dict[int, list[Detection]] = {}
    ambiguous: set[int] = set()
    for head in detections:
        if head.label not in {"helmet", "barehead"} or head.confidence < min_confidence:
            continue
        x1, y1, x2, y2 = head.box
        if min(x2-x1, y2-y1) < min_pixels:
            continue
        if image_shape and (x1 <= 1 or y1 <= 1 or x2 >= image_shape[1]-1 or y2 >= image_shape[0]-1):
            continue
        candidates = []
        for person in people:
            if person.confidence < min_confidence:
                continue
            score = candidate_score(person.box, head.box)
            if score is not None:
                candidates.append((score, person.track_id))
        candidates.sort(reverse=True)
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < ambiguity_margin:
            ambiguous.update(track_id for _, track_id in candidates)
            continue
        assigned.setdefault(candidates[0][1], []).append(head)
    for track_id, heads in assigned.items():
        if track_id in ambiguous or len({h.label for h in heads}) != 1:
            result[track_id] = HeadEvidence(track_id, reason="ambiguous_or_conflicting_heads")
            continue
        best = max(heads, key=lambda h: h.confidence)
        result[track_id] = HeadEvidence(track_id, HeadState(best.label), best.confidence,
                                        best.box, "visible_associated_head")
    for track_id in ambiguous:
        result[track_id] = HeadEvidence(track_id, reason="multiple_possible_people")
    return result
