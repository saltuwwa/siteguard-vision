from pathlib import Path

from siteguard.evaluation.events import VideoGroundTruth
from siteguard.events.schema import ViolationEvent
from siteguard.training.datasets import write_json


def main() -> None:
    write_json(Path("examples/video-ground-truth.schema.json"), VideoGroundTruth.model_json_schema())
    schema = ViolationEvent.model_json_schema()
    # Broker snapshot omits individual observation boxes to stay compact.
    schema["properties"].pop("supportingobservations")
    schema["required"].remove("supportingobservations")
    schema["properties"]["event_type"] = {"const": "helmet_violation.confirmed", "type": "string"}
    schema["required"].append("event_type")
    write_json(Path("examples/kafka-event-v1.schema.json"), schema)


if __name__ == "__main__":
    main()
