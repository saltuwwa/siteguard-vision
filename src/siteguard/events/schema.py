from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from siteguard.config.settings import StrictModel


class SupportingObservation(StrictModel):
    timestamp: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    head_box: tuple[float, float, float, float] | None = None


class ViolationEvent(StrictModel):
    schema_version: Literal[1] = 1
    eventid: str
    cameraid: str
    streamsessionid: str
    trackid: int
    starttimestamp: float = Field(ge=0)
    endtimestamp: float | None = None
    confirmedtimestamp: float = Field(ge=0)
    status: Literal["confirmed", "cleared", "ended"] = "confirmed"
    endreason: str | None = None
    modelversion: str
    confidencesummary: dict[str, float]
    supportingobservations: list[SupportingObservation]
    observationcount: int = Field(ge=2)
    evidenceimagepath: str | None = None
    evidencevideopath: str | None = None
    createdat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewstatus: Literal["pending", "accepted", "rejected"] = "pending"

    @model_validator(mode="after")
    def valid_interval(self) -> "ViolationEvent":
        if self.confirmedtimestamp < self.starttimestamp:
            raise ValueError("Confirmation precedes event start")
        if self.endtimestamp is not None and self.endtimestamp < self.starttimestamp:
            raise ValueError("Event end precedes start")
        return self
