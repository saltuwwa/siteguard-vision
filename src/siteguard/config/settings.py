from pathlib import Path
import os
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RulesConfig(StrictModel):
    min_observations: int = Field(default=4, ge=2)
    window_seconds: float = Field(default=4.0, gt=0)
    min_duration: float = Field(default=1.5, gt=0)
    max_gap: float = Field(default=0.8, gt=0)
    clear_seconds: float = Field(default=1.0, gt=0)
    cooldown_seconds: float = Field(default=5.0, ge=0)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    min_head_pixels: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def duration_fits(self) -> "RulesConfig":
        if self.min_duration > self.window_seconds:
            raise ValueError("min_duration must fit in window_seconds")
        return self


class RuntimeConfig(StrictModel):
    model: str = "models/ppe-best.pt"
    model_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    model_mapping: dict[str, str] = Field(
        default_factory=lambda: {"person": "person", "helmet": "helmet", "barehead": "barehead"}
    )
    device: str = "auto"
    half: bool = False
    confidence: float = Field(default=0.15, ge=0, le=1)
    imgsz: int = Field(default=640, ge=32)
    inference_fps: float = Field(default=8.0, gt=0)
    max_width: int | None = Field(default=1280, ge=32)
    queue_size: int = Field(default=8, ge=1)
    live_drop_policy: Literal["drop_oldest", "drop_newest"] = "drop_oldest"
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    open_timeout_ms: int = Field(default=5000, ge=100)
    read_timeout_ms: int = Field(default=5000, ge=100)
    reconnect_attempts: int = Field(default=10, ge=0)
    reconnect_delay: float = Field(default=1.0, gt=0)
    track_buffer_seconds: float = Field(default=2.0, gt=0)
    cut_threshold: float = Field(default=0.65, ge=0, le=1)
    database_url: str = "sqlite:///runs/siteguard.sqlite"
    kafka_enabled: bool = False
    rules: RulesConfig = Field(default_factory=lambda: RulesConfig())


T = TypeVar("T", bound=BaseModel)


def load_config(path: Path, schema: type[T]) -> T:
    with path.open(encoding="utf-8") as stream:
        contents = yaml.safe_load(stream) or {}
    if schema is RuntimeConfig and os.getenv("SITEGUARD_DATABASE_URL"):
        contents["database_url"] = os.environ["SITEGUARD_DATABASE_URL"]
    return schema.model_validate(contents)
