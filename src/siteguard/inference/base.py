from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float
    label: str


class Detector(Protocol):
    version: str

    def detect(self, image: ImageArray) -> list[Detection]: ...
