"""Generate codec/decode test input, not model accuracy ground truth."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/synthetic.mp4"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(arguments.output), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    try:
        for i in range(100):
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            offset = i % 20
            cv2.rectangle(image, (100+offset, 40), (170+offset, 230), (150, 150, 150), -1)
            cv2.circle(image, (135+offset, 45), 20, (100, 160, 200), -1)
            cv2.putText(image, "SYNTHETIC CODEC FIXTURE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (255, 255, 255), 1)
            writer.write(image)
    finally:
        writer.release()
    print(arguments.output)


if __name__ == "__main__":
    main()
