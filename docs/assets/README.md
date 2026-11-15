# Documentation image attribution

`helmet-detection-example.jpg` shows actual YOLO26s predictions after the recorded
one-epoch Construction-PPE training smoke. The source image is the validation
image `ultralytics-ppe-nov2026-021.jpg` from the **Ultralytics Construction-PPE
Dataset**, Ultralytics team, November 2026.

Images and annotations are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Dataset source: [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/).

Changes: resized/cropped for documentation and overlaid with model-predicted boxes,
class names and confidence scores. Only Person, helmet and no_helmet predictions
are shown. The photo is a selected illustration, not an accuracy measurement or
independent test. The dataset contains mixed human-collected/generated imagery;
per-image origin was not supplied in its manifest.

`preview-provenance.json` records the exact source image and model hashes.
The full dataset, model weights and private run artifacts are excluded from git.
