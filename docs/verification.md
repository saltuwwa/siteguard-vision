# Dataset validation and CUDA training verification — 2026-11-15

The supplied local Construction-PPE dataset passed file/annotation integrity checks. A genuine **one-epoch YOLO26s fine-tuning run completed on CUDA:0**, preserving all **12 original classes**. Best/last checkpoints, training CSV, validation JSON, annotated examples and hardware telemetry are saved locally. No long training run was started. This establishes the training/inference plumbing; the one-epoch model is not deployment-ready, and real MP4 event evaluation remains unverified.

## Local input and root resolution

The supplied directory is `construction-ppe/construction-ppe` relative to this repository. Its upstream YAML says `path: construction-ppe`, which is not used as a working-directory-relative guess. The generated `data/local/construction-ppe.source.yaml` contains the resolved absolute root and the unchanged source names/split paths. This file and both `configs/datasets.local.yaml` and `configs/train-smoke.local.yaml` are gitignored. No machine-specific absolute paths were added to committed configuration examples.

The upstream YAML was **not modified**. SHA256 before and after training:

`d099873f1abf4b9862870b0b1305d01cd203914b1afb2c8bb217563af52488c3`

The original images and TXT annotations still match the pre-training audit hashes. Ultralytics created its normal train/val `.cache` files; images, labels and the upstream YAML were left intact. The dataset directory, archives, all weights and run outputs are gitignored.

## Exact observed dataset counts

- Training: **800 JPEG images, 800 matching label files, 11,902 boxes**.
- Validation: **200 JPEG images, 200 matching label files, 2,961 boxes**.
- Total: **1,000 images, 1,000 labels, 14,863 boxes, 12 classes**.
- The 1,001st TXT file is `LICENSE.txt`, not an orphan annotation.
- Missing image/label pairs: **0**. Orphan label files: **0**.
- Malformed YOLO rows, invalid class indices, nonfinite coordinates and normalized-component errors: **0**.
- JPEG decode/verification errors: **0**. All 1,000 image hashes match `manifest.jsonl`, and every manifest annotation count matches the label file.
- Duplicate annotation rows: **0**. Manifest dimension mismatches where dimensions were supplied: **0**.
- Exact duplicate image hashes: **0**. Perceptual duplicate candidates at Hamming distance <=4: **0**, including across train/validation.

Box counts by source class, in train / validation order:

- `0 helmet`: 1,282 / 294.
- `1 gloves`: 1,548 / 381.
- `2 vest`: 1,476 / 340.
- `3 boots`: 2,003 / 481.
- `4 goggles`: 612 / 139.
- `5 Person`: 1,962 / 481.
- `6 no_helmet`: 634 / 179.
- `7 no_goggle`: 265 / 82.
- `8 no_gloves`: 839 / 203.
- `9 no_boots`: 569 / 178.
- `10 forklift`: 275 / 73.
- `11 no_vest`: 437 / 130.

The initial validator rejected 25 files because its lower corner bound was strict while its upper corner bound already tolerated rounding. A detailed audit found 94 rows with a reconstructed edge just outside the image, with maximum normalized overshoot **5e-6**. Every stored x/y/w/h component is in [0,1]. The validator now applies a symmetric **1e-5 corner-only tolerance**, appropriate for this five-decimal export, and explicitly reports these rows. It still rejects real out-of-bounds boxes. **No labels were clipped or rewritten.** Regression tests exercise both cases.

Exact machine-readable reports:

- `runs/dataset-local/report.json`: project validation, per-class/split image and box counts, hashes, rounded rows and duplicate audit.
- `runs/dataset-local/manifest-audit.json`: all 1,000 supplied image checksums and annotation-count consistency.
- `runs/dataset-local/image-integrity.json`: JPEG verification, dimensions, duplicate rows and small-head counts.
- `runs/dataset-local/boundary-audit.json`: every rounded boundary row and maximum overshoot.
- `runs/dataset-local/final-summary.json`: consolidated dataset, training, metrics and artifact references.

Extracted image/label content fingerprint:

`4cde0c66eb89f8051c1cb747a3fceaf161eb54863d7d08a3cf5dcb388f9c7a3f`

## Annotation inspection, provenance and leakage limits

I visually inspected eight full-image annotation overlays and twelve focused helmet/body/bare-head crops, covering ten distinct source files in both splits. Saved sheets are in `runs/dataset-local/visual-review/` (`train-annotations.jpg`, `val-annotations.jpg`, `focus-crops.jpg`, plus the exact sample list).

The samples support `Person` as a visible human body, `helmet` as a protective hard hat and `no_helmet` as a visible uncovered head. The review included crowded/overlapping people and distant workers. Not every tiny distant head can be verified from an overview: the dataset contains 69 no_helmet and 39 helmet boxes with a short side below 12 pixels. This is a sample inspection, not an independent human audit of all 14,863 boxes. The review record explicitly identifies the reviewer as an AI agent and authorizes **one_epoch_smoke_only**; no human approval or new labels were fabricated.

The supplied README identifies the November 2026 Ultralytics release, CC BY 4.0, mixed human-collected/generated images and reviewed human/AI annotations. The included license is CC BY 4.0. Per-image real/generated provenance, camera, site, original video and recording-session groups are absent from the supplied manifest. File hashes agree with that manifest, but its upstream authenticity was not independently established against a downloaded archive. The archive checksum and download time remain **unknown** in the local import record, rather than being invented.

No full-image near duplicates were detected by the configured audit. This does **not** establish source-disjoint independence: visually related warehouse/construction scenes or variants outside the pHash threshold could share a source/site. Official partitions were preserved; no random resplitting was performed. `independence_proven` remains **false**. Validation is development validation, not an unbiased independent test.

## Class preservation and validation gates

The existing three-class canonical build would project away nine source labels. For this smoke, native-source validation binds the exact local YAML, all original class IDs/names and every image/label checksum to the validated report. It does not copy, filter, reorder or relabel the annotations. The small semantic mapping used by association is separate from the training label space.

Native source spot-check approval is restricted to the smoke scope. `stage: smoke` rejects any epoch count other than one. Canonical dataset review/build gates remain in place for their existing workflow; the source schema cannot accidentally pass through the three-class builder. A long baseline requires a separate explicitly appropriate review/run scope.

Actual executed commands:

```powershell
. .venv/Scripts/Activate.ps1
siteguard dataset validate --config configs/datasets.local.yaml
# After inspection and the fingerprint-bound, honestly scoped review record:
siteguard dataset validate --config configs/datasets.local.yaml
siteguard train --config configs/train-smoke.local.yaml
```

Final project validation output:

```json
{"errors": [], "leakage_pairs": 0, "review_required": false}
```

The completed run name already exists. Do not overwrite it when reproducing; choose a new run name while retaining epochs=1, the dataset fingerprint, settings and review scope.

## Ultralytics compatibility and real runtime execution

The exact `==8.4.61` gate was replaced by the bounded compatibility range **>=8.4.0,<8.5.0**. The installed and actually exercised version is **8.4.213**; lap is **0.5.14**. Version acceptance is not execution proof: version status explicitly leaves execution unverified until a real model operation runs. Hardware doctor reports package presence separately from model readiness. `YoloDetector.ready` becomes true only after actual inference on the requested device; a device mismatch fails explicitly.

Runtime:

- Python 3.11.9, torch **2.8.0+cu128**, torchvision **0.23.0+cu128**.
- GPU: NVIDIA GeForce RTX 5090, capability **12.0 / sm_120**, driver 616.92.
- FP32/FP16 matrix operations, CUDA convolutions and torchvision CUDA NMS passed the hardware checks.
- A real dataset image passed the local YOLO26s COCO initialization checkpoint runtime test on **cuda:0**. That initial 80-class checkpoint was not treated as a PPE detector.
- After fine-tuning, best.pt reloads with **all 12 exact source class names** and performs genuine PPE predictions on a real validation image through the project adapter on **cuda:0**.

Ultralytics 8.4.213 deep-copies the model into its prediction backend. The original model object can remain on CPU while inference runs on GPU. Actual runtime verification checks `predictor.device` and result tensors; both were cuda:0. See `runs/dataset-local/runtime-smoke.json` and the trained run's `ppe-runtime-smoke.json` / `ppe-runtime-smoke.jpg`.

## Actual one-epoch training and saved outputs

Configuration: **YOLO26s**, local pretrained `yolo26s.pt`, all 12 original classes, **epochs=1**, batch=8, imgsz=640, device=0, workers=0, seed=42, deterministic=true, **FP32 / AMP=false**, no external experiment sync. The epoch used all 800 training images and validation used all 200 images. The trainer log explicitly reports `Overriding model.yaml nc=80 with nc=12`.

The CLI training process exited **0**, completed exactly **1 epoch** and recorded `actual_training_device: cuda:0`. Total process wall time, including diagnostics/setup/final validation, was **80.7196 seconds**. Epoch CSV time was **35.353 seconds**. These are smoke timings, not a throughput benchmark.

Artifacts are preserved in:

`runs/train/construction-ppe-12class-smoke-20261115/`

- `weights/best.pt`, SHA256 `30d8e7c1fc1f632e88f05f81dbf7160ef377bfe42fa0bc431fd9e7fc869ac85e`.
- `weights/last.pt`, SHA256 `aefd8281515e33e80d560f662e63559e5e87dc61761c8155344a365cafc480f5`.
- `experiment.json`: exact input/checkpoint/data hashes, seed, augmentation, installed versions, hardware, class names, resolved trainer settings, source review scope, checkpoint references and epoch metrics.
- `results.csv`, `metrics.json`, `args.yaml`, loss/PR/confusion plots and saved training/validation sample images.
- `ppe-runtime-smoke.json` / `.jpg`: genuine reloaded PPE checkpoint inference.

The installed Ultralytics version initially resolved a relative output project underneath `runs/detect/`. The actual outputs were relocated into the intended experiment directory after completion; weight hashes were checked unchanged. Metadata records `original_native_save_dir`, and future calls pass an absolute resolved project path to avoid the extra nesting. Original trainer args/log history remains intact.

The training log is `runs/dataset-local/train-smoke.log`, and process result/timestamps are `runs/dataset-local/training-process.json`. `gpu-training.jsonl` contains 75 nvidia-smi samples. GPU 0 reached **7,461 MiB total device memory used** and **37% sampled utilization**. These are whole-device measurements that include other processes, not isolated peak model allocation. GPU 1 was touched by the hardware diagnostics; the actual trainer was on CUDA:0, with no distributed training.

## Real labeled validation results and their limits

Reloading best.pt and validating it on the official 200-image, 2,961-box development split produced:

- Precision: **0.6037410809**.
- Recall: **0.5574640189**.
- mAP50: **0.5777146682**.
- mAP50-95: **0.3744502031**.

Exact per-class metrics are in the run's `metrics.json`. Epoch-time values in `results.csv` differ slightly from reloaded-checkpoint validation; both were retained and are identified separately. No independent test or CCTV event metrics are claimed.

There are material model-quality limitations. In the saved trained prediction, some visually bare heads are classified as helmets. Reloaded validation no_helmet recall is about **0.223**, and no_goggle recall is zero at the reported operating point. A one-epoch training pass is therefore a successful engineering smoke, **not** a model suitable for safety alert deployment. No extra training was started to conceal this result.

## Automated checks and remaining gaps

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q --basetemp runs/dataset-local/full-tests --junitxml=runs/dataset-local/tests.xml
python -m compileall -q src scripts
# Genuine saved-checkpoint GPU test, using local gitignored mapping:
$env:SITEGUARD_GPU_MODEL=(Join-Path (Get-Location) 'runs/train/construction-ppe-12class-smoke-20261115/weights/best.pt')
$env:SITEGUARD_GPU_MAPPING='data/local/ppe-mapping.json'
python -m pytest tests/test_optional_ml.py -m gpu -q --basetemp runs/dataset-local/gpu-smoke-tests
```

Core suite: **40 passed, 3 skipped**. Compilation passed. The separately enabled genuine checkpoint GPU smoke passed: **1 passed, 1 deselected**. The real ByteTrack adapter test also executes and passes with installed dependencies. Three default skips remain the explicit GPU opt-in and the two HTTP tests disabled in this Windows sandbox.

Ruff and mypy are still unavailable. A fresh installation attempt failed resolving `https://pypi.org/simple/mypy/` with Windows DNS error 11001. Their commands were run and failed with missing-module errors; no clean lint/type-check result is claimed. No new infrastructure or UI was added.

Still outstanding: independent human annotation review, source-disjoint held-out data, a justified longer training experiment, model-quality improvements, and genuine recorded-video tracking/association/temporal-event evaluation. The dataset validation and one-epoch CUDA smoke requested in this task have completed; the broader real-MP4 acceptance milestone has not been claimed complete.

## First repository publication

The v0.1.0 repository snapshot includes the existing optional frontend prototype,
with demo/mock data explicitly distinguished from detector evaluation. The README
preview is a genuine trained-model prediction on a licensed Construction-PPE
validation image; its attribution and exact source/model hashes are in
`docs/assets/`.

Publication checks: backend tests passed (40 passed, 3 skipped), and Python
compilation passed. Frontend TypeScript checking passed. Frontend test/build
attempts in this Windows sandbox stopped with Vite `EPERM` errors on file
`realpath` operations before executing tests or bundling; those attempts do not
establish a passing frontend build. The local installed frontend dependency
versions also differ from package.json pins, so use a fresh installation and
the frontend's documented checks when reproducing. No production frontend
validation is claimed from this publication session.
