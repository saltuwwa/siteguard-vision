# Dataset validation and CUDA baseline verification - 2026-11-15

The unchanged 12-class dataset passed integrity checks. After the one-epoch smoke, the user-requested **80-epoch YOLO26s baseline completed on CUDA:0**. Repeated validation increased `no_helmet` recall from **0.22346 to 0.65922** and overall mAP50–95 from **0.37445 to 0.60088**. These are development validation results, not independent test or production event performance. The checkpoint executes in the SiteGuard adapter. Genuine MP4 evaluation remains blocked by absent construction video assets.

## Full baseline: error analysis, training and comparison

The machine-readable [baseline report](reports/baseline-20261115.json) records all classes' object/image counts, precision, recall, AP50 and AP50–95, size distributions, checkpoint hashes and comparison settings. Full local artifacts are in `runs/nohelmet-baseline/` and `runs/train/construction-ppe-12class-baseline80-20261115/`. Weights, datasets, private configuration and run images remain gitignored.

### Phase 1: observed errors and annotation semantics

`scripts/analyze_checkpoint.py` evaluates genuine weights and saves native confusion/PR plots, individual `PR-<class>.png` files for all 12 classes, a fixed-threshold 13-by-13 confusion matrix including background, 24 full-image error overlays and head crops. Green boxes are source annotations; red boxes are predictions. Predictions never become labels. `errors/diagnostics.json` contains all errors, including those beyond the visualization limit.

The smoke model had **34** `no_helmet -> helmet` assignments at confidence 0.25 / matching IoU 0.5. It detected only **43/179** annotated no-helmet objects correctly at that threshold. Helmet has **1,282 train objects in 613 images**, while no_helmet has **634 in 337 images**. Validation has 294 helmets in 146 images and 179 no_helmet objects in 91 images. Person has 1,962 objects in 791 training images and 481 in all 200 validation images. Every class's counts are in the JSON report.

After resizing the long side to 640, **354/634 training no_helmet objects** and **86/179 validation objects** have a short side below 16 pixels. Validation contains **57** below 8 pixels. These are geometric diagnostics from real source boxes.

Visual review covered source overlays, smoke errors and baseline crops/full-image context. Helmet normally marks protective headwear; Person describes a visible body; no_helmet normally marks a visible head without protective headwear. Exceptions remain:

- `source-38d248f0e2100035cc7c30a81392baa8.jpg`: a knit cap is labeled no_helmet and predicted helmet. Thus no_helmet does not guarantee literally uncovered scalp. Another predicted helmet appears to cover non-protective headwear and needs human adjudication.
- `source-485eb05877af0e242d34e297d390c7cd.jpg`: one no_helmet box appears to span a bent worker's orange torso rather than a head; a vest prediction matches that region. This is a suspected semantic/box error, not necessarily a head-classification failure.
- Small, partially visible and distant head crops often lack enough pixels for confident adjudication. A smoke FP example appears to contain an unattended real helmet, suggesting possible annotation incompleteness. Errors remain annotation-relative until reviewed.

Source labels were not edited. This is an agent visual spot-check, not an exhaustive human audit. The user's explicit authorization for the native 12-class baseline is recorded separately from review identity. Runtime `barehead` maps native no_helmet with the semantic caveat above. UNKNOWN remains necessary for insufficient evidence; absence of a helmet alone does not establish a violation.

### Phase 2: actual training

Activate the CUDA-enabled `.venv` first; `python` and `siteguard` must resolve there. Use the gitignored resolved source data YAML and validation report described in the earlier audit below.

For a fresh environment reproducing the actual baseline dependency profile
(requires working package-index access):

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,ml-cu128]' ultralytics==8.4.213
siteguard doctor
```

```powershell
$env:YOLO_CONFIG_DIR=Join-Path (Get-Location) 'runs/ultralytics-config'
$env:YOLO_AUTOINSTALL='false'
siteguard dataset validate --config configs/datasets.local.yaml
siteguard train --config configs/train-baseline.local.yaml
# A portable copy of the executed settings is configs/train-baseline.yaml.
# Choose a new experiment name for reproduction; existing runs are protected.
```

Training starts from the same `yolo26s.pt` initialization as the smoke, rather than continuing the smoke checkpoint. It retains **all 12 classes**, the official 800/200 split, **imgsz=640, batch=8, seed=42, workers=0, deterministic=true, FP32 / AMP=false**, and the same augmentations. Batch 8 was retained after measuring the smoke's 7,461 MiB whole-device peak on a 32 GiB RTX 5090. Effective optimizer, learning rate, mosaic/close-mosaic settings and all resolved arguments are in `args.yaml` and `experiment.json`.

Training exited **0**, completed **80 epochs**, and stopped at the epoch limit. Patience was **15**; the best epoch was **66**, so 14 subsequent epochs did not reach early stopping. Measurements:

- CLI wall time: **2,653.0086 seconds** (44m 13s), including setup/doctor/finalization.
- Runner training/final-validation time: **2,621.6203 seconds**; epoch CSV cumulative time: **2,612.4 seconds**.
- Actual trainer: **cuda:0**, PyTorch **2.8.0+cu128**, torchvision **0.23.0+cu128**, Ultralytics **8.4.213**.
- PyTorch peak allocated memory: **5,123,610,624 bytes**; peak reserved: **5,895,094,272 bytes**.
- 1,276 nvidia-smi samples: whole-device peak **7,691 MiB**, sampled peak utilization **95%**, mean **15.11%**. These readings include other processes and are not an isolated throughput benchmark.

`weights/best.pt` SHA256: `2ccf91ed1827aeb0fb5b35f4fec42d3127feddbe7b9d1f050be3a02426fc45c0`.

`weights/last.pt` SHA256: `a51acdc5b1ba9133684f888fc115a337add31586f34cb4aef31aa9cb6ecd1e81`.

The run preserves `results.csv`, `args.yaml`, `experiment.json`, loss/PR/confusion plots, training/validation examples and **80** per-epoch class records in `per-class-epochs.json`. The controlled re-evaluation's `metrics.json` is also copied into the training directory and identified in `experiment.json`. Ultralytics invokes its fit callback again after final best-checkpoint validation and temporarily increments the epoch. That result is retained separately in `final-best-validation.json`, not counted as an 81st training epoch. The callback was corrected accordingly.

### Identical checkpoint comparison

```powershell
python scripts/analyze_checkpoint.py --model runs/train/construction-ppe-12class-smoke-20261115/weights/best.pt --data data/local/construction-ppe.source.yaml --output runs/nohelmet-baseline/smoke-analysis
python scripts/analyze_checkpoint.py --model runs/train/construction-ppe-12class-baseline80-20261115/weights/best.pt --data data/local/construction-ppe.source.yaml --output runs/nohelmet-baseline/baseline-analysis
python scripts/compare_checkpoints.py --smoke runs/nohelmet-baseline/smoke-analysis --baseline runs/nohelmet-baseline/baseline-analysis --data data/local/construction-ppe.source.yaml --output runs/nohelmet-baseline/comparison.json
```

Analysis output directories must be new. Both evaluations use **imgsz=640, batch=8, device=0, workers=0, FP32, conf=0.001, iou=0.7 (native validator setting), max_det=300, seed=42, deterministic=true, split=val**. Data YAML hashes and settings match. The comparison rejects protocol mismatches. The diagnostic matrix uses **confidence=0.25 / geometric matching IoU=0.5**, a different operating point from Ultralytics' reported precision/recall. Native training-time final validation can differ slightly; the controlled re-evaluations below are the comparison of record.

Smoke -> baseline measured results:

- Overall precision **0.603741 -> 0.906384**, recall **0.557464 -> 0.762866**.
- Overall mAP50 **0.577715 -> 0.823290**, mAP50-95 **0.374450 -> 0.600881**.
- Helmet precision **0.815944 -> 0.943081**, recall **0.775510 -> 0.845357**, AP50 **0.821962 -> 0.897458**, AP50-95 **0.615761 -> 0.719606**.
- No_helmet precision **0.690276 -> 0.879578**, recall **0.223464 -> 0.659218**, AP50 **0.378585 -> 0.738653**, AP50-95 **0.214266 -> 0.456881**.

At fixed confidence 0.25 / IoU 0.5, helmet TP/FN/FP changed **231/63/57 -> 251/43/27**; no_helmet **43/136/22 -> 125/54/30**. Wrong no_helmet-as-helmet assignments fell **34 -> 6**. Reverse helmet-as-no_helmet assignments increased **0 -> 6**. No_helmet FP increased at this threshold; recall improvement does not alone prove fewer false safety alerts.

### Phase 3: one recommended next experiment

**Recommend YOLO26s at imgsz=960**, retaining the source annotations, split, initialization, seed, schedule and augmentations. Measure memory before choosing a feasible batch and record any batch change. Compare recall/AP and memory/latency on the same validation set; retain a separate independent video test for later. This experiment has **not** been started.

Baseline fixed-threshold no_helmet recall is **21/57 (0.3684)** below 8 pixels, **20/29 (0.6897)** at 8-16, **19/23 (0.8261)** at 16-32 and **65/70 (0.9286)** at 32+. Of 54 remaining misses/wrong classes, **36** are below 8 pixels. Higher resolution directly tests this observed failure mode. Upscaling cannot recover missing source detail; annotation noise remains unresolved.

### Phase 4: adapter readiness and video blockers

The best baseline checkpoint executed in `YoloDetector` on CUDA:0 with `Person->person`, `helmet->helmet`, `no_helmet->barehead`. Readiness was false before real inference and true afterward. Three actual validation images from the mixed-provenance dataset produced annotated images and `runs/nohelmet-baseline/adapter-readiness/readiness.json`; the predictor reported `cuda:0`, FP32, with actual mapped detections. This verifies checkpoint loading/mapping/execution, not video accuracy. The genuine saved-checkpoint GPU pytest passed (1 passed, 1 deselected).

No genuine construction MP4 or manually reviewed CCTV recording is present. The frontend FFmpeg test-pattern fixture and backend generated video are synthetic integration assets, not proof of genuine video inference. Real MP4 ByteTrack identity stability, overlapping-worker association, temporal precision/recall, duplicate alerts, detection delay, false alerts per camera-hour, end-to-end FPS/latency and frame drops remain **unverified**. A validated, gitignored `configs/runtime-baseline.local.yaml` now points at the real best checkpoint and explicit source-class map. After supplying a licensed construction MP4 and reviewed intervals, run:

```powershell
siteguard analyze --input '<real-construction.mp4>' --config configs/runtime-baseline.local.yaml --output runs/real-video-baseline
```

Then perform the documented event evaluation. No video results were invented.

### Final checks and publication

The post-training dataset validation exited **0** with `errors: []`, `leakage_pairs: 0`, `review_required: false`. This does not prove source independence or exhaustively correct semantics. Original image/label hashes remain bound to the audit; source annotations were not modified.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest -q --basetemp runs/nohelmet-baseline/thread-limited-final
python -m compileall -q src scripts
$env:SITEGUARD_GPU_MODEL=Join-Path (Get-Location) 'runs/train/construction-ppe-12class-baseline80-20261115/weights/best.pt'
$env:SITEGUARD_GPU_MAPPING='data/local/ppe-mapping.json'
python -m pytest tests/test_optional_ml.py -m gpu -q --basetemp runs/nohelmet-baseline/gpu-tests-new
```

Local core tests report **41 passed, 3 skipped**. Compilation passed. The genuine best-checkpoint GPU test passed separately with **exit 0** (1 passed, 1 deselected). CPU ByteTrack executes in the local suite. The two Windows HTTP skips run in Linux CI; GPU testing is opt-in. The new diagnostic unit test uses synthetic boxes only to verify the 12-class confusion bookkeeping, wrong assignments, duplicates and misses; it does not measure model accuracy.

This Windows environment intermittently terminates during shutdown with **3221225480 / 0xC0000008**, despite all assertions passing. Isolation reproduced this with combined video tests; each video test file separately exited 0. Limiting OpenCV's own threads did not resolve it. BLAS/OpenMP thread limits produced one exit-0 full run, but a repeat with JUnit still failed at shutdown, so this is **not a verified fix**. The unsuccessful thread-limit workaround was not retained in the check runner. Cached OpenCV 4.11 and FFmpeg thread/backend overrides also failed to resolve the shutdown fault; no package upgrade was retained. Production training/inference thread settings were not changed. Isolation results and JUnit/logs are retained in `runs/nohelmet-baseline/`. The Linux CI suite below completed cleanly.

Local Ruff/mypy remain unavailable because package-index DNS access fails; `scripts/check.py` honestly exits 1 for their missing modules. The same actual code passed both on the connected GitHub runner: **Ruff: all checks passed; mypy: no issues in 8 source files; pytest: 42 passed, 2 skipped**, plus the separate real CPU ByteTrack job. See [baseline CI run](https://github.com/saltuwwa/siteguard-vision/actions/runs/43746319490). Pydantic default declarations and reused optional coverage variables were corrected to satisfy type checking. No local lint success is inferred from absent tools.

The first version and genuine README detection photo are published at [saltuwwa/siteguard-vision](https://github.com/saltuwwa/siteguard-vision). The selected photo retains its honest one-epoch caption and CC BY 4.0 source attribution. The baseline report is public; raw data, weights, private source URLs and local configurations are excluded. No new infrastructure or frontend functionality was developed in this baseline task.

## Earlier one-epoch verification record

The sections below preserve the input audit, environment repair and one-epoch smoke that preceded the authorized full baseline. Smoke-only readiness statements refer to that earlier checkpoint.

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
