# Controlled experiments and honest evaluation

No detector accuracy or event performance is bundled. Unit-test labels are synthetic software fixtures, not measurements of model quality. Still-image validation cannot establish camera-hour false alert rates.

## Experimental stages

Stage A: Construction-PPE only, `yolo26s.pt` fine-tuning using `configs/train.yaml`. Stage B: same model and training settings, adding only reviewed Hard Hat Workers training annotations (`configs/train-b.yaml`). Freeze the official Construction-PPE validation set and any independent object test set before both runs. Compare the `evaluation_fingerprint` values in build reports/experiments; they must be identical. Do not select hyperparameters on an independent test set.

Stage C: select the model using development validation, freeze it, and evaluate it on independent CCTV-style recordings with tracking and temporal rules enabled. Record camera viewpoints, visibility, site provenance, class definitions and temporal settings. YOLO26m (`configs/train-m.yaml`) is a later comparison on the same evaluation sets; larger does not imply better small-head performance or acceptable latency.

Every experiment saves data and checkpoint hashes, seed, augmentations, batch/image size, precision, device/hardware, resolved arguments, CSV metrics and best/last checkpoint references. Archive the actual environment freeze and datasets/manifests with each run. Deterministic settings improve repeatability but do not promise bitwise equality across CUDA/compiler/hardware changes. Changing data or configuration creates a new run name. Resume uses the training arguments restored by Ultralytics and records them afterward.

## Object evaluation

`siteguard validate-model` runs genuine Ultralytics validation on labeled built data. It writes precision, recall, mAP50, mAP50-95, class-wise metrics, confusion matrix, saved predicted YOLO labels and native validation plots. `val` reports explicitly identify development validation. `test` requires a built independent split with known provenance.

```bash
siteguard validate-model --model runs/train/stage-a-seed42/weights/best.pt \
  --data data/built/baseline-a-v1/data.yaml --split val --output runs/val-a
siteguard frame-diagnostics --data data/built/baseline-a-v1/data.yaml \
  --predictions runs/val-a/labels --output runs/errors-a --split val
```

Diagnostics use a configurable confidence/IoU operating point (Python API defaults .25/.5), one-to-one geometric assignment and a background row/column. Green boxes are truth, red boxes are predictions; wrong-class, FP and FN records are saved with examples. Missing prediction files count as no detections, so supply a complete successful validation output. These diagnostics are not a replacement for confidence-swept AP.

## Video annotation contract

`examples/video-ground-truth.schema.json` is the generated full Pydantic JSON schema. A labeled manifest must declare `label_origin: human_reviewed`. Each recording supplies camera/source ID, source groups, timestamp basis, reviewer/time and reviewed time intervals. Incidents have unique IDs, a human person ID, source/camera and a positive-duration violation interval. Per-person observations record source timestamp, visible helmet/barehead/unknown, occlusion, visibility and optionally a body box. UNKNOWN intervals must not be fabricated into violations.

Record source IDs as immutable recording hashes or stable capture IDs, including for saved RTSP recordings. Source-relative seconds must match inference timestamps. Store actual decoder PTS for VFR; do not assume frame/FPS timing for VFR recordings. A live receive-monotonic run needs a saved timestamp sidecar to align its recordings and annotations; recording transport itself is not implemented by this tool.

For CVAT video XML, annotate `person` tracks with a `helmet_state` attribute (`helmet`, `barehead`, `unknown`). Then import a constant-FPS recording:

```bash
siteguard import-cvat --xml /path/to/annotations.xml --camera-id cam01 \
  --source-id recording-sha256 --fps 25 --output runs/cvat-import.json
```

The importer preserves track IDs and boxes, maps occluded/outside boxes to UNKNOWN and marks the file `cvat_import_requires_review`. It does **not** generate ground-truth incidents. Review interpolation, coverage, timestamps and incident boundaries, add real recording provenance/reviewer metadata, and convert it to the schema before evaluation. Automatic pseudo-labels are proposals, not truth.

## Event matching

Analysis JSONL contains event lifecycle snapshots; the evaluator uses the last snapshot per event ID and refuses open event intervals. Track IDs from a detector are not the same identities as human annotation IDs. Supply a manually verified correspondence file:

```json
{
  "cam01/predicted-stream-session-id/7": {"source_id": "recording-sha256", "person_id": "human-person-3"}
}
```

Mappings are mandatory even for unmatched predicted people, so false alerts cannot be silently removed. Use a distinct reviewed person ID for a spurious track with no real corresponding person. Evaluating an event from an undeclared source is an error.

```bash
siteguard evaluate --predictions runs/demo/events.jsonl --ground-truth /path/to/reviewed-video.json \
  --identities /path/to/track-correspondence.json --output runs/video-report.json \
  --training-groups /path/to/training-source-groups.json --min-iou 0.1 --tolerance 0.5
```

`--training-groups` is a JSON list of real globally named provenance groups. The manifest's `training_source_groups` is also checked. Declared overlap with any evaluated source group fails. No algorithm can discover unrecorded provenance: the report states that independence rests on the supplied human provenance.

Matching requires the same camera, recording and reviewed person identity. Ground-truth boundaries are padded by the configured tolerance; a pair is eligible when its temporal intersection-over-union exceeds `min_iou`. Global assignment maximizes one-to-one matches and overlap. Unmatched predictions that overlap an already matched incident count as duplicates and false alerts. Other unmatched predictions are false alerts; unmatched incidents are misses. An event must lie wholly within a reviewed coverage interval or it is explicitly excluded and counted in the report. Review continuous coverage for deployment-quality rates; clipping/excluding boundary events can bias partial-coverage evaluations.

Metrics: event precision/recall; unmatched alerts divided by reviewed camera-hours; missed incident fraction; duplicate count divided by total evaluated predictions; signed delay from true onset to **confirmation time**, p50/p95. Empty denominators yield null, not made-up zeros. Tolerance can yield a negative signed delay; inspect those matches. Overlapping reviewed intervals are rejected to prevent denominator double counting. Different recording files must represent non-overlapping footage for camera-hour aggregation; relative timestamps cannot independently prove that.

## Runtime performance

`summary.json` reports processed/source FPS, dropped sampled-frame fraction and p50/p95 decode-to-output latency. Decode-to-output latency excludes camera/network delay and asynchronous event commit; use external synchronized capture timestamps and DB commit tracing for true capture-to-durable-alert latency. Frame JSONL retains decode UTC and processing Unix time for analysis. `benchmark` separately warms up and times real detector calls, compares FP32/FP16 only when supported, and samples nvidia-smi memory/utilization. It reports no accuracy. Never mix detector-only latency with full-pipeline latency claims.
