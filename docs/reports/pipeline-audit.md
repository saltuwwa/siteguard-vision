# SiteGuard P1 pipeline audit — 2026-11-16

Scope: existing ordered worker, installed Ultralytics 8.4.213, PyTorch 2.8.0+cu128,
two RTX 5090 devices. No frontend source or HTTP endpoint schema was changed.
The original 640px experiment remains protected by checkpoint/data hashes.
This audit does not establish production readiness or independent accuracy.

## High priority: fixed reliability defects

- **Stale detector readiness after an exception.** `YoloDetector.detect` previously
  set readiness on successful inference but left it true when a later prediction
  raised. Readiness now resets before every call. Device and actual backend FP16
  mode must match the validated request. Tests exercise success followed by
  failure and device mismatch; genuine model tests separately establish execution.
- **Failure erased the run summary.** In `pipeline.analyze`, metrics were written
  after a try/finally that re-raised inference failures. They were never reached.
  Final diagnostics/summary/metrics now persist during cleanup. Every close is
  attempted even if another close fails. The original processing exception is
  preserved; cleanup errors remain visible. A test confirms that an injected
  detector failure closes an already confirmed event and retains its evidence.
- **Timestamp regressions could be disguised.** `VideoReader._decode_session`
  treated backwards valid PTS as missing PTS and substituted frame-index timing.
  A regression now raises MediaError, causing a live reconnect or an explicit
  offline failure. Missing/duplicate timestamp fallback remains labeled. Pipeline
  session/reset guards and temporal rules also reject or isolate regressions.
- **Unchecked checkpoint integrity/type.** The detector now rejects unsupported
  formats, reports unreadable checkpoints explicitly, validates detection task,
  and optionally checks `model_sha256` before deserialization. PyTorch checkpoints
  must still come from a trusted source; a digest is integrity, not sandboxing.

## Medium priority: association and state corrections

- **Ambiguity contaminated unrelated candidates.** `associate_heads` previously
  marked every geometric candidate ambiguous when only the top two were close.
  It now marks candidates within the actual ambiguity margin. A three-worker
  test verifies the distant third worker retains its own unambiguous helmet.
- **Two distinct heads could become one worker's evidence.** Multiple same-class
  heads were collapsed to the highest confidence. Spatially distinct heads now
  produce UNKNOWN (`multiple_distinct_heads`); overlapping duplicate predictions
  select one head with explicit duplicate rejection. Helmet/bare-head conflict
  remains UNKNOWN. No confidence threshold was blindly lowered.
- **Confirmed event and displayed track state disagreed.** `ViolationEngine.tick`
  previously changed an absent track to UNKNOWN while its confirmed event was
  still open. It now retains CONFIRMED until closure; current visible head state
  is displayed separately as UNKNOWN. End time remains the last supporting bare
  head, rather than the time at which absence was finally noticed.
- **A bare head during cooldown could retain a COMPLIANT/CLEARED display.**
  It now becomes SUSPECT with an explicit `cooldown_active` decision while the
  existing suppression deadline still prevents confirmation. A missing worker
  also resets an in-progress visible-helmet clearance interval, so absence
  cannot complete a clearance. Dedicated regressions cover both cases.
- **Automatic CUDA selection required every GPU to be healthy.** A valid GPU was
  ignored if another failed the global doctor result. Auto selection now chooses
  a compatible reported device. Explicit incompatible device/CPU-FP16 requests
  still fail. The hardware doctor remains an execution check, not `is_available`.
- **Unbounded event-ID history / latency shifting.** Runtime deduplication now
  retains only open incidents and a total confirmation count. Timing percentiles
  use bounded deques rather than shifting a large list on each long-running frame.

## Measurable observability and output changes

`diagnostics.json` records post-detector person/helmet/barehead counts, fixed-bin
confidence histograms, track observations, attempted/accepted/rejected head
associations, actual rejection reasons, unknown/suspect observations, confirmed,
cleared/ended/expired events and reset reasons. Counts are observations across
frames unless explicitly named events; they are not unique-person estimates.
The existing frame JSONL includes compact frame counters. No extra image stream,
unbounded diagnostic history or broker transport was introduced.

Main-thread detector/ByteTrack/association/temporal/rendering/enqueue timings are
separate from decoder and durable event I/O timings. Each timing retains 2,048
recent samples, total/count/max and p50/p95; thread totals overlap. Decode-to-output
latency excludes camera/network time and asynchronous database commit. GPU
allocator peaks are process-local and explicitly scoped. Cold inference is
included. Throughput on a blank synthetic scene is not crowded-site throughput.

The annotated MP4 separates thin detection corners (`det:`) from tracked-body
rectangles. Track labels show worker ID, current helmet/barehead/unknown evidence
and temporal state. SUSPECT and automated CONFIRMED have distinct styles. Missing
head evidence does not itself create a violation. Odd dimensions are padded
instead of needlessly resampling the entire image; changing dimensions are
letterboxed. The existing three-argument output adapter interface and
`track:<id>:<state>` labels used by the separate frontend sidecar are preserved.

## Small-object findings and controlled experiment

The 12-pixel association gate is measured in the **decoded frame**, not the 640px
model tensor. At decoded width 1280, 12 pixels correspond approximately to six
model pixels. It is incorrect to equate this gate directly with the baseline's
under-8px model-input size bin. Detector threshold .15 and rule/person threshold
.6 serve different purposes; ByteTrack has its own .4/.5 initiation thresholds.
Low-score candidates can therefore reach tracking but remain UNKNOWN for alerts.

The installed predictor explicitly selects its detection head with
`end2end=self.args.nms is False`. The actual saved 640px model ran with
`nms=None`, `end2end_backend=False`, and FP32 (`quantize=32`) on both CUDA and CPU.
Thus this verified path uses the one-to-many/NMS branch, despite the YOLO26 family
supporting an NMS-free head. No extra project NMS is applied and no branch switch
was silently introduced. Resolution comparison must preserve this behavior.
Current Ultralytics deprecates `half` in favor of `quantize`; the adapter uses
the supported field when available, avoiding per-frame warning spam, and checks
actual backend precision. Three genuine validation-image prediction counts
matched the earlier adapter record; an actual CPU image also executed successfully.

VideoReader defaults to width 1280 for compatibility; `max_width: null` now
preserves native source detail for controlled tests. Shrinking a wide image to
1280 before YOLO resizes it again can discard high-frequency detail. Downscaling
retains the existing linear interpolation: the paired genuine-inference ablation
found no no_helmet benefit from changing it to area interpolation. ROI/tiling would add duplicate reconciliation, camera
calibration and latency; it is not justified before measuring the simpler 960px
experiment. Higher resolution cannot recover detail absent in the original video.

`scripts/audit_association.py` reprojects saved genuine 640px validation detections
to decoded-size scenarios. It reports rejected classes/reasons without pretending
per-image detector boxes are verified tracked identities. This isolates gating
sensitivity; it does not simulate a new detector prediction after resizing.

A genuine three-batch forward/backward probe at 960px/batch8/CUDA:1 measured
9,560,156,160 allocated and 11,377,049,600 reserved bytes. Batch8 was retained.
The new unique 960px experiment uses the same initialization, native 12 classes,
split, seed, augmentations, FP32, 80-epoch limit and patience15. Physical GPU and
resolution are explicit differences. Final comparison must retain identical
evaluation settings except resolution and use the same source-size bins anchored
to 640px. The original best/last checkpoints must remain byte-identical.

## Unresolved limitations and risks

- No verified independent source-disjoint still/video test exists. Construction-PPE
  is mixed-provenance development data and contains suspected semantic/box noise.
  A native `no_helmet` knit cap is not necessarily literally uncovered scalp.
- The geometric head window assumes the head lies in the upper portion of a body
  box. Bent workers, top-down views, severe occlusion and crowded boxes remain
  difficult. Accepted associations are not ground-truth association accuracy.
- ByteTrack is genuine, person-only and stream/session scoped. Its Kalman time is
  frame based; irregular live drops can affect motion estimates. Source-time gaps
  reset identities, but no cross-track re-identification or cross-ID incident
  deduplication is claimed. Histogram camera cuts can miss similar scenes or
  overreact to motion/lighting; no threshold change was justified by labeled video.
- Original private MP4 uploads found locally are QA patterns or a stationary photo.
  Private HLS camera fragments show an empty indoor office. They can exercise real
  decoder/model plumbing but cannot validate construction-worker tracking or alerts.
  Suitable construction footage plus reviewed temporal labels remain external
  dependencies. Do not turn the office recording into production accuracy claims.
- Windows pytest's native shutdown 0xC0000008 was reproduced with FD-level capture.
  Short video tests exited 0 with sys/no capture; three full sys-capture runs
  each passed 70 tests/3 skips and exited 0. The pytest configuration now uses
  sys capture, retaining assertions and visible native stderr. This is a verified
  test-launch mitigation, not a claim to have repaired the native library's exact
  internal cause. Failed FD-capture runs and the unrelated initial probe nonzero
  exit remain recorded; no dependency upgrade was retained.
- Startup failures before the worker loop and hard process termination can leave
  incomplete run files. Database evidence/JSONL are not one atomic transaction;
  a filesystem/SQL failure remains explicit, and the database is authoritative.
  Event snapshots still commit per supporting frame; optimize only if measurements
  show persistence is a bottleneck. Detector batching adds queue latency and is
  deferred until a throughput bottleneck is demonstrated.
