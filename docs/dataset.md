# Dataset engineering and provenance

Still images train the detector. They cannot validate tracking, temporal persistence or false alerts per camera-hour. Do not infer video alert quality from object mAP.

## Verified source references

The [current Construction-PPE documentation](https://docs.ultralytics.com/datasets/detect/construction-ppe/) describes the November 2026 release: 1,000 images, 800 train/200 validation, 12 classes, CC BY 4.0, mixed human-collected/generated imagery with reviewed human/AI annotations. The [official archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip) is downloaded explicitly. Older descriptions refer to 1,416 images/11 classes. The tool reads the archive's actual YAML, hashes it, records class/split counts and warns when it differs from the new documented release. Neither documentation snapshot is silently substituted for archive inspection.

The current named concepts are `Person`, `helmet`, `no_helmet`; the last describes a visible uncovered head. Older `nohelmet` spellings require an explicit updated mapping. All other source classes must also appear in the config with `null` for deliberate exclusion. Ultralytics attribution and a link to CC BY 4.0 must accompany derived data. Preserve the archive license and metadata alongside the raw import.

[Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers) exposes about 7,000 source images with head/helmet/person classes; versioned exports can have different counts because of processing or augmentation. The page currently advertises Public Domain. Verify the selected export's license files and annotation semantics. In particular, neither `head` nor `person` is automatically treated as a body or a bare head. A sample review must resolve each class. [Roboflow's version export documentation](https://docs.roboflow.com/datasets/versions/dataset-versions/exporting-data) describes snapshots via ZIP, API or SDK.

The two supported acquisition methods are:

```bash
# After selecting a version, inspecting licensing, and manually exporting YOLO:
siteguard dataset download --name hard-hat-workers --archive /path/to/export.zip \
  --version 13 --license-id Public-Domain --accept-license
# Or explicitly authorized authenticated versioned export:
export ROBOFLOW_API_KEY='your-private-key'
siteguard dataset download --name hard-hat-workers --version 13 \
  --license-id Public-Domain --accept-license
```

The API branch requests a temporary versioned export link and does not persist its credentials. The manual ZIP method is the fallback if API access differs by account or export version. No authenticated download is attempted without the flags. Imports refuse to overwrite an existing manifest; use a new destination for a new version. A content-addressed archive hash supplies the default immutable version when no release label is given.

[SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) documents 7,581 Pascal VOC images. Its original `person` represents negative-head examples, not necessarily a human body. The converter names this `negative_head_person` and the helmet class `hat`. It never converts the original `person` into canonical `person`. Download requires manual access and license verification; a repository's software license is not proof of rights to every included image.

```bash
siteguard dataset download --name shwd --archive /path/to/SHWD.zip \
  --version original-voc --license-id YOUR-VERIFIED-LICENSE --accept-license
siteguard dataset convert --name shwd --root /path/to/VOC-root --output data/converted/shwd-v1
```

The converter expects `Annotations/`, `JPEGImages/` and reviewed `ImageSets/Main/train.txt`, `val.txt`, optionally `test.txt`. It preserves those partitions and converts 1-based inclusive VOC boxes. It rejects unknown object classes and invalid boxes. A converted dataset still needs an import manifest, semantic review and provenance before training.

[SH17](https://github.com/ahmadmughees/SH17dataset) has 8,099 images and 17 classes under CC BY-NC-SA 4.0. It is research only by default and has no built-in download route. Commercial builds allow only the configured verified permissive dataset licenses; SH17 cannot silently enter them. Obtain separate permission and perform a legal review before changing that policy.

## Canonical representation and approval

Canonical detector classes are `person` (visible body), `helmet` (visible protective helmet) and `barehead` (visible uncovered head). UNKNOWN is insufficient visual evidence, **not a detection class**. Missing helmets are not negative labels unless a head is visibly uncovered. Keep raw images and annotations immutable; human corrections belong in a new versioned derived source.

1. Download/import to `data/raw/`. `manifest.json` records original URL, license, version, UTC acquisition time, archive SHA256 and known real/generated provenance.
2. Inspect actual extracted paths, class mapping, README and license. Update `configs/datasets.yaml`; paths resolve from the repository working directory. YOLO directories are supported, not arbitrary executable YAML download hooks or list-file datasets.
3. Run `siteguard dataset validate --config configs/datasets.yaml`. The JSON report contains per-source/split/class image and box counts, hashes, invalid/missing label errors, orphan-label warnings and suspected duplicate pairs. Empty labels are accepted as explicitly annotated background; missing label files are errors. Every class must be accounted for by exact name.
4. Review labeled overlays in `runs/annotation-samples/` (up to 12 per split, deterministic). For difficult/rare classes inspect additional source images before approving. Record the real review in `data/reviews/source.json` using the report's fingerprint:

```json
{
  "reviewer": "actual-reviewer-name",
  "reviewed_at": "actual-UTC-ISO8601-time",
  "fingerprint": "copy-the-current-report-fingerprint",
  "semantics_verified": true,
  "canonical_annotations_complete": true,
  "sample_paths": ["paths-to-images-you-actually-reviewed"],
  "notes": "Explain visible-head and full-body semantics, exclusions, and any uncertainties"
}
```

This is a human attestation, not an automated proof that annotations are correct. Changing images, labels, YAML or mapping invalidates it. Do not copy a signed example or pretend automated labels have been reviewed.

Completeness applies to all visible canonical objects, not just the source's convenient classes. Head-only datasets may leave visible bodies unannotated. Mixing those images into ordinary fully supervised YOLO training would penalize correct person detections as background. Do not approve completeness in that situation: human-annotate missing canonical objects in a new derived version, exclude incomplete examples with documented curation, or implement and validate a separate partial-label training method. This repository does not pretend that ignored source classes solve missing-label supervision.

5. Add provenance JSON mapping image-relative path to known groups and origin:

```json
{
  "images/train/example.jpg": {
    "groups": ["video:siteA-shift01", "session:siteA-2026-11-01", "camera:siteA-cam01", "location:siteA"],
    "origin": "real"
  }
}
```

Use globally namespaced group values, not just `cam01`. The auditor rejects any shared group across splits, including camera/location groups where provided. Public source provenance is often insufficient: leave missing values unknown, never invent video/camera IDs from filenames. Setting `allow_unknown_provenance: true` is an explicit development-only acknowledgement. It remains in the report and disqualifies the resulting test split from independent-test claims.

6. Resolve leakage before `dataset build`. Exact file hashes and perceptual DCT hashes (64-bit, configurable Hamming distance) find cross-source and cross-split duplicates. Suspected cross-split pairs block build; within-split pairs remain reported. Perceptual hashing is heuristic and cannot prove independence. For false matches, review the pair and curate a new documented source/version rather than disabling detection silently. For real leaks quarantine a group in one partition, preserving raw official partitions and recording the derived change. Never randomly distribute adjacent extracted video frames across train/test.
7. Build into a new output directory. Canonical labels, a YOLO YAML, source report, evaluation fingerprint and content inventory are written. Training checks the inventory again and refuses modifications after build.

For stage B, enable the verified Hard Hat Workers source, resolve its mappings, include only its official train partition and set a new output such as `data/built/baseline-b-v1`. The default disabled config proposes no mappings for this source. Preserve its validation data separately for diagnostics; do not add it to the frozen A/B comparison set. Leakage auditing still reports all inspected source partitions conservatively.

## Independent evaluation

Prepare a source-disjoint camera/video set with rights to use the recordings. Use a separately curated still-image source with `role: independent` and only a `test` directory when an independent object test is available. Provide all source groups, review labels and audit cross-dataset pHashes. Include this exact test set in both A/B builds. Do not call Construction-PPE validation an unbiased final test. For event evaluation, manually reviewed video intervals and person states are mandatory; see [evaluation](evaluation.md).

## Native 12-class local smoke

For a source-preserving Construction-PPE baseline, the local source config sets
`label_schema: source`. A gitignored resolved data YAML points to the actual
dataset root and retains every original class ID. Native training uses
`data_schema: source` and an explicit validation report; all image/label hashes,
class names, split membership and review scope are checked before execution.
The semantic association map does not filter native training annotations.

An agent visual spot-check initially authorizes only
`training_scope: one_epoch_smoke_only`, with original label IDs preserved.
A native baseline additionally requires `training_scope: baseline` and either
an actual human review or explicit user authorization recorded as
`authorized_by_user: true`. The current 80-epoch experiment was explicitly
requested by the user after the smoke; that authorization does not turn the
agent spot-check into an independent human annotation audit. `stage: smoke`
still enforces exactly one epoch. Existing canonical remapping requires its
separate completeness review and cannot accidentally consume source-schema data.
See [the current verification record](verification.md) for the actual observed
counts, scoped review, baseline outputs and remaining model-quality limits.

The baseline error review found a non-protective knit cap labeled `no_helmet`
and a suspected `no_helmet` box covering a bent worker's torso. Native labels
were preserved. Thus this source class should be interpreted as the dataset's
visible no-protective-helmet category, with annotation noise, rather than a
guarantee of literal uncovered scalp. The runtime name `barehead` is an internal
mapping, not proof that every source annotation satisfies stricter semantics.
See the verification record for the exact image IDs and unresolved cases.

## Selected expansion candidate, 2026-11-17

[Hard Hat Workers v10 raw_AllClasses](https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/10)
is preferable for review to the3x augmented exports when seeking more distinct
source images. Its catalog reports7035 images (4916train,1413validation,706test),
auto-orient only, no augmentation, and Public Domain. These are **catalog claims**;
no corresponding archive has been downloaded, hashed or validated here. The
original source linked by Roboflow and the export README/license must agree
before a commercial build. Do not count augmentation copies as independent data.

Obtain the version10 YOLO export through Roboflow. After explicitly accepting
its actual license, import the local archive using the existing command:

```powershell
siteguard dataset download --name hard-hat-workers --archive data/incoming/hard-hat-workers-v10.zip --version 10 --license-id Public-Domain --accept-license
```

Authentication or manual export is an external prerequisite; no API key is
configured locally. Do not put keys in source files or chat. The source remains
disabled in the training configuration. Proposed head/person mappings remain
unset until labeled samples are reviewed. In particular, assess **full-body
annotation completeness**: adding head-only images to a joint12-class detector
as though missing body labels were negatives could harm worker recall. Do not
merge ambiguous person/head annotations by spelling alone. Keep all original
Construction-PPE classes and labels unchanged, frozen development validation,
cross-source pHash/leakage review and source-group reporting. Public partitions
without source provenance do not establish an independent CCTV test.

The Pexels construction video selected for a demo is excluded from training.
Its [Pexels License](https://www.pexels.com/license/) permits free use/modification
with conditions; it is not CC0. The download helper records provenance and waits
for content review; it does not create ground truth or claim the clip is CCTV.
