# Linux/WSL2, GPU and RTSP deployment

Use a dedicated Python 3.11 environment, a supported NVIDIA driver, and FFmpeg. Follow [NVIDIA's WSL guidance](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for WSL2; the Windows host supplies the GPU driver. Do not install a conflicting Linux display driver inside WSL. Follow [official PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/) when choosing stable CUDA/PyTorch/torchvision pairs.

The repository pins torch 2.14.1 / torchvision 0.29.1. Use the `cu130` index for RTX 5090, or the CPU index for CI. Older stable torch 2.8.0 / torchvision 0.23.0 with cu128 is another officially documented Blackwell-capable pair, but changing the repository pin requires revalidation. Never mix a cached torch wheel with an unrelated torchvision version for inference.

`siteguard doctor` must report compiled `sm_120`, actual CUDA FP32 matrix operations, FP16 operations and available GPU memory. `is_available()` alone is insufficient. A successful tensor check does not validate YOLO, torchvision kernels, a checkpoint, ONNX or TensorRT. Run the optional genuine model smoke test and a labeled MP4 analysis before deployment. CPU fallback is suitable for testing and must be benchmarked before any live-service promise.

## Containers and local services

```bash
docker build -f docker/Dockerfile --build-arg INSTALL_ML=true \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 -t siteguard-worker .
# Mount your model, configs, footage and a writable private run directory.
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/runs:/app/runs" -v "$PWD/models:/app/models:ro" \
  -v "$PWD/examples:/app/examples:ro" siteguard-worker \
  siteguard analyze --input examples/site.mp4 --output runs/container-run \
  --model models/ppe-best.pt
```

The Dockerfile defaults to a CPU/API-only install; it runs as UID 10001. Arrange bind-mount ownership or explicitly select your user. NVIDIA container runtime installation is a host prerequisite. Container builds have not been exercised in the restricted development environment.

```bash
docker compose -f docker/compose.yaml up -d api
docker compose -f docker/compose.yaml --profile kafka up -d kafka
# Set a private password first; none is bundled.
export SITEGUARD_POSTGRES_PASSWORD='your-private-password'
docker compose -f docker/compose.yaml --profile postgres up -d postgres
```

API and infrastructure ports bind localhost. Local Kafka PLAINTEXT and a single broker are for development. Use SASL/TLS, access controls and a suitable replication policy externally; pass private producer settings using `SITEGUARD_KAFKA_CONFIG_JSON`, never committed files. `SITEGUARD_DATABASE_URL` selects PostgreSQL and overrides runtime YAML for workers as well as the API. Publisher/cleanup also accept an explicit database URL; prefer the environment to avoid process-list exposure. Back up metadata and evidence together.

## RTSP operation

Put a private URL in `SITEGUARD_RTSP_SOURCE`. Avoid `--source` for secret URLs because the operating system may expose process arguments. Application exceptions suppress URLs, and capture logging is reduced. Do not enable verbose backend/FFmpeg diagnostics on secret-bearing URLs without redaction.

Configure transport (`tcp` or `udp`), open/read timeout, retry count/backoff, inference FPS, maximum width, queue size/drop policy, confidence and temporal rules in runtime YAML. Default live policy is drop-oldest for latency; offline analysis always blocks the frame queue. Source/session reset metrics and timestamp fallbacks are auditable. ByteTrack TTL is scaled to the configured inference rate; long timestamp gaps additionally force resets. Low inference FPS or long network gaps can prevent temporal confirmation by design; tune `max_gap` against measured sampling intervals.

One process owns each camera and tracker. Use a process supervisor that restarts failed workers into **new run directories**. Ctrl-C/SIGTERM closes current events, drains event I/O, releases encoders and attempts bounded decoder cleanup. Hard kills require reconciliation of open incidents. Do not run several threads with different FFmpeg transport settings inside one process.

For temporal evaluation, capture authorized RTSP footage externally with its timestamp sidecar, then analyze the recording offline. OpenCV capture does not guarantee original RTP/NTP camera-clock extraction, nor does this repository provide an RTSP recorder service. When camera-origin timestamps are mandatory, implement a GStreamer/DeepStream decoder that preserves them and reuse the frame contract.

## Export and benchmarking

ONNX export is available with `pip install -e '.[export]'`. TensorRT is optional and requires explicit `--format engine`, validated CUDA, an installed compatible runtime and version checks. Consult [NVIDIA TensorRT release notes/support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html). The minimum version guard is not proof of compatibility; successfully build and execute an engine on the target GPU, verify predictions against the reference checkpoint and benchmark it before use. Engine portability is not assumed. TensorRT inference through the production adapter is not validated here.

## Retention and security

Set an API token before using protected routes. An unset token returns 503 instead of exposing camera/event information. `/health` reveals only liveness. There is no public file server. Use filesystem permissions, a private network/TLS boundary and a reverse proxy if remote access is needed. The code never logs environment credentials or raw RTSP URLs intentionally.

Event cleanup uses closed event creation age and protects pending/retry/dead outbox records. Schedule it deliberately. It removes referenced images/videos, not arbitrary run logs or orphan images. Retain frame logs and annotated footage according to your privacy and audit requirements. Their volume can dominate event storage. No production camera footage, model weights, credentials or datasets belong in git.
