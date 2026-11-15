import json
import os
import signal
from pathlib import Path
from typing import Annotated

import typer

from siteguard.inference.device import doctor as inspect_runtime
from siteguard.config.settings import RuntimeConfig, load_config
from siteguard.training.datasets import (
    build_dataset,
    convert_shwd,
    download_dataset,
    validate_datasets,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


@app.callback()
def main() -> None:
    """Siteguard headless video analytics."""
    from siteguard.logging import configure_logging
    configure_logging()


@app.command()
def doctor() -> None:
    """Check dependencies, FFmpeg, CUDA architecture and actual tensor operations."""
    typer.echo(json.dumps(inspect_runtime(), indent=2))


@dataset_app.command("download")
def dataset_download(
    name: Annotated[str, typer.Option()],
    destination: Path = Path("data/raw"), archive: Path | None = None,
    version: str | None = None, license_id: str | None = None, accept_license: bool = False,
) -> None:
    """Acquire a public archive or explicitly authorized versioned manual export."""
    typer.echo(download_dataset(name, destination, archive, version, license_id, accept_license))


@dataset_app.command("validate")
def dataset_validate(config: Path = Path("configs/datasets.yaml")) -> None:
    report = validate_datasets(config)
    typer.echo(json.dumps({"errors": report["errors"], "leakage_pairs": len(report["leakage"]),
        "review_required": any(not s["reviewed"] for s in report["sources"])}))
    if report["errors"] or report["leakage"]:
        raise typer.Exit(1)


@dataset_app.command("report")
def dataset_report(config: Path = Path("configs/datasets.yaml")) -> None:
    """Regenerate the complete report and labeled review sample images."""
    report = validate_datasets(config)
    typer.echo(json.dumps({"sources": len(report["sources"]), "errors": report["errors"],
                           "independence_proven": report["independence_proven"]}))


@dataset_app.command("build")
def dataset_build(config: Path = Path("configs/datasets.yaml")) -> None:
    typer.echo(build_dataset(config))


@dataset_app.command("convert")
def dataset_convert(
    name: Annotated[str, typer.Option()], root: Annotated[Path, typer.Option()],
    output: Path = Path("data/converted/shwd"),
) -> None:
    if name != "shwd":
        raise typer.BadParameter("Only the explicit SHWD VOC adapter is supported")
    typer.echo(convert_shwd(root, output))


@app.command("analyze")
def analyze_file(
    input_path: Annotated[Path, typer.Option("--input")], output: Path = Path("runs/demo"),
    config: Path = Path("configs/runtime.yaml"), camera_id: str = "offline",
    model: str | None = None, device: str | None = None,
) -> None:
    """Process MP4 with a genuine PPE checkpoint; never substitutes fake predictions."""
    from siteguard.inference.yolo import YoloDetector
    from siteguard.pipeline import analyze

    settings = load_config(config, RuntimeConfig)
    if model:
        settings.model = model
    if device:
        settings.device = device
    typer.echo(json.dumps(analyze(str(input_path), output, camera_id, settings, YoloDetector(settings)),
                          indent=2))


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve authenticated programmatic endpoints (no UI)."""
    import uvicorn
    uvicorn.run("siteguard.api.app:create_app", factory=True, host=host, port=port)


@app.command("train")
def train(config: Path = Path("configs/train.yaml")) -> None:
    from siteguard.training.runner import run_training
    typer.echo(run_training(config))


@app.command("validate-model")
def validate_model(model: Annotated[Path, typer.Option()], data: Annotated[Path, typer.Option()],
    output: Path = Path("runs/validation"), split: str = "val", device: str = "auto",
    imgsz: int = 640) -> None:
    from siteguard.training.runner import validate_checkpoint
    typer.echo(json.dumps(validate_checkpoint(model, data, output, split, device, imgsz), indent=2))


@app.command("export")
def export(model: Annotated[Path, typer.Option()], format: str = "onnx", device: str = "cpu",
           half: bool = False, imgsz: int = 640) -> None:
    from siteguard.training.runner import export_checkpoint
    typer.echo(export_checkpoint(model, format, device, half, imgsz))


@app.command("evaluate")
def evaluate(predictions: Annotated[Path, typer.Option()],
    ground_truth: Annotated[Path, typer.Option()], identities: Annotated[Path, typer.Option()],
    output: Path = Path("runs/event-evaluation.json"), min_iou: float = .1,
    tolerance: float = .5, training_groups: Path | None = None) -> None:
    from siteguard.evaluation.events import evaluate_events
    groups = set(json.loads(training_groups.read_text())) if training_groups else None
    typer.echo(json.dumps(evaluate_events(predictions, ground_truth, output, identities,
                                         min_iou, tolerance, groups), indent=2))


@app.command("import-cvat")
def import_cvat(xml: Annotated[Path, typer.Option()], camera_id: Annotated[str, typer.Option()],
    source_id: Annotated[str, typer.Option()], fps: Annotated[float, typer.Option()],
    output: Path = Path("runs/cvat-import.json"), timestamp_offset: float = 0.) -> None:
    from siteguard.evaluation.cvat import import_cvat as convert
    convert(xml, output, camera_id, source_id, fps, timestamp_offset)
    typer.echo(output)


@app.command("frame-diagnostics")
def frame_diagnostics(data: Annotated[Path, typer.Option()],
    predictions: Annotated[Path, typer.Option()], output: Path = Path("runs/frame-errors"),
    split: str = "val") -> None:
    from siteguard.evaluation.objects import frame_diagnostics as diagnostics
    diagnostics(data, predictions, output, split)
    typer.echo(output)


@app.command("stream")
def stream(camera_id: Annotated[str, typer.Option()], source: str | None = None,
    output: Path = Path("runs/stream"), config: Path = Path("configs/runtime.yaml"),
    max_frames: int | None = None) -> None:
    """Analyze RTSP; use SITEGUARD_RTSP_SOURCE to keep credentials out of shell history."""
    from siteguard.inference.yolo import YoloDetector
    from siteguard.pipeline import analyze
    actual_source = source or os.getenv("SITEGUARD_RTSP_SOURCE")
    if not actual_source or not actual_source.lower().startswith(("rtsp://", "rtsps://")):
        raise typer.BadParameter("Provide RTSP source via environment or --source")
    settings = load_config(config, RuntimeConfig)
    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        typer.echo(json.dumps(analyze(actual_source, output, camera_id, settings,
            YoloDetector(settings), live=True, max_frames=max_frames), indent=2))
    except KeyboardInterrupt:
        typer.echo("Stream stopped; persisted events retained")


@app.command("benchmark")
def benchmark(input_path: Annotated[Path, typer.Option("--input")], model: Annotated[str, typer.Option()],
    config: Path = Path("configs/runtime.yaml"), output: Path = Path("runs/benchmark.json"),
    max_frames: int = 100, fp16: bool = True) -> None:
    from siteguard.evaluation.benchmark import benchmark as measure
    typer.echo(json.dumps(measure(input_path, model, output, load_config(config, RuntimeConfig),
                                 max_frames, fp16), indent=2))


@app.command("publish-outbox")
def publish_outbox(database_url: str | None = None, limit: int = 100,
    max_attempts: int = 5, replay_dead: bool = False) -> None:
    from siteguard.events.kafka import KafkaPublisher, drain_outbox
    from siteguard.storage.database import EventStore
    store = EventStore(database_url or os.getenv("SITEGUARD_DATABASE_URL", "sqlite:///runs/siteguard.sqlite"))
    if replay_dead:
        typer.echo(json.dumps({"requeued": store.replay_dead()}))
    bootstrap = os.getenv("SITEGUARD_KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        raise typer.BadParameter("Set SITEGUARD_KAFKA_BOOTSTRAP_SERVERS")
    configuration = json.loads(os.getenv("SITEGUARD_KAFKA_CONFIG_JSON", "{}"))
    result = drain_outbox(store, KafkaPublisher({"bootstrap.servers": bootstrap, **configuration}),
                          limit, max_attempts)
    typer.echo(json.dumps(result))
    if result.get("dead") or result.get("failed"):
        raise typer.Exit(1)


@app.command("cleanup")
def cleanup(retention_days: int = 30, evidence_root: Path = Path("runs"),
    database_url: str | None = None) -> None:
    from datetime import UTC, datetime, timedelta
    from siteguard.storage.database import EventStore
    if retention_days < 1:
        raise typer.BadParameter("Retention must be at least one day")
    store = EventStore(database_url or os.getenv("SITEGUARD_DATABASE_URL", "sqlite:///runs/siteguard.sqlite"))
    typer.echo(json.dumps({"deleted_events": store.cleanup(
        datetime.now(UTC)-timedelta(days=retention_days), evidence_root)}))


if __name__ == "__main__":
    app()
