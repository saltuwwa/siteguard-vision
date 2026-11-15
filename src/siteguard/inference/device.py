import importlib.metadata
import shutil
from typing import Any

from siteguard.inference.runtime import ultralytics_version_status


def doctor() -> dict[str, Any]:
    report: dict[str, Any] = {"ffmpeg": shutil.which("ffmpeg"), "gpus": [], "cuda_ok": False}
    report["dependencies"] = {}
    for name in ("siteguard-vision", "ultralytics", "torch", "torchvision", "opencv-python",
                 "opencv-python-headless", "lap"):
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    try:
        import torch

        report["torch_runtime_version"] = str(torch.__version__)
        torch_pair = str(torch.__version__).split("+")[0].split(".")[:2]
        vision = report["dependencies"]["torchvision"]
        expected_vision = f"0.{int(torch_pair[1])+15}." if torch_pair[0] == "2" else None
        report["torchvision_version_pair_ok"] = bool(
            vision and expected_vision and vision.startswith(expected_vision))
        report["torch_cuda"] = torch.version.cuda
        report["compiled_architectures"] = torch.cuda.get_arch_list()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(index)
                arch = f"sm_{capability[0]}{capability[1]}"
                supported = arch in report["compiled_architectures"]
                with torch.cuda.device(index):
                    x = torch.randn((128, 128), device=f"cuda:{index}")
                    y = x @ x.T
                    torch.cuda.synchronize(index)
                    fp32 = bool(torch.isfinite(y).all().item())
                    h = x.half() @ x.half().T
                    torch.cuda.synchronize(index)
                    fp16 = bool(torch.isfinite(h).all().item())
                    # YOLO exercises convolution/cuDNN as well as GEMM. Verify
                    # actual torchvision kernels, not only distribution versions.
                    with torch.inference_mode():
                        convolution = torch.nn.Conv2d(3, 8, 3, padding=1).to(f"cuda:{index}")
                        pixels = torch.randn((1, 3, 32, 32), device=f"cuda:{index}")
                        conv32 = bool(torch.isfinite(convolution(pixels)).all().item())
                        conv16 = bool(torch.isfinite(convolution.half()(pixels.half())).all().item())
                    torchvision_ok = False
                    torchvision_error = None
                    try:
                        from torchvision.ops import nms

                        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.],
                                              [20., 20., 30., 30.]], device=f"cuda:{index}")
                        scores = torch.tensor([.9, .8, .7], device=f"cuda:{index}")
                        torchvision_ok = nms(boxes, scores, .5).cpu().tolist() == [0, 2]
                    except Exception as exc:
                        torchvision_error = f"{type(exc).__name__}: {exc}"
                    torch.cuda.synchronize(index)
                    free, total = torch.cuda.mem_get_info(index)
                blackwell_cuda_ok = capability[0] < 12 or (
                    torch.version.cuda is not None
                    and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8)
                )
                report["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                    "capability": list(capability), "architecture_supported": supported,
                    "memory_free": free, "memory_total": total, "fp32_ok": fp32,
                    "fp16_ok": fp16 and conv16, "convolution_fp32_ok": conv32,
                    "convolution_fp16_ok": conv16, "torchvision_cuda_nms_ok": torchvision_ok,
                    "torchvision_cuda_error": torchvision_error,
                    "compatible": supported and fp32 and conv32 and blackwell_cuda_ok
                                  and torchvision_ok and report["torchvision_version_pair_ok"]})
            report["cuda_ok"] = bool(report["gpus"]) and all(
                gpu["compatible"] for gpu in report["gpus"])
        report["ultralytics"] = ultralytics_version_status()
        report["ppe_inference_dependencies_present"] = (
            report["ultralytics"]["version_supported"] and report["dependencies"]["lap"] is not None)
        # Model execution is tested by training/inference, not inferred from a version.
        report["ppe_inference_ready"] = None
        report["ppe_inference_readiness_note"] = "Not tested by hardware doctor; YoloDetector becomes ready after actual model execution"
    except Exception as exc:
        report["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return report


def select_device(requested: str, half: bool = False) -> tuple[str, bool]:
    if requested == "cpu":
        if half:
            raise ValueError("FP16 requires a validated CUDA device")
        return "cpu", False
    report = doctor()
    if requested == "auto":
        return ("0", half and report["gpus"][0]["fp16_ok"]) if report["cuda_ok"] else ("cpu", False)
    index = int(requested.removeprefix("cuda:"))
    gpus = report["gpus"]
    if index < 0 or index >= len(gpus) or not gpus[index]["compatible"]:
        raise RuntimeError("Requested CUDA device failed doctor; use --device cpu or repair PyTorch")
    if half and not gpus[index]["fp16_ok"]:
        raise RuntimeError("FP16 validation failed")
    return str(index), half
