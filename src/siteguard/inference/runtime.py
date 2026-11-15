"""Version bounds are a compatibility guard; execution verifies capabilities."""
import importlib.metadata
import os
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

ULTRALYTICS_RANGE = ">=8.4.0,<8.5.0"


def ultralytics_version_status() -> dict[str, Any]:
    try:
        installed = importlib.metadata.version("ultralytics")
        supported = Version(installed) in SpecifierSet(ULTRALYTICS_RANGE)
        return {"version": installed, "supported_range": ULTRALYTICS_RANGE,
                "version_supported": supported, "execution_verified": False}
    except (importlib.metadata.PackageNotFoundError, InvalidVersion):
        return {"version": None, "supported_range": ULTRALYTICS_RANGE,
                "version_supported": False, "execution_verified": False}


def require_ultralytics() -> str:
    os.environ["YOLO_AUTOINSTALL"] = "false"
    status = ultralytics_version_status()
    if not status["version_supported"]:
        raise RuntimeError(f"Use ultralytics{ULTRALYTICS_RANGE}; found {status['version']}")
    return str(status["version"])
