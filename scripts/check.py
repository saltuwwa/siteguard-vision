"""Run checks with the active interpreter; missing check tools are failures."""
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # The shared interpreter may expose unrelated pytest plugins. Keep this
    # repository's checks independent of their hooks/native shutdown handlers.
    environment = {**os.environ, "PYTHONPATH": str(root/"src"),
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    (root/"runs").mkdir(exist_ok=True)
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts", "examples"],
        [sys.executable, "-m", "mypy"],
        [sys.executable, "-m", "pytest", "-q", "--basetemp", f"runs/tests-{uuid4().hex[:8]}"],
        [sys.executable, "-m", "compileall", "-q", "src", "scripts"],
    ]
    failed = False
    for command in commands:
        print("Running", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=root, env=environment)
        print(f"Exit code: {result.returncode}", flush=True)
        failed |= result.returncode != 0
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
