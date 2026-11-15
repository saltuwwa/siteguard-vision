"""Save exact installed dependencies after a successful platform-specific install.

Keep CPU and GPU freezes separate. This captures transitive versions; it is not
a cross-platform solver lock. Do not freeze an unrelated system interpreter.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate a clean project virtualenv before freezing")
    result = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
                            check=True, capture_output=True, text=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
