#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/install.sh cpu|cuda
mode="${1:-cpu}"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
case "$mode" in
  cpu) index=https://download.pytorch.org/whl/cpu ;;
  cuda) index=https://download.pytorch.org/whl/cu130 ;;
  *) echo 'Use cpu or cuda' >&2; exit 2 ;;
esac
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url "$index"
python -m pip install -e '.[dev,ml]'
siteguard doctor
