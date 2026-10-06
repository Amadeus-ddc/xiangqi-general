#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -m xqgeneral.train --config configs/pilot.json > runs/pilot.log 2>&1
