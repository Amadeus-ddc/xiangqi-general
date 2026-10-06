#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -m pytest -q tests > runs/tests.log 2>&1
.venv/bin/python -m xqgeneral.data --games 80 > runs/data-generation.log 2>&1
.venv/bin/python -m xqgeneral.cache_features > runs/feature-cache.log 2>&1
