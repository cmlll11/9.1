#!/usr/bin/env bash

# Shared paths and checks for all server experiments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/MDL/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
BACKDOORBENCH_ROOT="${REPO_ROOT}/third_party/BackdoorBench"
GAP_ROOT="${REPO_ROOT}/third_party/GAP"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/artifacts/models" "${REPO_ROOT}/artifacts/mappings" "${REPO_ROOT}/outputs" "${REPO_ROOT}/reports" "${DATA_ROOT}"
if [[ ! -e "${BACKDOORBENCH_ROOT}/data" ]]; then
    ln -s "${DATA_ROOT}" "${BACKDOORBENCH_ROOT}/data"
fi

log_step() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1"
}
