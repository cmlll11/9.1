#!/usr/bin/env bash

# Run the CIFAR-100 Probe transfer experiment.
#
# The Python script keeps the reference Clean models (seed3/4) separate from
# the target Clean models (seed0/1/2), and evaluates only same-seed backdoor
# checkpoints in the paired detection stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/hard_sample_gap}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1b_cifar100_transfer}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU_ID="${GPU_ID:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}/cifar100" ]]; then
    echo "ERROR: CIFAR-100 directory not found: ${DATA_ROOT}/cifar100" >&2
    exit 1
fi
if [[ ! -d "${BACKDOORBENCH_ROOT}" ]]; then
    echo "ERROR: BackdoorBench directory not found: ${BACKDOORBENCH_ROOT}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUTPUT_ROOT}"
LOG_PATH="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[$(date --iso-8601=seconds)] CIFAR-100 Probe transfer experiment" | tee -a "${LOG_PATH}"
echo "[$(date --iso-8601=seconds)] device=cuda:0 batch_size=${BATCH_SIZE}" | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/probe_cifar100_transfer.py" \
    --data-root "${DATA_ROOT}" \
    --model-root "${MODEL_ROOT}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --clean-group "${CLEAN_GROUP:-clean_select_shared}" \
    --reference-clean-seeds "3,4" \
    --target-clean-seeds "0,1,2" \
    --backdoor-groups "badnet,lf,blended,wanet" \
    --target 0 \
    --train-count 1000 \
    --test-count 1000 \
    --split-seed 2028 \
    --top-coarse 300 \
    --top-final 100 \
    --batch-size "${BATCH_SIZE}" \
    --device cuda:0 2>&1 | tee -a "${LOG_PATH}"

echo "[$(date --iso-8601=seconds)] launch log: ${LOG_PATH}" | tee -a "${LOG_PATH}"
