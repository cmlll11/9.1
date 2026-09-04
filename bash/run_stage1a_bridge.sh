#!/usr/bin/env bash

# Launch the core Stage 1A experiment on a server.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/hard_sample_gap}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1a_bridge}"

log_step "Stage 1A: coarse targeted PGD, refine Top-40, evaluate Top-30"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/oracle_bridge_pilot.py" \
    --data-root "${DATA_ROOT}" \
    --model-root "${MODEL_ROOT}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --clean-group "${CLEAN_GROUP:-clean_select_shared}" \
    --clean-seeds "0,1" \
    --backdoor-groups "blended,wanet" \
    --target 0 \
    --candidate-count 200 \
    --candidate-seed 2026 \
    --top-coarse 40 \
    --top-final 30 \
    --batch-size "${BATCH_SIZE:-32}" \
    --device cuda:0
