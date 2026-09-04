#!/usr/bin/env bash

# Launch the core Stage 1B Probe experiment.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/hard_sample_gap}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1b_probe}"

log_step "Stage 1B: within-model, leave-one-clean-model-out and OOD Probe"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/probe_loco_pilot.py" \
    --data-root "${DATA_ROOT}" \
    --partition-root "${DATA_ROOT}/hard_sample_gap" \
    --model-root "${MODEL_ROOT}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --stage1a-candidates "${STAGE1A_RUN_DIR}/candidate_pool.csv" \
    --output-root "${OUTPUT_ROOT}" \
    --clean-group "${CLEAN_GROUP:-clean_select_shared}" \
    --clean-seeds "0,1,2" \
    --backdoor-groups "blended,wanet" \
    --target 0 \
    --source-count 300 \
    --validation-count 100 \
    --ood-count 200 \
    --source-seed 2026 \
    --ood-seed 2027 \
    --batch-size "${BATCH_SIZE:-32}" \
    --device cuda:0
