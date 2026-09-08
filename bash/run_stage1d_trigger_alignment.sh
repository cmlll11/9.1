#!/usr/bin/env bash

# Run Stage 1D: targeted-robust sample selection and trigger-direction analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/hard_sample_gap}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_trigger_alignment}"
QUALITY_REPORT="${QUALITY_REPORT:-}"
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
TRIGGER_PATH="${TRIGGER_PATH:-${BACKDOORBENCH_ROOT}/resource/badnet/trigger_image.png}"
LF_TRIGGER_PATH="${LF_TRIGGER_PATH:-${BACKDOORBENCH_ROOT}/resource/lowFrequency/cifar10_preactresnet18_0_255.npy}"
BLENDED_TRIGGER_PATH="${BLENDED_TRIGGER_PATH:-${BACKDOORBENCH_ROOT}/resource/blended/hello_kitty.jpeg}"
WANET_STATE_PATH="${WANET_STATE_PATH:-${MODEL_ROOT}/wanet/seed0/state_dict.pt}"
for trigger_file in "${TRIGGER_PATH}" "${LF_TRIGGER_PATH}" "${BLENDED_TRIGGER_PATH}" "${WANET_STATE_PATH}"; do
    if [[ ! -f "${trigger_file}" ]]; then
        echo "ERROR: required trigger/state file not found: ${trigger_file}" >&2
        exit 1
    fi
done

IFS=',' read -r -a BACKDOOR_GROUP_ARRAY <<< "${BACKDOOR_GROUPS:-badnet,lf,blended,wanet}"
for group in "${CLEAN_GROUP:-clean_select_shared}" "${BACKDOOR_GROUP_ARRAY[@]}"; do
    for seed in 0; do
        checkpoint="${MODEL_ROOT}/${group}/seed${seed}/attack_result.pt"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "ERROR: checkpoint not found: ${checkpoint}" >&2
            exit 1
        fi
    done
done

if ! "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "ERROR: CUDA is not available in the selected Python environment." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
QUALITY_ARGS=()
if [[ -n "${QUALITY_REPORT}" ]]; then
    QUALITY_ARGS+=(--quality-report "${QUALITY_REPORT}")
fi

{
    echo "[$(date --iso-8601=seconds)] Stage 1D targeted-robust trigger alignment"
    echo "[$(date --iso-8601=seconds)] device=cuda:0 batch_size=${BATCH_SIZE}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trigger_alignment_mechanism.py" \
        --data-root "${DATA_ROOT}" \
        --model-root "${MODEL_ROOT}" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --trigger-path "${TRIGGER_PATH}" \
        --lf-trigger-path "${LF_TRIGGER_PATH}" \
        --blended-trigger-path "${BLENDED_TRIGGER_PATH}" \
        --wanet-state-path "${WANET_STATE_PATH}" \
        --output-root "${OUTPUT_ROOT}" \
        --clean-group "${CLEAN_GROUP:-clean_select_shared}" \
        --backdoor-groups "${BACKDOOR_GROUPS:-badnet,lf,blended,wanet}" \
        --clean-seeds "0" \
        --candidate-count "${CANDIDATE_COUNT:-1000}" \
        --candidate-seed "${CANDIDATE_SEED:-2031}" \
        --top-coarse "${TOP_COARSE:-300}" \
        --top-final "${TOP_FINAL:-100}" \
        --batch-size "${BATCH_SIZE}" \
        "${QUALITY_ARGS[@]}" \
        --device cuda:0
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
