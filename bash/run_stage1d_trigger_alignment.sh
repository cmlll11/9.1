#!/usr/bin/env bash

# Run Stage 1D: CIFAR-100 Probe selection and multi-backdoor trigger alignment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/hard_sample_gap}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
RECORD_ROOT="${RECORD_ROOT:-${DATA_ROOT%/data}/third_party/BackdoorBench/record}"
ADAPTIVE_BLEND_ROOT="${ADAPTIVE_BLEND_ROOT:-}"
ADAPTIVE_BLEND_MODEL_PATH="${ADAPTIVE_BLEND_MODEL_PATH:-${MODEL_ROOT}/adaptive_blend/seed0/official_model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_trigger_alignment}"
QUALITY_REPORT="${QUALITY_REPORT:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU_ID="${GPU_ID:-0}"
BACKDOOR_GROUPS="${BACKDOOR_GROUPS:-badnet,blended,wanet,ssba,inputaware,adaptive_blend}"
BADNET_TRIGGER_PATH="${BADNET_TRIGGER_PATH:-}"
BLENDED_TRIGGER_PATH="${BLENDED_TRIGGER_PATH:-}"
WANET_STATE_PATH="${WANET_STATE_PATH:-}"
SSBA_TEST_PATH="${SSBA_TEST_PATH:-}"
INPUTAWARE_STATE_PATH="${INPUTAWARE_STATE_PATH:-}"
ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-}"

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

resolve_checkpoint() {
    local group="$1"
    case "${group}" in
        inputaware)
            for candidate in inputaware input_aware input-aware; do
                if [[ -f "${MODEL_ROOT}/${candidate}/seed0/attack_result.pt" ]]; then
                    echo "${MODEL_ROOT}/${candidate}/seed0/attack_result.pt"
                    return 0
                fi
            done
            ;;
        adaptive_blend)
            for candidate in adaptive_blend adaptive-blend adaptiveblend adap_blend; do
                if [[ -f "${MODEL_ROOT}/${candidate}/seed0/attack_result.pt" ]]; then
                    echo "${MODEL_ROOT}/${candidate}/seed0/attack_result.pt"
                    return 0
                fi
            done
            ;;
        *)
            if [[ -f "${MODEL_ROOT}/${group}/seed0/attack_result.pt" ]]; then
                echo "${MODEL_ROOT}/${group}/seed0/attack_result.pt"
                return 0
            fi
            ;;
    esac
    return 1
}

for seed in 1 2 3; do
    checkpoint="${MODEL_ROOT}/${CLEAN_GROUP:-clean_select_shared}/seed${seed}/clean_model.pth"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: Probe reference checkpoint not found: ${checkpoint}" >&2
        exit 1
    fi
done
if [[ ! -f "${MODEL_ROOT}/${CLEAN_GROUP:-clean_select_shared}/seed0/clean_model.pth" ]]; then
    echo "ERROR: Clean seed0 checkpoint not found" >&2
    exit 1
fi

IFS=',' read -r -a BACKDOOR_GROUP_ARRAY <<< "${BACKDOOR_GROUPS}"
for group in "${BACKDOOR_GROUP_ARRAY[@]}"; do
    if ! checkpoint="$(resolve_checkpoint "${group}")"; then
        if [[ "${group}" == "adaptive_blend" && -f "${ADAPTIVE_BLEND_MODEL_PATH}" ]]; then
            echo "checkpoint ${group}: ${ADAPTIVE_BLEND_MODEL_PATH}"
            continue
        fi
        echo "ERROR: seed0 checkpoint not found for backdoor group: ${group}" >&2
        exit 1
    fi
    echo "checkpoint ${group}: ${checkpoint}"
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
RECORD_ARGS=()
if [[ -d "${RECORD_ROOT}" ]]; then
    RECORD_ARGS+=(--record-root "${RECORD_ROOT}")
fi
TRIGGER_ARGS=()
[[ -n "${BADNET_TRIGGER_PATH}" ]] && TRIGGER_ARGS+=(--badnet-trigger-path "${BADNET_TRIGGER_PATH}")
[[ -n "${BLENDED_TRIGGER_PATH}" ]] && TRIGGER_ARGS+=(--blended-trigger-path "${BLENDED_TRIGGER_PATH}")
[[ -n "${WANET_STATE_PATH}" ]] && TRIGGER_ARGS+=(--wanet-state-path "${WANET_STATE_PATH}")
[[ -n "${SSBA_TEST_PATH}" ]] && TRIGGER_ARGS+=(--ssba-test-path "${SSBA_TEST_PATH}")
[[ -n "${INPUTAWARE_STATE_PATH}" ]] && TRIGGER_ARGS+=(--inputaware-state-path "${INPUTAWARE_STATE_PATH}")
[[ -n "${ADAPTIVE_BLEND_TRIGGER_PATH}" ]] && TRIGGER_ARGS+=(--adaptive-blend-trigger-path "${ADAPTIVE_BLEND_TRIGGER_PATH}")

{
    echo "[$(date --iso-8601=seconds)] Stage 1D CIFAR-100 Probe multi-backdoor alignment"
    echo "[$(date --iso-8601=seconds)] device=cuda:0 batch_size=${BATCH_SIZE} target=0 groups=${BACKDOOR_GROUPS}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trigger_alignment_probe_multitype.py" \
        --data-root "${DATA_ROOT}" \
        --model-root "${MODEL_ROOT}" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --adaptive-blend-root "${ADAPTIVE_BLEND_ROOT}" \
        --adaptive-blend-model-path "${ADAPTIVE_BLEND_MODEL_PATH}" \
        "${RECORD_ARGS[@]}" \
        "${TRIGGER_ARGS[@]}" \
        --output-root "${OUTPUT_ROOT}" \
        --clean-group "${CLEAN_GROUP:-clean_select_shared}" \
        --backdoor-groups "${BACKDOOR_GROUPS}" \
        --probe-pool-count "${PROBE_POOL_COUNT:-1000}" \
        --candidate-count "${CANDIDATE_COUNT:-1000}" \
        --top-final "${TOP_FINAL:-100}" \
        --candidate-seed "${CANDIDATE_SEED:-2031}" \
        --batch-size "${BATCH_SIZE}" \
        "${QUALITY_ARGS[@]}" \
        --device cuda:0
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
