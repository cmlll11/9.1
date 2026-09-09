#!/usr/bin/env bash

# Train the Stage 1D models with the official attack implementations.
#
# BackdoorBench is used for Clean, BadNet, Blended, WaNet, SSBA, and
# Input-Aware.  Adaptive-Blend is delegated to the official
# backdoor-toolbox checkout because it is not implemented by BackdoorBench.
# The script intentionally uses the complete CIFAR-10 train split; it does
# not use the previous hard-sample partitions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/stage1d_official}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
ADAPTIVE_BLEND_ROOT="${ADAPTIVE_BLEND_ROOT:-}"
ADAPTIVE_BLEND_MODEL_PATH="${ADAPTIVE_BLEND_MODEL_PATH:-}"
ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-}"
GPU_ID="${GPU_ID:-0}"
CLEAN_SEEDS="${CLEAN_SEEDS:-0,1,2,3}"
BACKDOOR_SEED="${BACKDOOR_SEED:-0}"
EPOCHS="${EPOCHS:-100}"
PRATIO="${PRATIO:-0.1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}/cifar10" ]]; then
    echo "ERROR: full CIFAR-10 directory not found: ${DATA_ROOT}/cifar10" >&2
    exit 1
fi
if [[ ! -d "${BACKDOORBENCH_ROOT}/attack" ]]; then
    echo "ERROR: BackdoorBench checkout not found: ${BACKDOORBENCH_ROOT}" >&2
    exit 1
fi

mkdir -p "${MODEL_ROOT}" "${REPO_ROOT}/outputs/stage1d_training"

run_name() {
    echo "stage1d_${1}_seed${2}"
}

copy_backdoorbench_result() {
    local group="$1" seed="$2" run="$3"
    local source="${BACKDOORBENCH_ROOT}/record/${run}/attack_result.pt"
    local destination="${MODEL_ROOT}/${group}/seed${seed}"
    if [[ ! -f "${source}" ]]; then
        echo "ERROR: official attack did not create ${source}" >&2
        exit 1
    fi
    mkdir -p "${destination}"
    cp "${source}" "${destination}/attack_result.pt"
    cp -f "${BACKDOORBENCH_ROOT}/record/${run}/info.pickle" "${destination}/" 2>/dev/null || true
    if [[ "${group}" == "inputaware" ]]; then
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/netCGM.pt" "${destination}/" 2>/dev/null || true
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/mask_state_dict.pt" "${destination}/" 2>/dev/null || true
    fi
    if [[ "${group}" == "wanet" ]]; then
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/identity_grid" "${destination}/state_identity_grid.pt" 2>/dev/null || true
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/noise_grid" "${destination}/state_noise_grid.pt" 2>/dev/null || true
    fi
    if [[ "${group}" == "ssba" ]]; then
        # Keep the source artifact for provenance.  Stage 1D will not use it
        # as a CIFAR-100 trigger unless a separate CIFAR-100 array is supplied.
        cp -f "${SSBA_TEST_PATH}" "${destination}/cifar10_test_replace_imgs.npy"
    fi
}

train_clean() {
    local seed="$1" run
    run="$(run_name clean "${seed}")"
    if [[ -f "${MODEL_ROOT}/clean_select_shared/seed${seed}/attack_result.pt" ]]; then
        echo "Skip existing Clean model: ${MODEL_ROOT}/clean_select_shared/seed${seed}/attack_result.pt"
        return
    fi
    echo "Training official Clean model: seed=${seed}, epochs=${EPOCHS}"
    (
        cd "${BACKDOORBENCH_ROOT}"
        "${PYTHON_BIN}" attack/prototype.py \
            --yaml_path config/attack/prototype/cifar10.yaml \
            --dataset_path "${DATA_ROOT}" \
            --save_folder_name "${run}" \
            --random_seed "${seed}" \
            --epochs "${EPOCHS}" \
            --frequency_save 0 \
            --device cuda:0
    ) 2>&1 | tee "${REPO_ROOT}/outputs/stage1d_training/${run}.log"
    copy_backdoorbench_result clean_select_shared "${seed}" "${run}"
}

train_backdoorbench() {
    local group="$1" script="$2" config="$3" seed="${BACKDOOR_SEED}" run
    run="$(run_name "${group}" "${seed}")"
    if [[ -f "${MODEL_ROOT}/${group}/seed${seed}/attack_result.pt" ]]; then
        echo "Skip existing ${group} model: ${MODEL_ROOT}/${group}/seed${seed}/attack_result.pt"
        return
    fi
    echo "Training official ${group} model: seed=${seed}, epochs=${EPOCHS}"
    (
        cd "${BACKDOORBENCH_ROOT}"
        "${PYTHON_BIN}" "${script}" \
            --yaml_path config/attack/prototype/cifar10.yaml \
            --bd_yaml_path "${config}" \
            --dataset_path "${DATA_ROOT}" \
            --save_folder_name "${run}" \
            --attack_target 0 \
            --attack_label_trans all2one \
            --pratio "${PRATIO}" \
            --random_seed "${seed}" \
            --epochs "${EPOCHS}" \
            --frequency_save 0 \
            --device cuda:0
    ) 2>&1 | tee "${REPO_ROOT}/outputs/stage1d_training/${run}.log"
    copy_backdoorbench_result "${group}" "${seed}" "${run}"
}

train_ssba() {
    local seed="${BACKDOOR_SEED}" run
    SSBA_TEST_PATH="${SSBA_TEST_PATH:-${BACKDOORBENCH_ROOT}/resource/ssba/cifar10_ssba_test_b1.npy}"
    SSBA_TRAIN_PATH="${SSBA_TRAIN_PATH:-${BACKDOORBENCH_ROOT}/resource/ssba/cifar10_ssba_train_b1.npy}"
    if [[ ! -f "${SSBA_TRAIN_PATH}" || ! -f "${SSBA_TEST_PATH}" ]]; then
        echo "ERROR: official SSBA replacement arrays are required." >&2
        echo "       train=${SSBA_TRAIN_PATH}" >&2
        echo "       test=${SSBA_TEST_PATH}" >&2
        exit 1
    fi
    run="$(run_name ssba "${seed}")"
    if [[ -f "${MODEL_ROOT}/ssba/seed${seed}/attack_result.pt" ]]; then
        echo "Skip existing SSBA model: ${MODEL_ROOT}/ssba/seed${seed}/attack_result.pt"
        return
    fi
    echo "Training official SSBA model: seed=${seed}, epochs=${EPOCHS}"
    (
        cd "${BACKDOORBENCH_ROOT}"
        "${PYTHON_BIN}" attack/ssba.py \
            --yaml_path config/attack/prototype/cifar10.yaml \
            --bd_yaml_path config/attack/ssba/default.yaml \
            --dataset_path "${DATA_ROOT}" \
            --save_folder_name "${run}" \
            --attack_target 0 \
            --attack_label_trans all2one \
            --pratio "${PRATIO}" \
            --attack_train_replace_imgs_path "${SSBA_TRAIN_PATH}" \
            --attack_test_replace_imgs_path "${SSBA_TEST_PATH}" \
            --random_seed "${seed}" \
            --epochs "${EPOCHS}" \
            --frequency_save 0 \
            --device cuda:0
    ) 2>&1 | tee "${REPO_ROOT}/outputs/stage1d_training/${run}.log"
    copy_backdoorbench_result ssba "${seed}" "${run}"
}

train_adaptive_blend() {
    if [[ -z "${ADAPTIVE_BLEND_ROOT}" ]]; then
        echo "ERROR: ADAPTIVE_BLEND_ROOT is required for official Adaptive-Blend training." >&2
        exit 1
    fi
    if [[ ! -f "${ADAPTIVE_BLEND_ROOT}/create_poisoned_set.py" || ! -f "${ADAPTIVE_BLEND_ROOT}/train_on_poisoned_set.py" ]]; then
        echo "ERROR: invalid backdoor-toolbox checkout: ${ADAPTIVE_BLEND_ROOT}" >&2
        exit 1
    fi
    if [[ -z "${ADAPTIVE_BLEND_MODEL_PATH}" ]]; then
        echo "ERROR: set ADAPTIVE_BLEND_MODEL_PATH to the official toolbox model file after its training path is known." >&2
        exit 1
    fi
    mkdir -p "${ADAPTIVE_BLEND_ROOT}/data"
    if [[ ! -e "${ADAPTIVE_BLEND_ROOT}/data/cifar10" ]]; then
        ln -s "${DATA_ROOT}/cifar10" "${ADAPTIVE_BLEND_ROOT}/data/cifar10"
    fi
    echo "Training official Adaptive-Blend model with backdoor-toolbox"
    (
        cd "${ADAPTIVE_BLEND_ROOT}"
        "${PYTHON_BIN}" create_clean_set.py -dataset cifar10
        "${PYTHON_BIN}" create_poisoned_set.py \
            -dataset cifar10 -poison_type adaptive_blend \
            -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15
        "${PYTHON_BIN}" train_on_poisoned_set.py \
            -dataset cifar10 -poison_type adaptive_blend \
            -poison_rate 0.003 -cover_rate 0.003 \
            -alpha 0.15 -test_alpha 0.2 -seed "${BACKDOOR_SEED}" -devices 0
    ) 2>&1 | tee "${REPO_ROOT}/outputs/stage1d_training/stage1d_adaptive_blend_seed${BACKDOOR_SEED}.log"
    if [[ ! -f "${ADAPTIVE_BLEND_MODEL_PATH}" ]]; then
        echo "ERROR: Adaptive-Blend model not found after official training: ${ADAPTIVE_BLEND_MODEL_PATH}" >&2
        exit 1
    fi
    mkdir -p "${MODEL_ROOT}/adaptive_blend/seed${BACKDOOR_SEED}"
    cp "${ADAPTIVE_BLEND_MODEL_PATH}" "${MODEL_ROOT}/adaptive_blend/seed${BACKDOOR_SEED}/official_model.pt"
    ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-${ADAPTIVE_BLEND_ROOT}/triggers/hellokitty_32.png}"
    if [[ ! -f "${ADAPTIVE_BLEND_TRIGGER_PATH}" ]]; then
        echo "ERROR: official Adaptive-Blend trigger not found: ${ADAPTIVE_BLEND_TRIGGER_PATH}" >&2
        exit 1
    fi
    cp "${ADAPTIVE_BLEND_TRIGGER_PATH}" "${MODEL_ROOT}/adaptive_blend/seed${BACKDOOR_SEED}/adaptive_blend_trigger.png"
}

IFS=',' read -r -a CLEAN_SEED_ARRAY <<< "${CLEAN_SEEDS}"
for clean_seed in "${CLEAN_SEED_ARRAY[@]}"; do
    train_clean "${clean_seed}"
done
train_backdoorbench badnet attack/badnet.py config/attack/badnet/default.yaml
train_backdoorbench blended attack/blended.py config/attack/blended/default.yaml
train_backdoorbench wanet attack/wanet.py config/attack/wanet/default.yaml
train_ssba
train_backdoorbench inputaware attack/inputaware.py config/attack/inputaware/default.yaml
train_adaptive_blend

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_stage1d_official_models.py" \
    --data-root "${DATA_ROOT}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --model-root "${MODEL_ROOT}" \
    --output "${MODEL_ROOT}/stage1d_model_gates.json" \
    --adaptive-blend-root "${ADAPTIVE_BLEND_ROOT}" \
    --adaptive-blend-model-path "${MODEL_ROOT}/adaptive_blend/seed${BACKDOOR_SEED}/official_model.pt" \
    --adaptive-blend-trigger-path "${MODEL_ROOT}/adaptive_blend/seed${BACKDOOR_SEED}/adaptive_blend_trigger.png" \
    --device cuda:0

echo "Official Stage 1D training complete. Models: ${MODEL_ROOT}"
