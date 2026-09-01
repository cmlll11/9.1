#!/usr/bin/env bash

# Train disjoint-data CIFAR-10 models, fit x+f GAP mappings, and probe hard samples.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_LABEL=0
EPOCHS=100
MODEL_ROOT="${REPO_ROOT}/artifacts/models/hard_sample_gap"
MAPPING_ROOT="${REPO_ROOT}/artifacts/mappings/hard_sample_gap"
# The old clean_select artifacts were trained on the 3,000-image partition.
# Keep them untouched and write the corrected 47,000-image runs separately.
CLEAN_SELECT_GROUP="clean_select_shared"
# Keep partitions below DATA_ROOT so BackdoorBench can reconstruct them from
# the saved dataset_path inside attack_result.pt.
PARTITION_ROOT="${DATA_ROOT}/hard_sample_gap"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/hard_samples/epsilon4"
SUMMARY="${REPO_ROOT}/reports/hard_sample_gap_summary.json"
CSV="${REPO_ROOT}/reports/hard_sample_gap_per_model.csv"

mkdir -p "${MODEL_ROOT}" "${MAPPING_ROOT}" "${OUTPUT_ROOT}"

if [[ ! -f "${PARTITION_ROOT}/manifest.json" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_hard_sample_partitions.py" \
        --data-root "${DATA_ROOT}" --output-root "${PARTITION_ROOT}" \
        --selection-size 3000 --seed 2026
fi

"${PYTHON_BIN}" - "${PARTITION_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("selection_size") != 3000 or manifest.get("shared_size") != 47000:
    raise SystemExit(
        "partition mismatch: expected selection=3000 and shared=47000"
    )
PY

SELECTION_DATA="${PARTITION_ROOT}/selection"
SHARED_DATA="${PARTITION_ROOT}/shared"
PARTITION_PYTHONPATH="${REPO_ROOT}/scripts/partition_runtime${PYTHONPATH:+:${PYTHONPATH}}"

train_clean() {
    local group="$1" seed="$2" data_root="$3"
    local run_name="mdl_uap_hard_${group}_seed${seed}"
    local run_dir="${BACKDOORBENCH_ROOT}/record/${run_name}"
    local weights="${run_dir}/clean_model.pth"
    local output="${MODEL_ROOT}/${group}/seed${seed}/attack_result.pt"
    if [[ -f "${output}" ]]; then
        log_step "Skip classifier: group=${group} seed=${seed}"
        return
    fi
    if [[ -d "${run_dir}" && ! -f "${weights}" ]]; then
        echo "ERROR: incomplete run directory exists: ${run_dir}" >&2
        exit 1
    fi
    if [[ ! -f "${weights}" ]]; then
        log_step "Train clean classifier: group=${group} seed=${seed}"
        (
            cd "${BACKDOORBENCH_ROOT}"
            PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" attack/prototype.py \
                --yaml_path config/attack/prototype/cifar10.yaml \
                --dataset_path "${data_root}" \
                --save_folder_name "${run_name}" \
                --random_seed "${seed}" --frequency_save 10 --device cuda:0
        ) > "${REPO_ROOT}/outputs/hard_${group}_seed${seed}.log" 2>&1
    fi
    mkdir -p "$(dirname "${output}")"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/package_clean_model.py" \
        --weights "${weights}" --output "${output}"
}

train_trigger() {
    local trigger="$1" seed="$2"
    local run_name="mdl_uap_hard_${trigger}_seed${seed}"
    local run_dir="${BACKDOORBENCH_ROOT}/record/${run_name}"
    local result="${run_dir}/attack_result.pt"
    local output="${MODEL_ROOT}/${trigger}/seed${seed}/attack_result.pt"
    if [[ -f "${output}" ]]; then
        log_step "Skip classifier: trigger=${trigger} seed=${seed}"
        return
    fi
    if [[ -d "${run_dir}" ]]; then
        echo "ERROR: incomplete run directory exists: ${run_dir}" >&2
        exit 1
    fi
    log_step "Train BackdoorBench classifier: trigger=${trigger} seed=${seed}"
    (
        cd "${BACKDOORBENCH_ROOT}"
        case "${trigger}" in
            badnet)
                PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" attack/badnet.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/badnet/default.yaml \
                    --patch_mask_path resource/badnet/trigger_image.png \
                    --dataset_path "${SHARED_DATA}" --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${seed}" --frequency_save 10 --device cuda:0
                ;;
            lf)
                PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" attack/lf.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/lf/default.yaml \
                    --dataset_path "${SHARED_DATA}" --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${seed}" --frequency_save 10 --device cuda:0
                ;;
            blended)
                PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" attack/blended.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/blended/default.yaml \
                    --dataset_path "${SHARED_DATA}" --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${seed}" --frequency_save 10 --device cuda:0
                ;;
            wanet)
                PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" attack/wanet.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/wanet/default.yaml \
                    --dataset_path "${SHARED_DATA}" --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${seed}" --frequency_save 10 --device cuda:0
                ;;
            *)
                echo "ERROR: unsupported trigger=${trigger}" >&2
                exit 1
                ;;
        esac
    ) > "${REPO_ROOT}/outputs/hard_${trigger}_seed${seed}.log" 2>&1
    mkdir -p "$(dirname "${output}")"
    cp "${result}" "${output}"
}

train_mapping() {
    local group="$1" seed="$2" data_root="$3"
    local result="${MODEL_ROOT}/${group}/seed${seed}/attack_result.pt"
    local output="${MAPPING_ROOT}/${group}/seed${seed}"
    if [[ -f "${output}/mapping.pt" ]]; then
        log_step "Skip GAP mapping: group=${group} seed=${seed}"
        return
    fi
    log_step "Train GAP mapping: group=${group} seed=${seed}"
    PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_residual_mapping.py" \
        --result "${result}" --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --gap-root "${GAP_ROOT}" --data-root "${data_root}" --output "${output}" \
        --attack-goal targeted --target "${TARGET_LABEL}" --epsilon 4/255 \
        --seed "${seed}" --split-seed 2026 --epochs "${EPOCHS}" \
        --max-batches 50 --device cuda:0
}

log_step "Purpose: hard-sample targeted x+f GAP probe"
log_step "Data: train=47000, hard-sample candidate pool=3000 held out"

for seed in 0 1 2 3 4; do
    train_clean "${CLEAN_SELECT_GROUP}" "${seed}" "${SHARED_DATA}"
done
for seed in 5 6 7 8 9; do
    train_clean clean_eval "${seed}" "${SHARED_DATA}"
done
for trigger in badnet lf blended wanet; do
    for seed in 0 1 2 3 4; do
        train_trigger "${trigger}" "${seed}"
    done
done

PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_hard_sample_models.py" \
    --data-root "${DATA_ROOT}" --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --model-root "${MODEL_ROOT}" --output "${REPO_ROOT}/reports/hard_sample_gap_model_gates.json" \
    --selection-group "${CLEAN_SELECT_GROUP}" \
    --device cuda:0 --workers 4

for seed in 0 1 2 3 4; do
    train_mapping "${CLEAN_SELECT_GROUP}" "${seed}" "${SHARED_DATA}"
done
for seed in 5 6 7 8 9; do
    train_mapping clean_eval "${seed}" "${SHARED_DATA}"
done
for trigger in badnet lf blended wanet; do
    for seed in 0 1 2 3 4; do
        train_mapping "${trigger}" "${seed}" "${SHARED_DATA}"
    done
done

PYTHONPATH="${PARTITION_PYTHONPATH}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_hard_sample_gap.py" \
    --data-root "${DATA_ROOT}" --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --gap-root "${GAP_ROOT}" --model-root "${MODEL_ROOT}" \
    --mapping-root "${MAPPING_ROOT}" --output-root "${OUTPUT_ROOT}" \
    --summary "${SUMMARY}" --csv "${CSV}" --target 0 --epsilon 4/255 \
    --candidate-root "${SELECTION_DATA}" \
    --selection-group "${CLEAN_SELECT_GROUP}" \
    --gate-report "${REPO_ROOT}/reports/hard_sample_gap_model_gates.json" \
    --selection-seeds 0,1,2,3,4 --evaluation-seeds 5,6,7,8,9 \
    --backdoor-seeds 0,1,2,3,4 --hard-count 100 --batch-size 128 \
    --workers 4 --device cuda:0

log_step "Complete: summary=${SUMMARY} CSV=${CSV}"
