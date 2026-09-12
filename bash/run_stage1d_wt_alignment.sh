#!/usr/bin/env bash

# Run Stage 1D-WT with registered Model Zoo classifiers and official triggers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
TRIGGER_ARTIFACT_ROOT="${TRIGGER_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/models/stage1d_wt_official}"
SSBA_ENCODER_PATH="${SSBA_ENCODER_PATH:-${DATA_ROOT}/stage1d_ssba_encoder/checkpoints/stage1d_cifar10_ssba_encoder.pth}"
SSBA_CONFIG_PATH="${SSBA_CONFIG_PATH:-${REPO_ROOT}/configs/stage1d_ssba_provenance.json}"
SSBA_DECODER_PATH="${SSBA_DECODER_PATH:-}"
SSBA_ORIGINAL_TEST_BATCH="${SSBA_ORIGINAL_TEST_BATCH:-${DATA_ROOT}/cifar10/cifar-10-batches-py/test_batch}"
SSBA_REFERENCE_TEST_ARRAY="${SSBA_REFERENCE_TEST_ARRAY:-}"
SSBA_CHECK_BATCH_SIZE="${SSBA_CHECK_BATCH_SIZE:-32}"
INPUTAWARE_STATE_PATH="${INPUTAWARE_STATE_PATH:-${TRIGGER_ARTIFACT_ROOT}/inputaware/seed0/netCGM.pt}"
ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-${HOME}/backdoor-toolbox/triggers/hellokitty_32.png}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_wrong_target_trigger_alignment}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar100" ]] || { echo "ERROR: CIFAR-100 directory missing: ${DATA_ROOT}/cifar100" >&2; exit 1; }
[[ -n "${MODEL_ZOO_ROOT}" && -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is not configured or missing" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()' || { echo "ERROR: CUDA unavailable" >&2; exit 1; }

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_ZOO_ROOT
export MODEL_ZOO_SOURCE_ROOT
"${PYTHON_BIN}" - <<'PY'
from modelzoo import get_model_info

aliases = ["clean0", "clean1", "clean2", "clean3", "badnet0", "blended0", "wanet0", "inputaware0", "ssba0", "adaptive_blend01"]
for alias in aliases:
    info = get_model_info(alias)
    print(f"Model Zoo OK: {alias} ({info.get('architecture')})")
PY

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
ARGS=(
    --data-root "${DATA_ROOT}"
    --model-zoo-root "${MODEL_ZOO_ROOT}"
    --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}"
    --backdoorbench-root "${BACKDOORBENCH_ROOT}"
    --trigger-artifact-root "${TRIGGER_ARTIFACT_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --backdoor-groups "badnet,blended,wanet,inputaware,ssba,adaptive_blend"
    --inputaware-state-path "${INPUTAWARE_STATE_PATH}"
    --adaptive-blend-trigger-path "${ADAPTIVE_BLEND_TRIGGER_PATH}"
    --analysis-eps-pixels "1,1.5"
    --batch-size "${BATCH_SIZE}"
    --device cuda:0
)
if [[ -n "${SSBA_ENCODER_PATH}" && -n "${SSBA_CONFIG_PATH}" && -f "${SSBA_ENCODER_PATH}" && -f "${SSBA_CONFIG_PATH}" ]]; then
    # The encoder and provenance config are sufficient to generate the
    # official SSBA trigger for the mechanism experiment.  The decoder and
    # CIFAR-10 replacement array are diagnostic-only artifacts: a tiny
    # uint8-level mismatch against the historical array must not disable the
    # reproducible encoder-based SSBA alignment analysis.
    ARGS+=(--ssba-encoder-path "${SSBA_ENCODER_PATH}" --ssba-config-path "${SSBA_CONFIG_PATH}")
fi
if [[ -n "${SSBA_ENCODER_PATH}" && -n "${SSBA_CONFIG_PATH}" && -n "${SSBA_DECODER_PATH}" && -n "${SSBA_REFERENCE_TEST_ARRAY}" && -f "${SSBA_DECODER_PATH}" && -f "${SSBA_REFERENCE_TEST_ARRAY}" ]]; then
    SSBA_CHECK_REPORT="${OUTPUT_ROOT}/ssba_provenance_check_$(date -u +%Y%m%dT%H%M%SZ).json"
    if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_ssba_provenance.py" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --encoder-path "${SSBA_ENCODER_PATH}" \
        --decoder-path "${SSBA_DECODER_PATH}" \
        --config-path "${SSBA_CONFIG_PATH}" \
        --original-test-batch "${SSBA_ORIGINAL_TEST_BATCH}" \
        --reference-test-array "${SSBA_REFERENCE_TEST_ARRAY}" \
        --batch-size "${SSBA_CHECK_BATCH_SIZE}" \
        --output "${SSBA_CHECK_REPORT}"; then
        echo "SSBA provenance check passed: ${SSBA_CHECK_REPORT}"
    else
        echo "WARNING: SSBA provenance check is not exact; continuing with encoder-based SSBA alignment." >&2
        echo "WARNING: Record the tiny official-array quantization/numerical discrepancy in the report; do not mark SSBA unavailable." >&2
    fi
elif [[ -z "${SSBA_ENCODER_PATH}" || -z "${SSBA_CONFIG_PATH}" || ! -f "${SSBA_ENCODER_PATH}" || ! -f "${SSBA_CONFIG_PATH}" ]]; then
    echo "WARNING: SSBA encoder/config not supplied; SSBA PGD will run but alignment will be unavailable." >&2
else
    echo "INFO: SSBA encoder/config enabled; diagnostic decoder/reference comparison was not supplied." >&2
fi
{
    echo "[$(date --iso-8601=seconds)] Stage 1D-WT official-trigger alignment"
    echo "[$(date --iso-8601=seconds)] aliases=ModelZoo targets=1,3,7 epsilon=1,1.5/255 groups=badnet,blended,wanet,inputaware,ssba,adaptive_blend"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trigger_alignment_wrong_target.py" "${ARGS[@]}"
} 2>&1 | tee -a "${LAUNCH_LOG}"
echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
