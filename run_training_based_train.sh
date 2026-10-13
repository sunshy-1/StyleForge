#!/usr/bin/env bash
set -euo pipefail

# Prepare data and train FLUX-based methods. Image generation is separate.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3}"
DATASET_ROOT="${DATASET_ROOT:-data/styles}"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-${ROOT}/data/flux_training}"
MODEL_SIZE="${MODEL_SIZE:-4b}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/training_based_baselines/checkpoints}"
MODEL="${MODEL:-models/FLUX.2-klein-${MODEL_SIZE^^}}"
METHODS="${METHODS:-vanilla_sft pso dopsd}"
TASKS="${TASKS:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: bash run_training_based_train.sh

Prepare training data and train the selected FLUX methods. Prepared data is
stored in data/flux_training; checkpoints in training_based_baselines/checkpoints.

Environment variables:
  METHODS='vanilla_sft pso dopsd'  MODEL_SIZE=4b|9b
  DATASET_ROOT=data/styles
  TRAINING_DATA_ROOT=data/flux_training
  CHECKPOINT_ROOT=training_based_baselines/checkpoints
  MODEL=models/FLUX.2-klein-4B or models/FLUX.2-klein-9B
  DRY_RUN=1                         validate training dispatches only

When TASKS is unset, every style directory under DATASET_ROOT is selected for
training automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

die() {
    echo "run_training_based_train.sh: $*" >&2
    exit 1
}

require_dir() {
    [[ -d "$1" ]] || die "required directory does not exist: $1"
}

resolve_path() {
    "${PYTHON}" - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

case "${MODEL_SIZE}" in
    4b|9b) ;;
    *) die "MODEL_SIZE must be 4b or 9b" ;;
esac
require_dir "${DATASET_ROOT}"
require_dir "${MODEL}"

# Resolve these paths once, before dispatching into the original method
# scripts.  They are then passed as absolute paths so their internal `cd`
# commands cannot redirect outputs to a parent directory.
DATASET_ROOT_ABS="$(resolve_path "${DATASET_ROOT}")"
MODEL_ABS="$(resolve_path "${MODEL}")"
TRAINING_DATA_ROOT_ABS="$(resolve_path "${TRAINING_DATA_ROOT}")"
CHECKPOINT_ROOT_ABS="$(resolve_path "${CHECKPOINT_ROOT}")"
EXPECTED_CHECKPOINT_ROOT="$(resolve_path "${ROOT}/training_based_baselines/checkpoints")"
[[ "${CHECKPOINT_ROOT_ABS}" == "${EXPECTED_CHECKPOINT_ROOT}" ]] || die "CHECKPOINT_ROOT must be ${EXPECTED_CHECKPOINT_ROOT}; got ${CHECKPOINT_ROOT_ABS}"
mkdir -p "${CHECKPOINT_ROOT_ABS}"

TASK_ARGS=()
if [[ -n "${TASKS}" ]]; then
    TASKS_NORMALIZED="${TASKS//,/ }"
    read -r -a TASK_ARGS <<< "${TASKS_NORMALIZED}"
fi

METHOD_ARGS=()
read -r -a METHOD_ARGS <<< "${METHODS}"
[[ ${#METHOD_ARGS[@]} -gt 0 ]] || die "METHODS cannot be empty"
for method in "${METHOD_ARGS[@]}"; do
    case "${method}" in
        vanilla_sft|pso|dopsd) ;;
        *) die "unsupported method: ${method}" ;;
    esac
done

CHECKPOINT_TASKS=("${TASK_ARGS[@]}")
if [[ ${#CHECKPOINT_TASKS[@]} -eq 0 ]]; then
    while IFS= read -r task_dir; do
        CHECKPOINT_TASKS+=("$(basename "${task_dir}")")
    done < <(find "${DATASET_ROOT_ABS}" -mindepth 1 -maxdepth 1 -type d -print | sort)
fi
[[ ${#CHECKPOINT_TASKS[@]} -gt 0 ]] || die "no training tasks found under ${DATASET_ROOT_ABS}"

checkpoint_adapter_subdir() {
    case "$1" in
        vanilla_sft) printf '%s\n' 'sft' ;;
        pso) printf '%s\n' 'pso' ;;
        dopsd) printf '%s\n' 'student' ;;
        *) die "unsupported method for checkpoint validation: $1" ;;
    esac
}

verify_checkpoint_layout() {
    local method="$1"
    local method_root="${CHECKPOINT_ROOT_ABS}/${method}_${MODEL_SIZE}"
    local adapter_subdir
    adapter_subdir="$(checkpoint_adapter_subdir "${method}")"
    local task
    local adapter
    local found

    [[ -d "${method_root}" ]] || die "missing method checkpoint directory: ${method_root}"
    for task in "${CHECKPOINT_TASKS[@]}"; do
        found=0
        while IFS= read -r -d '' adapter; do
            if [[ -f "${adapter}/adapter_config.json" && -f "${adapter}/adapter_model.safetensors" ]]; then
                found=1
                break
            fi
        done < <(
            find "${method_root}" \
                -mindepth 4 -maxdepth 4 \
                -type d -name "${adapter_subdir}" \
                -path "*${task}*" -print0
        )
        if [[ "${found}" -ne 1 ]]; then
            die "no complete ${method}/${task} adapter under ${method_root}; expected .../checkpoints/lora_gen_step_*/${adapter_subdir}/"
        fi
    done
}

PREPARE_ARGS=(
    --dataset-root "${DATASET_ROOT_ABS}"
    --output-root "${TRAINING_DATA_ROOT_ABS}"
    --overwrite
)
if [[ ${#TASK_ARGS[@]} -gt 0 ]]; then
    PREPARE_ARGS+=(--tasks "${TASK_ARGS[@]}")
fi

echo "[training-based/train] preparing FLUX training data"
"${PYTHON}" training_based_baselines/prepare_flux_dataset.py "${PREPARE_ARGS[@]}"

for method in "${METHOD_ARGS[@]}"; do
    METHOD_OUTPUT_DIR="${CHECKPOINT_ROOT_ABS}/${method}_${MODEL_SIZE}"
    case "${METHOD_OUTPUT_DIR}" in
        "${CHECKPOINT_ROOT_ABS}"/*) ;;
        *) die "refusing to write ${method} outside ${CHECKPOINT_ROOT_ABS}" ;;
    esac
    echo "[training-based/train] training ${method} (${MODEL_SIZE})"
    DATASET_DIR="${TRAINING_DATA_ROOT_ABS}" \
    PRETRAINED_MODEL="${MODEL_ABS}" \
    OUTPUT_DIR="${METHOD_OUTPUT_DIR}" \
    DRY_RUN="${DRY_RUN}" \
    bash training_based_baselines/run_flux_training.sh "${method}" "${MODEL_SIZE}" "${TASK_ARGS[@]}"

    if [[ "${DRY_RUN}" != "1" && ! -d "${METHOD_OUTPUT_DIR}" ]]; then
        die "training exited without creating its checkpoint directory: ${METHOD_OUTPUT_DIR}"
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
        verify_checkpoint_layout "${method}"
    fi
done

echo "Training checkpoints generated under: ${CHECKPOINT_ROOT_ABS}"
