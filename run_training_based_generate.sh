#!/usr/bin/env bash
set -euo pipefail

# Generate images from existing FLUX checkpoints. This stage never trains.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/runtime.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}}"
METHODS="${METHODS:-vanilla_sft pso dopsd}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: bash run_training_based_generate.sh

Generate images from existing FLUX 4B and 9B checkpoints sequentially. This stage does not train and
writes summary.json under OUTPUT_ROOT/<method>/.

Environment variables:
  METHODS='vanilla_sft pso dopsd'  MODEL_SIZES='4b 9b' (default)
  MODEL_SIZE=4b|9b                  run only one model
  DATASET_ROOT=data/styles          CHECKPOINT_STEP=1001
  CHECKPOINT_ROOT=training_based_baselines/checkpoints
  MODEL='models/FLUX.2-klein-{SIZE}'
  RUN_ROOT='outputs/flux_{size}'
  OVERWRITE=1                       regenerate existing images
  DRY_RUN=1                         validate task and argument selection only

When TASKS is unset, every style with reference images under DATASET_ROOT is
selected automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

die() {
    echo "run_training_based_generate.sh: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "required file does not exist: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "required directory does not exist: $1"
}

source "${ROOT}/scripts/output_paths.sh"

export METHODS
runtime_dispatch_models training_based "${BASH_SOURCE[0]}"
runtime_preflight training_based
assert_output_under_outputs "${OUTPUT_ROOT}" "OUTPUT_ROOT" || die "refusing to write outside outputs/"
OUTPUT_ROOT="${OUTPUT_PATH_ABS}"
if [[ "${DRY_RUN}" != 1 ]]; then mkdir -p "${OUTPUT_ROOT}"; fi
runtime_tasks

METHOD_ARGS=()
read -r -a METHOD_ARGS <<< "${METHODS}"
[[ ${#METHOD_ARGS[@]} -gt 0 ]] || die "METHODS cannot be empty"
for method in "${METHOD_ARGS[@]}"; do
    case "${method}" in
        vanilla_sft|pso|dopsd) ;;
        *) die "unsupported method: ${method}" ;;
    esac
done

for method in "${METHOD_ARGS[@]}"; do
    echo "[training-based/generate] generating ${method} (${MODEL_SIZE})"
    GENERATE_ARGS=(
        --method "${method}"
        --checkpoint-root "${CHECKPOINT_ROOT}"
        --checkpoint-step "${CHECKPOINT_STEP}"
        --model-size "${MODEL_SIZE}"
        --model "${MODEL}"
        --dataset-root "${DATASET_ROOT}"
        --content-prompts-file "${PROMPTS_FILE}"
        --output-root "${OUTPUT_ROOT}/${method}"
        --steps "${STEPS}"
        --guidance "${GUIDANCE}"
        --seed "${SEED}"
        --dtype "${DTYPE}"
        --device "${DEVICE}"
    )
    if [[ ${#TASK_ARGS[@]} -gt 0 ]]; then
        GENERATE_ARGS+=(--tasks "${TASK_ARGS[@]}")
    fi
    if [[ "${OVERWRITE}" == "1" ]]; then
        GENERATE_ARGS+=(--overwrite)
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        GENERATE_ARGS+=(--dry-run)
    fi
    "${PYTHON}" training_based_baselines/generate_flux.py "${GENERATE_ARGS[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        require_file "${OUTPUT_ROOT}/${method}/summary.json"
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Training-based generation dry run completed."
else
    echo "Training-based images generated under: ${OUTPUT_ROOT}"
fi
