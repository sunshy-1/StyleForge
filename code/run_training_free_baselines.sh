#!/usr/bin/env bash
set -euo pipefail

# One-command generation for the two training-free baselines.
# Outputs are kept under outputs/ by default.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/runtime.sh"

# VLM API settings for DSE extraction. RCG does not use them.
# The request protocol is selected by backend.vlm_protocol in the config.
VLM_BASE_URL=
VLM_MODEL=gpt-5.5
VLM_API_KEY=""
VLM_REASONING_EFFORT=xhigh

export VLM_BASE_URL VLM_MODEL VLM_API_KEY VLM_REASONING_EFFORT
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}}"

usage() {
    cat <<'EOF'
Usage: bash run_training_free_baselines.sh

Runs DSE and RCG for both FLUX 4B and 9B sequentially.
Run run_eval.sh afterward for Score, Pairwise Win Rate, CLIP-Text, CLIP-Style,
VLM-F, and VLM-S.

Configure VLM_BASE_URL, VLM_MODEL, and VLM_API_KEY in the API settings block
at the top of this script.

Common overrides:
  DATASET_ROOT=data/styles  MODEL_SIZES='4b 9b' (default)
  MODEL='models/FLUX.2-klein-{SIZE}'  RUN_ROOT='outputs/flux_{size}'
  MODEL_SIZE=4b|9b                  run only one model
  DRY_RUN=1                         check paths and print commands without generation

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
    echo "run_training_free_baselines.sh: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "required file does not exist: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "required directory does not exist: $1"
}

source "${ROOT}/scripts/output_paths.sh"

runtime_dispatch_models training_free "${BASH_SOURCE[0]}"
runtime_preflight training_free
runtime_tasks

RUN_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --content-prompts-file "${PROMPTS_FILE}"
    --model "${MODEL}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --steps "${STEPS}"
    --guidance "${GUIDANCE}"
    --seed "${SEED}"
    --max-ref-images "${MAX_REF_IMAGES}"
    --candidate-count "${CANDIDATE_COUNT}"
    --vlm-config "${CONFIG}"
)
if [[ ${#TASK_ARGS[@]} -gt 0 ]]; then
    RUN_ARGS+=(--tasks "${TASK_ARGS[@]}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    RUN_ARGS+=(--overwrite)
fi

require_dir "${DATASET_ROOT}"
require_file "${PROMPTS_FILE}"
require_dir "${MODEL}"
assert_output_under_outputs "${OUTPUT_ROOT}" "OUTPUT_ROOT" || die "refusing to write outside outputs/"
OUTPUT_ROOT="${OUTPUT_PATH_ABS}"
if [[ "${DRY_RUN}" != 1 ]]; then mkdir -p "${OUTPUT_ROOT}"; fi

for method in dse rcg; do
    method_root="${OUTPUT_ROOT}/${method}"
    if [[ "${DRY_RUN}" != 1 ]]; then mkdir -p "${method_root}"; fi
    echo "[training-free] generating ${method} (FLUX ${MODEL_SIZE})"
    runtime_command "${PYTHON}" training_free_baselines/run_baselines.py \
        "${RUN_ARGS[@]}" \
        --methods "${method}" \
        --output-root "${method_root}"
done

echo "Training-free FLUX ${MODEL_SIZE} completed (DRY_RUN=${DRY_RUN}): ${OUTPUT_ROOT}"
