#!/usr/bin/env bash
set -euo pipefail

# One-command StyleForge extraction, schema compilation, and generation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/runtime.sh"

# VLM API settings for StyleForge reference extraction.
# The request protocol is selected by backend.vlm_protocol in the config.
VLM_BASE_URL=
VLM_MODEL=gpt-5.5
VLM_API_KEY=""
VLM_REASONING_EFFORT=xhigh

export VLM_BASE_URL VLM_MODEL VLM_API_KEY VLM_REASONING_EFFORT
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/styleforge}"
CANDIDATE_SEED_STRIDE=1000
SCENE_KEY=all16
MAX_DETAIL_CHARS=900

usage() {
    cat <<'EOF'
Usage: bash run_styleforge.sh

Runs reference style extraction, schema compilation, and FLUX 4B then 9B generation.
Run run_training_free_baselines.sh for DSE/RCG and run_eval.sh for all six metrics.

Configure these in the API settings block at the top of this script:
  VLM_BASE_URL, VLM_MODEL, VLM_API_KEY

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
    echo "run_styleforge.sh: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "required file does not exist: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "required directory does not exist: $1"
}

source "${ROOT}/scripts/output_paths.sh"

runtime_dispatch_models styleforge "${BASH_SOURCE[0]}"
runtime_preflight styleforge
assert_output_under_outputs "${RUN_ROOT}" "RUN_ROOT" || die "refusing to write outside outputs/"
RUN_ROOT="${OUTPUT_PATH_ABS}"
assert_output_under_outputs "${OUTPUT_ROOT}" "OUTPUT_ROOT" || die "refusing to write outside outputs/"
OUTPUT_ROOT="${OUTPUT_PATH_ABS}"
if [[ "${DRY_RUN}" != 1 ]]; then mkdir -p "${OUTPUT_ROOT}"; fi
runtime_tasks
CONTRACT_ROOT="${RUN_ROOT}/extraction"
SCHEMA_ROOT="${RUN_ROOT}/style_schema"
STYLEFORGE_ROOT="${OUTPUT_ROOT}"

echo "[styleforge] extracting reference-only style contracts"
EXTRACT_ARGS=(
    --config "${CONFIG}"
    --dataset-root "${DATASET_ROOT}"
    --output-root "${CONTRACT_ROOT}"
    --tasks "${TASKS_CSV}"
    --scene-key "${SCENE_KEY}"
    --seed "${SEED}"
    --max-detail-chars "${MAX_DETAIL_CHARS}"
    --container-guard
    --reference-only-inputs
    --contracts-only
)
if [[ -n "${STYLEFORGE_CONTRACT_SOURCE_ROOT:-}" ]]; then
    EXTRACT_ARGS+=(--contract-source-root "${STYLEFORGE_CONTRACT_SOURCE_ROOT}")
elif [[ "${OVERWRITE}" == "1" ]]; then
    EXTRACT_ARGS+=(--force-contracts)
fi
runtime_command "${PYTHON}" scripts/extract_style.py "${EXTRACT_ARGS[@]}"

echo "[styleforge] compiling style schemas"
COMPILE_ARGS=(
    --config "${CONFIG}"
    --detail-contract-root "${CONTRACT_ROOT}/contracts"
    --output-root "${SCHEMA_ROOT}"
    --dataset-root "${DATASET_ROOT}"
    --max-chars "${MAX_SCHEMA_CHARS}"
    --actuator-mode "${ACTUATOR_MODE}"
    --tasks "${TASK_ARGS[@]}"
)
runtime_command "${PYTHON}" scripts/compile_style.py "${COMPILE_ARGS[@]}"

echo "[styleforge] generating FLUX ${MODEL_SIZE} images"
GENERATE_ARGS=(
    --config "${CONFIG}"
    --dataset-root "${DATASET_ROOT}"
    --style-schema-run "${SCHEMA_ROOT}"
    --output-root "${STYLEFORGE_ROOT}"
    --tasks "${TASKS_CSV}"
    --content-prompts-file "${PROMPTS_FILE}"
    --scene-key "${SCENE_KEY}"
    --seed "${SEED}"
    --steps "${STEPS}"
    --guidance "${GUIDANCE}"
    --candidate-count "${CANDIDATE_COUNT}"
    --candidate-seed-stride "${CANDIDATE_SEED_STRIDE}"
    --flux2-base-model "${MODEL}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
)
if [[ "${OVERWRITE}" == 1 ]]; then GENERATE_ARGS+=(--overwrite); fi
runtime_command "${PYTHON}" scripts/generate_method.py "${GENERATE_ARGS[@]}"
STYLEFORGE_SUMMARY="${STYLEFORGE_ROOT}/summary.json"
if [[ "${DRY_RUN}" != 1 ]]; then require_file "${STYLEFORGE_SUMMARY}"; fi
echo "StyleForge FLUX ${MODEL_SIZE} completed (DRY_RUN=${DRY_RUN}): ${OUTPUT_ROOT}"
