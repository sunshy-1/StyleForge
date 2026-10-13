#!/usr/bin/env bash
# Source after setting ROOT and changing to the repository directory.
# API credentials are configured by the API-using entry points themselves.
# Use the Python selected by the caller's activated environment.  The entry
# points deliberately do not activate or switch environments themselves.
source "${ROOT}/configs/run.env"
PYTHON="${PYTHON:-python3}"

# Accept either the style directory itself (data/styles) or its dataset
# container (data). This keeps switching between data and data-all simple.
DATASET_ROOT="$(${PYTHON} - "${DATASET_ROOT}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser()

def has_task_dirs(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    return any(
        child.is_dir() and (child / "target_images").is_dir()
        for child in candidate.iterdir()
    )

if not has_task_dirs(root) and has_task_dirs(root / "styles"):
    root = root / "styles"
print(root)
PY
)"
export CUDA_VISIBLE_DEVICES
export PYTHON MODEL_SIZES MODEL RUN_ROOT DATASET_ROOT PROMPTS_FILE CONFIG DRY_RUN
export CHECKPOINT_ROOT CHECKPOINT_STEP TASKS SEED STEPS GUIDANCE DEVICE DTYPE
export CANDIDATE_COUNT MAX_REF_IMAGES MAX_SCHEMA_CHARS ACTUATOR_MODE OVERWRITE

source "${ROOT}/scripts/model_dispatch.sh"

runtime_command() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

runtime_preflight() {
    "${PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python >=3.10 is required; activate your prepared environment before running this script.")' || return "$?"
    "${PYTHON}" "${ROOT}/scripts/workflow.py" preflight "$1"
}

runtime_tasks() {
    TASK_ARGS=()
    if [[ -n "${TASKS}" ]]; then
        local normalized="${TASKS//,/ }"
        read -r -a TASK_ARGS <<< "${normalized}"
        TASK_SELECTION_SOURCE="TASKS override"
    else
        mapfile -t TASK_ARGS < <("${PYTHON}" "${ROOT}/scripts/workflow.py" tasks)
        TASK_SELECTION_SOURCE="automatic discovery"
    fi
    [[ ${#TASK_ARGS[@]} -gt 0 ]] || { echo "No style tasks selected" >&2; return 1; }
    TASKS_CSV="$(IFS=,; echo "${TASK_ARGS[*]}")"
    echo "[runtime] ${TASK_SELECTION_SOURCE}: ${#TASK_ARGS[@]} style task(s) under ${DATASET_ROOT}"
}
