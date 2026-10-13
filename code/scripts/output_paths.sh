#!/usr/bin/env bash
# Shared guard for generated artifacts. Call after setting ROOT and PYTHON.

resolve_repo_path() {
    "${PYTHON:-python3}" - "$1" "${ROOT}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
if not path.is_absolute():
    path = Path(sys.argv[2]) / path
print(path.resolve())
PY
}

assert_output_under_outputs() {
    local requested_path="$1"
    local description="${2:-output path}"
    local canonical_path

    canonical_path="$("${PYTHON:-python3}" - "${requested_path}" "${ROOT}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[2]).resolve()
outputs_path = repo_root / "outputs"
outputs_root = outputs_path.resolve()
if outputs_root != outputs_path:
    raise SystemExit("the repository outputs/ directory must not resolve through a symlink")

requested = Path(sys.argv[1]).expanduser()
if not requested.is_absolute():
    requested = repo_root / requested
candidate = requested.resolve()
if candidate != outputs_root and outputs_root not in candidate.parents:
    raise SystemExit("path is outside the repository outputs/ directory")
print(candidate)
PY
)" || {
        echo "${description} must resolve inside ${ROOT}/outputs: ${requested_path}" >&2
        return 1
    }

    OUTPUT_PATH_ABS="${canonical_path}"
}
