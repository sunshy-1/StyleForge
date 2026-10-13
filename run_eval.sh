#!/usr/bin/env bash
set -euo pipefail

# Use python3 from the caller's activated environment. Evaluation configuration
# comes from these defaults and CLI options, independently of generation envs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHON="$(command -v python3)"
source "${ROOT}/scripts/output_paths.sh"

MODEL_SIZES=(4b 9b)
JUDGE_SLUGS=(qwen3_vl_8b qwen3_vl_32b qwen3_5_9b)
METHODS="styleforge dse rcg vanilla_sft pso dopsd"
RUN_ROOT_PATTERN="outputs/flux_{size}"
OUTPUT_ROOT_PATTERN=""
DATASET_ROOT="data/styles"
MODELS_ROOT="models"
GPUS="1,2"
TP_SIZE=2
PORT=32133
STARTUP_TIMEOUT=600
DRY_RUN=0
WORKER=0

usage() {
    cat <<'EOF'
Usage: bash run_eval.sh [options]

Activate your SGLang environment first. With no options, evaluate all six
methods for FLUX 4B and 9B, using these judges sequentially:
  qwen3_vl_8b   Qwen3-VL-8B-Instruct
  qwen3_vl_32b  Qwen3-VL-32B-Instruct
  qwen3_5_9b   Qwen3.5-9B

All VLM metrics use SGLang on physical GPUs 1,2 with TP=2. The script starts
and stops its own services. Reports and resumable judgments are saved under
outputs/flux_{size}/evaluation/. CLIP runs before each judge service starts.
Missing methods are reported as n/a; Pairwise needs at least two summaries.

Options (no custom environment variables are required):
  --dry-run                 Check paths/summaries without creating results
  --model-size 4b|9b         Evaluate just one FLUX size (default: both)
  --judge SLUG              Evaluate just one judge (default: all three)
  --methods 'NAME ...'      Select methods (default: all six)
  --run-root PATH           Summary root (default: outputs/flux_{size})
  --output-root PATH        Report root (default: RUN_ROOT/evaluation)
  --dataset-root PATH       Reference styles or container (default: data/styles)
  --models-root PATH        Directory containing judge/CLIP weights (default: models)
  --gpus IDS                Physical CUDA device IDs (default: 1,2)
  --tp-size N               Tensor parallel size (default: 2)
  --port N                  Local SGLang port (default: 32133)
  --startup-timeout SECONDS Startup timeout (default: 600)
  -h, --help                Show this help

Use {size} in path templates when evaluating both FLUX sizes.
EOF
}

die() { echo "run_eval.sh: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift; continue ;;
        --worker) WORKER=1; shift; continue ;; # Internal isolated judge process.
        --model-size|--judge|--methods|--run-root|--output-root|--dataset-root|--models-root|--gpus|--tp-size|--port|--startup-timeout)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"
            ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
    case "$1" in
        --model-size) MODEL_SIZES=("$2") ;;
        --judge) JUDGE_SLUGS=("$2") ;;
        --methods) METHODS="$2" ;;
        --run-root) RUN_ROOT_PATTERN="$2" ;;
        --output-root) OUTPUT_ROOT_PATTERN="$2" ;;
        --dataset-root) DATASET_ROOT="$2" ;;
        --models-root) MODELS_ROOT="$2" ;;
        --gpus) GPUS="$2" ;;
        --tp-size) TP_SIZE="$2" ;;
        --port) PORT="$2" ;;
        --startup-timeout) STARTUP_TIMEOUT="$2" ;;
    esac
    shift 2
done

for size in "${MODEL_SIZES[@]}"; do
    case "$size" in 4b|9b) ;; *) die "--model-size must be 4b or 9b" ;; esac
done
for slug in "${JUDGE_SLUGS[@]}"; do
    case "$slug" in qwen3_vl_8b|qwen3_vl_32b|qwen3_5_9b) ;; *) die "unknown --judge: $slug" ;; esac
done
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus must be comma-separated device IDs"
for number in "$TP_SIZE" "$PORT" "$STARTUP_TIMEOUT"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || die "TP size, port and startup timeout must be positive integers"
done
(( PORT <= 65535 )) || die "--port must be at most 65535"
IFS=, read -r -a gpu_ids <<< "$GPUS"
(( TP_SIZE <= ${#gpu_ids[@]} )) || die "--tp-size exceeds the number of GPUs in --gpus"
read -r -a selected_methods <<< "$METHODS"
[[ ${#selected_methods[@]} -gt 0 ]] || die "--methods cannot be empty"
for method in "${selected_methods[@]}"; do
    [[ "$method" =~ ^[a-zA-Z0-9_-]+$ ]] || die "invalid method name: $method"
done
if [[ ${#MODEL_SIZES[@]} -gt 1 ]]; then
    [[ "$RUN_ROOT_PATTERN" == *'{size}'* ]] || die "--run-root must contain {size} when evaluating both sizes"
    [[ -z "$OUTPUT_ROOT_PATTERN" || "$OUTPUT_ROOT_PATTERN" == *'{size}'* ]] ||
        die "--output-root must contain {size} when evaluating both sizes"
fi
if [[ "$WORKER" == 1 ]]; then
    [[ ${#MODEL_SIZES[@]} == 1 && ${#JUDGE_SLUGS[@]} == 1 ]] || die "internal worker requires one size and judge"
fi

resolve_path() { resolve_repo_path "$1"; }
DATASET_ROOT="$("${PYTHON}" - "${DATASET_ROOT}" <<'PY'
from pathlib import Path
import sys
from scripts.workflow import resolve_dataset_root
print(resolve_dataset_root(Path(sys.argv[1]).expanduser()).resolve())
PY
)"
"${PYTHON}" "${ROOT}/scripts/workflow.py" preflight evaluation --dataset-root "${DATASET_ROOT}"
"${PYTHON}" -c 'import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec("sglang") else "SGLang is unavailable; activate your prepared evaluation environment.")'

if [[ "$WORKER" == 0 ]]; then
    overall_status=0
    dry_run_args=()
    [[ "$DRY_RUN" == 0 ]] || dry_run_args=(--dry-run)
    for size in "${MODEL_SIZES[@]}"; do
        run_root="${RUN_ROOT_PATTERN//\{size\}/$size}"
        output_root="${OUTPUT_ROOT_PATTERN//\{size\}/$size}"
        output_root="${output_root:-${run_root}/evaluation}"
        assert_output_under_outputs "$output_root" "--output-root"
        output_root="$OUTPUT_PATH_ABS"
        report_paths=()
        echo "[eval] FLUX ${size}: ${JUDGE_SLUGS[*]}"
        for slug in "${JUDGE_SLUGS[@]}"; do
            report_paths+=("${output_root}/${slug}/evaluation_report.md")
            if ! bash "${ROOT}/run_eval.sh" --worker \
                --model-size "$size" --judge "$slug" \
                --run-root "$run_root" --output-root "${output_root}/${slug}" \
                --dataset-root "$DATASET_ROOT" --models-root "$MODELS_ROOT" \
                --methods "$METHODS" --gpus "$GPUS" --tp-size "$TP_SIZE" \
                --port "$PORT" --startup-timeout "$STARTUP_TIMEOUT" "${dry_run_args[@]}"; then
                overall_status=1
                echo "[eval] FLUX ${size}, judge ${slug} failed; continuing with remaining evaluations" >&2
            fi
        done
        combined_report="${output_root}/evaluation_report.md"
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "[dry-run] FLUX ${size} combined report: ${combined_report}"
            continue
        fi
        "${PYTHON}" - "${combined_report}" "${report_paths[@]}" <<'PY'
import sys
from pathlib import Path

destination = Path(sys.argv[1]).expanduser().resolve()
reports = [Path(value).expanduser() for value in sys.argv[2:]]
destination.parent.mkdir(parents=True, exist_ok=True)
lines = [
    "# Judge Set Evaluation Report",
    "",
    "This report contains one independent evaluation table for each paper judge.",
    "",
]
for report in reports:
    label = report.parent.name
    known_slugs = ("qwen3_vl_8b", "qwen3_vl_32b", "qwen3_5_9b")
    for slug in known_slugs:
        if label == slug or label.endswith("_" + slug):
            label = slug
            break
    else:
        if label.startswith("evaluation_"):
            label = label[len("evaluation_"):]
    lines.append(f"## {label}")
    lines.append("")
    if report.is_file():
        content = report.read_text(encoding="utf-8").strip()
        if content.startswith("# Evaluation Report"):
            content = content[len("# Evaluation Report"):].lstrip()
        lines.append(content or "n/a")
    else:
        lines.append(f"Evaluation did not produce a report: `{report}`")
    lines.append("")
destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(f"Saved judge-set report: {destination}")
PY
    done
    exit "$overall_status"
fi

MODEL_SIZE="${MODEL_SIZES[0]}"
JUDGE_SLUG="${JUDGE_SLUGS[0]}"
case "$JUDGE_SLUG" in
    qwen3_vl_8b) JUDGE_LABEL="Qwen3-VL-8B-Instruct"; JUDGE_API_CONCURRENCY=8 ;;
    qwen3_vl_32b) JUDGE_LABEL="Qwen3-VL-32B-Instruct"; JUDGE_API_CONCURRENCY=64 ;;
    qwen3_5_9b) JUDGE_LABEL="Qwen3.5-9B"; JUDGE_API_CONCURRENCY=8 ;;
esac
JUDGE_MODEL_PATH="${MODELS_ROOT}/${JUDGE_LABEL}"
FREESTYLE_API_CONCURRENCY=12
PAIRWISE_API_CONCURRENCY=8
PAIRWISE_OUTPUT_NAME="pairwise_${JUDGE_SLUG}"
CLIP_MODEL_PATH="${MODELS_ROOT}/CLIP-ViT-H-14-laion2B-s32B-b79K"
METRIC_DEVICE=cuda
ENDPOINT="http://127.0.0.1:${PORT}/v1/chat/completions"
RUN_ROOT="${RUN_ROOT_PATTERN//\{size\}/$MODEL_SIZE}"
OUTPUT_ROOT="${OUTPUT_ROOT_PATTERN//\{size\}/$MODEL_SIZE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/evaluation/${JUDGE_SLUG}}"
assert_output_under_outputs "$OUTPUT_ROOT" "--output-root"
OUTPUT_ROOT="$OUTPUT_PATH_ABS"

metric_python() { env CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" "$@"; }

RUN_ROOT_ABS="$(resolve_path "${RUN_ROOT}")"
DATASET_ROOT_ABS="$(resolve_path "${DATASET_ROOT}")"
OUTPUT_ROOT_ABS="$(resolve_path "${OUTPUT_ROOT}")"
if [[ "${DRY_RUN}" != 1 ]]; then mkdir -p "${OUTPUT_ROOT_ABS}"; fi

JUDGE_MODEL_PATH_ABS="$(resolve_path "${JUDGE_MODEL_PATH}")"
[[ -d "${JUDGE_MODEL_PATH_ABS}" ]] || die "SGLang judge model directory does not exist: ${JUDGE_MODEL_PATH_ABS}"
echo "[eval] Judge ${JUDGE_LABEL}: SGLang API=${ENDPOINT}, GPUs=${GPUS}, TP=${TP_SIZE}"

METHOD_ARGS=()
read -r -a METHOD_ARGS <<< "${METHODS}"
[[ ${#METHOD_ARGS[@]} -gt 0 ]] || die "METHODS cannot be empty"

SUMMARY_ARGS=()
SUMMARY_METHODS=()
PANEL_ROOT="${OUTPUT_ROOT_ABS}/summary_panel"
if [[ "${DRY_RUN}" != 1 ]]; then mkdir -p "${PANEL_ROOT}"; fi
for method in "${METHOD_ARGS[@]}"; do
    candidates=()
    case "${method}" in
        styleforge)
            candidates=(
                "${RUN_ROOT_ABS}/styleforge/summary.json"
                "${RUN_ROOT_ABS}/styleforge/styleforge/summary.json"
            )
            ;;
        dse|rcg)
            candidates=(
                "${RUN_ROOT_ABS}/training_free_baselines/${method}/summary.json"
                "${RUN_ROOT_ABS}/styleforge/${method}/summary.json"
                "${RUN_ROOT_ABS}/${method}/summary.json"
            )
            ;;
        vanilla_sft|pso|dopsd)
            candidates=(
                "${RUN_ROOT_ABS}/training_based_baselines/${method}/summary.json"
                "${RUN_ROOT_ABS}/${method}/summary.json"
            )
            ;;
        *)
            candidates=("${RUN_ROOT_ABS}/${method}/summary.json")
            ;;
    esac

    summary=""
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            summary="${candidate}"
            break
        fi
    done
    if [[ -n "${summary}" ]]; then
        SUMMARY_ARGS+=("${summary}")
        SUMMARY_METHODS+=("${method}")
        if [[ "${DRY_RUN}" != 1 ]]; then
            mkdir -p "${PANEL_ROOT}/${method}"
            cp "${summary}" "${PANEL_ROOT}/${method}/summary.json"
        fi
        echo "[eval] ${method}: ${summary}"
    else
        echo "[eval] ${method}: n/a (searched ${candidates[*]})"
    fi
done

if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[dry-run] FLUX ${MODEL_SIZE}, judge=${JUDGE_LABEL}, summaries=${#SUMMARY_ARGS[@]}, input=${RUN_ROOT_ABS}"
    echo "[dry-run] Report: ${OUTPUT_ROOT_ABS}/evaluation_report.md"
    exit 0
fi

if [[ ${#SUMMARY_ARGS[@]} -eq 0 ]]; then
    die "no method summary.json files found under ${RUN_ROOT_ABS}"
fi

METRICS_ROOT="${OUTPUT_ROOT_ABS}/metrics"
mkdir -p "${METRICS_ROOT}"

SCORE_STATUS="not_run"
CLIP_STATUS="not_run"
FREESTYLE_STATUS="not_run"
PAIRWISE_STATUS="skipped: fewer than two methods"

# Run CLIP before loading the judge service so it has enough free GPU memory.
echo "[eval] CLIP-Text and CLIP-Style for: ${SUMMARY_METHODS[*]}"
if metric_python evaluation/metrics.py clip \
    --summary "${SUMMARY_ARGS[@]}" \
    --output-dir "${METRICS_ROOT}/clip" \
    --project-root "${ROOT}" \
    --model-path "${CLIP_MODEL_PATH}" \
    --canonical-style-metadata "${DATASET_ROOT}" \
    --device "${METRIC_DEVICE}"; then
    CLIP_STATUS="completed"
else
    CLIP_STATUS="failed"
    echo "[eval] CLIP metrics failed; report will show n/a" >&2
fi

source "${ROOT}/evaluation/judge_service.sh"
trap stop_judge_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_judge_server "${PYTHON}" "${JUDGE_MODEL_PATH_ABS}" "${JUDGE_LABEL}" \
    "${ENDPOINT}" "${OUTPUT_ROOT_ABS}/sglang_${JUDGE_SLUG}.log" \
    "${GPUS}" "${TP_SIZE}" "${STARTUP_TIMEOUT}" || die "could not start the SGLang judge service"

echo "[eval] Score for: ${SUMMARY_METHODS[*]}"
score_command_status=0
metric_python evaluation/metrics.py score \
    --summary "${SUMMARY_ARGS[@]}" \
    --output-dir "${METRICS_ROOT}/score" \
    --project-root "${ROOT}" \
    --endpoint "${ENDPOINT}" \
    --model "${JUDGE_LABEL}" \
    --concurrency "${JUDGE_API_CONCURRENCY}" || score_command_status=$?
if [[ "${score_command_status}" -eq 0 ]]; then
    SCORE_STATUS="completed"
else
    SCORE_STATUS="failed"
    echo "[eval] Score failed; report will show n/a" >&2
fi

echo "[eval] VLM-F and VLM-S for: ${SUMMARY_METHODS[*]}"
freestyle_command_status=0
metric_python evaluation/metrics.py freestyle \
    --summary "${SUMMARY_ARGS[@]}" \
    --output-dir "${METRICS_ROOT}/freestyle" \
    --project-root "${ROOT}" \
    --endpoint "${ENDPOINT}" \
    --model "${JUDGE_LABEL}" \
    --retry-failures \
    --concurrency "${FREESTYLE_API_CONCURRENCY}" || freestyle_command_status=$?
if [[ "${freestyle_command_status}" -eq 0 ]]; then
    FREESTYLE_STATUS="completed"
else
    FREESTYLE_STATUS="failed"
    echo "[eval] FreeStyle metrics failed; report will show n/a" >&2
fi

if [[ ${#SUMMARY_ARGS[@]} -ge 2 ]]; then
    echo "[eval] Pairwise Win Rate for: ${SUMMARY_METHODS[*]}"
    pairwise_args=(
        pairwise
        --run-root "${PANEL_ROOT}"
        --methods "${SUMMARY_METHODS[@]}"
        --dataset-root "${DATASET_ROOT_ABS}"
        --model-path "${JUDGE_MODEL_PATH_ABS}"
        --judge-label "${JUDGE_LABEL}"
        --output-name "${PAIRWISE_OUTPUT_NAME}"
        --backend api
    )
    echo "[eval] Pairwise: SGLang API, GPUs=${GPUS}, TP=${TP_SIZE}"
    pairwise_args+=(--endpoint "${ENDPOINT}" --model "${JUDGE_LABEL}"
        --concurrency "${PAIRWISE_API_CONCURRENCY}" --image-max-side 768)
    pairwise_command_status=0
    if ! metric_python evaluation/metrics.py "${pairwise_args[@]}"; then
        pairwise_command_status=1
    fi
    if ! metric_python evaluation/metrics.py "${pairwise_args[@]}" --summarize-only; then
        pairwise_command_status=1
    fi
    if [[ "${pairwise_command_status}" -eq 0 ]]; then
        PAIRWISE_STATUS="completed"
    else
        PAIRWISE_STATUS="failed"
        echo "[eval] Pairwise failed; report will show n/a" >&2
    fi
else
    echo "[eval] Pairwise skipped: fewer than two methods have summaries"
fi

if [[ -n "${JUDGE_SERVER_PID:-}" ]]; then
    stop_judge_server
fi

REPORT_PATH="${OUTPUT_ROOT_ABS}/evaluation_report.md"
PAIRWISE_SUMMARY_PATH="${PANEL_ROOT}/pairwise_win_rate/${PAIRWISE_OUTPUT_NAME}/summary.json"
"${PYTHON}" - "${REPORT_PATH}" "${RUN_ROOT_ABS}" "${PANEL_ROOT}" "${OUTPUT_ROOT_ABS}" \
    "${SCORE_STATUS}" "${PAIRWISE_STATUS}" "${CLIP_STATUS}" "${FREESTYLE_STATUS}" \
    "${METHODS}" "${METRICS_ROOT}" "${PAIRWISE_SUMMARY_PATH}" "${JUDGE_LABEL}" <<'PY'
import json
import math
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
panel_root = Path(sys.argv[3])
output_root = Path(sys.argv[4])
score_status, pairwise_status, clip_status, freestyle_status = sys.argv[5:9]
requested_methods = sys.argv[9].split()
metrics_root = Path(sys.argv[10])
pairwise_summary_path = Path(sys.argv[11])
judge_label = sys.argv[12]

def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def mean_from_rows(payload, key):
    rows = payload.get("records", []) if isinstance(payload, dict) else []
    values = [float(row[key]) for row in rows if isinstance(row, dict) and number(row.get(key))]
    return sum(values) / len(values) if values else None

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"

clip = read_json(metrics_root / "clip" / "clip_metrics.json") or {}
freestyle = read_json(metrics_root / "freestyle" / "summary.json") or {}
score = read_json(metrics_root / "score" / "qwen_vlm_j_metrics.json") or {}
pairwise = read_json(pairwise_summary_path) or {}

clip_records = clip.get("records", []) if isinstance(clip, dict) else []
clip_by_method = {}
for row in clip_records:
    if isinstance(row, dict):
        method = str(row.get("method") or "")
        if not method:
            method = Path(str(row.get("source_summary") or "")).parent.name
        clip_by_method.setdefault(method, []).append(row)

free_records = freestyle.get("records", []) if isinstance(freestyle, dict) else []
if not free_records:
    # FreeStyle summary.json stores aggregate groups; individual scores live
    # in records.jsonl. Keep the latest result per key when retries append rows.
    latest_free_records = {}
    try:
        with (metrics_root / "freestyle" / "records.jsonl").open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    key = row.get("record_key") or (
                        row.get("source_summary", ""), row.get("summary_record_index", index)
                    )
                    latest_free_records[key] = row
    except OSError:
        pass
    free_records = list(latest_free_records.values())
free_by_method = {}
for row in free_records:
    if isinstance(row, dict):
        method = str(row.get("method") or "")
        if not method:
            method = Path(str(row.get("source_summary") or "")).parent.name
        free_by_method.setdefault(method, []).append(row)

score_records = score.get("records", []) if isinstance(score, dict) else []
score_by_method = {}
for row in score_records:
    if isinstance(row, dict):
        method = str(row.get("method") or "")
        if not method:
            method = Path(str(row.get("source_summary") or "")).parent.name
        score_by_method.setdefault(method, []).append(row)

pairwise_by_method = {}
for row in pairwise.get("methods", []) if isinstance(pairwise, dict) else []:
    if isinstance(row, dict) and row.get("method"):
        pairwise_by_method[str(row["method"])] = row.get("average_pairwise_win_rate")

rows = []
for method in requested_methods:
    summary_exists = (panel_root / method / "summary.json").is_file()
    score_value = None
    if score_by_method.get(method):
        score_value = mean_from_rows({"records": score_by_method[method]}, "score")
        if score_value is None:
            score_value = mean_from_rows({"records": score_by_method[method]}, "qwen_vlm_j_score")
        if score_value is None:
            score_value = mean_from_rows({"records": score_by_method[method]}, "qwen_vlm_j_style_score")
    clip_text = mean_from_rows({"records": clip_by_method.get(method, [])}, "clip_text_score")
    clip_style = mean_from_rows({"records": clip_by_method.get(method, [])}, "clip_style_score")
    vlm_f = mean_from_rows({"records": free_by_method.get(method, [])}, "vlm_f_score")
    vlm_s = mean_from_rows({"records": free_by_method.get(method, [])}, "vlm_s_score")
    pairwise_value = pairwise_by_method.get(method)
    rows.append((method, summary_exists, score_value, pairwise_value, clip_text, clip_style, vlm_f, vlm_s))

lines = [
    "# Evaluation Report",
    "",
    f"- Run root: `{run_root}`",
    f"- Generated at: `{output_root}`",
    f"- Score: `{score_status}`",
    f"- SGLang judge: `{judge_label}`",
    f"- Pairwise Win Rate: `{pairwise_status}`",
    f"- CLIP metrics: `{clip_status}`",
    f"- VLM-F/VLM-S: `{freestyle_status}`",
    "",
    "| Method | Summary | Score | Pairwise Win Rate | CLIP-Text | CLIP-Style | VLM-F | VLM-S |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for method, exists, score_value, pairwise_value, clip_text, clip_style, vlm_f, vlm_s in rows:
    lines.append("| " + " | ".join([
        method,
        "available" if exists else "n/a",
        fmt(score_value), fmt(pairwise_value), fmt(clip_text), fmt(clip_style), fmt(vlm_f), fmt(vlm_s),
    ]) + " |")
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved report: {report_path}")
PY

echo "Evaluation report saved: ${REPORT_PATH}"

if [[ "${SCORE_STATUS}" != "completed" ||
      "${CLIP_STATUS}" != "completed" ||
      "${FREESTYLE_STATUS}" != "completed" ||
      "${PAIRWISE_STATUS}" == failed* ]]; then
    echo "Evaluation completed with failed metric stages; inspect ${REPORT_PATH}" >&2
    exit 1
fi
