#!/usr/bin/env bash
# Run a public entry point once per FLUX size, in separate processes so model
# memory is released between runs. Judge-set recursion stays within one size.

model_path_template() {
    local value="$1"
    local size="$2"
    value="${value//\{size\}/${size}}"
    value="${value//\{SIZE\}/${size^^}}"
    printf '%s\n' "${value}"
}

runtime_dispatch_models() {
    local stage="$1"
    local entry="$2"
    [[ "${FLUX_MODEL_CHILD:-0}" == "1" ]] && return 0

    local normalized="${MODEL_SIZES//,/ }"
    local -a sizes=() models=() runs=() outputs=()
    read -r -a sizes <<< "${normalized}"
    [[ ${#sizes[@]} -gt 0 ]] || { echo 'MODEL_SIZES cannot be empty' >&2; exit 2; }

    source "${ROOT}/scripts/output_paths.sh"
    local size name index previous
    for size in "${sizes[@]}"; do
        case "${size}" in
            4b|9b) ;;
            *) echo "Unsupported MODEL_SIZE: ${size}; use 4b or 9b" >&2; exit 2 ;;
        esac
    done
    if [[ ${#sizes[@]} -gt 1 ]]; then
        for name in MODEL RUN_ROOT OUTPUT_ROOT; do
            [[ "${name}" == MODEL && "${stage}" == evaluation ]] && continue
            case "${!name}" in
                *'{size}'*|*'{SIZE}'*) ;;
                *) echo "${name} must contain {size} or {SIZE} for both models; use MODEL_SIZE=4b or MODEL_SIZE=9b for one model." >&2; exit 2 ;;
            esac
        done
    fi

    # Validate all destinations before any API request or model execution.
    for size in "${sizes[@]}"; do
        models+=("$(model_path_template "${MODEL}" "${size}")")
        assert_output_under_outputs "$(model_path_template "${RUN_ROOT}" "${size}")" RUN_ROOT || exit 1
        runs+=("${OUTPUT_PATH_ABS}")
        assert_output_under_outputs "$(model_path_template "${OUTPUT_ROOT}" "${size}")" OUTPUT_ROOT || exit 1
        for previous in "${outputs[@]}"; do
            [[ "${previous}" != "${OUTPUT_PATH_ABS}" ]] || { echo "Model output directories overlap: ${OUTPUT_PATH_ABS}" >&2; exit 2; }
        done
        outputs+=("${OUTPUT_PATH_ABS}")
    done
    for index in "${!sizes[@]}"; do
        MODEL_SIZE="${sizes[index]}" MODEL="${models[index]}" RUN_ROOT="${runs[index]}" \
            runtime_preflight "${stage}" || exit "$?"
    done

    local overall_status=0 child_status
    local contract_source="${STYLEFORGE_CONTRACT_SOURCE_ROOT:-}"
    echo "[runtime] FLUX models: ${sizes[*]} (sequential)"
    for index in "${!sizes[@]}"; do
        size="${sizes[index]}"
        echo "[runtime] FLUX ${size}: ${models[index]} -> ${outputs[index]}"
        child_status=0
        env FLUX_MODEL_CHILD=1 MODEL_SIZES="${size}" MODEL_SIZE="${size}" \
            MODEL="${models[index]}" RUN_ROOT="${runs[index]}" OUTPUT_ROOT="${outputs[index]}" \
            STYLEFORGE_CONTRACT_SOURCE_ROOT="${contract_source}" \
            bash "${entry}" || child_status=$?
        case "${child_status}" in
            130|143) exit "${child_status}" ;;
        esac
        if [[ "${child_status}" -ne 0 ]]; then
            overall_status=1
            echo "[runtime] FLUX ${size} failed (exit ${child_status}); continuing with remaining models." >&2
        elif [[ "${stage}" == styleforge && -z "${contract_source}" ]]; then
            # Later sizes copy these same contracts; schema compilation remains
            # local to each run and does not make a second extraction request.
            contract_source="${runs[index]}/extraction"
        fi
    done
    if [[ "${overall_status}" -eq 0 ]]; then
        echo "[runtime] ${stage} completed for FLUX ${sizes[*]} (DRY_RUN=${DRY_RUN})."
    fi
    exit "${overall_status}"
}
