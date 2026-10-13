#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash training_based_baselines/run_flux_training.sh <vanilla_sft|pso|dopsd> <4b|9b> [style-task ...]

The command dispatches to the original method-specific FLUX training script.
Set DATASET_DIR, PRETRAINED_MODEL, OUTPUT_DIR, and other method variables to
override its relative defaults.
EOF
}

method="${1:-}"
model_size="${2:-}"
if [[ -z "${method}" || -z "${model_size}" || "${method}" == "-h" || "${method}" == "--help" ]]; then
    usage
    exit 2
fi

case "${method}" in
    vanilla_sft|pso|dopsd) ;;
    *) usage >&2; exit 2 ;;
esac
case "${model_size}" in
    4b|9b) ;;
    *) usage >&2; exit 2 ;;
esac

case "${method}" in
    vanilla_sft) method_dir="Vanilla-SFT" ;;
    pso) method_dir="PSO" ;;
    dopsd) method_dir="D-OPSD" ;;
esac
script="${ROOT}/training_based_baselines/${method_dir}/flux2-klein_self-distill-edit/scripts/train_lora_${model_size}_all_classes.sh"
[[ -f "${script}" ]] || { echo "Missing training script: ${script}" >&2; exit 1; }
shift 2
exec bash "${script}" "$@"
