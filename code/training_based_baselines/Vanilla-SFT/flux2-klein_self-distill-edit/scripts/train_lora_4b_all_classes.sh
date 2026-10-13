#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOPSD_PROJECT_DIR="${DOPSD_PROJECT_DIR:-../../D-OPSD/flux2-klein_self-distill-edit}"
cd "${PROJECT_DIR}"

DATASET_DIR="${DATASET_DIR:-../../../data/flux_training}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
if [[ -z "${GPU_IDS}" ]]; then
    echo "GPU_IDS/CUDA_VISIBLE_DEVICES cannot be empty." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export NCCL_NVLS_ENABLE NCCL_DEBUG TRANSFORMERS_VERBOSITY

if [[ "${OFFLINE:-0}" == "1" ]]; then
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
    export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DIFFUSERS_OFFLINE
fi

CONFIG_FILE="${CONFIG_FILE:-${DOPSD_PROJECT_DIR}/configs/default.yaml}"
MAIN_PORT="${MAIN_PORT:-60222}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-train_sft_lora.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DOPSD_PROJECT_DIR}/configs/z2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-../../../training_based_baselines/checkpoints/vanilla_sft_4b}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-../../../models/FLUX.2-klein-4B}"

MODEL_SIZE_TAG="${MODEL_SIZE_TAG:-4b}"

SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
EPOCHS="${EPOCHS:-1001}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1001}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-4}"
USE_LORA="${USE_LORA:-2}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
SEED="${SEED:-30}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZE_TEST="${BATCH_SIZE_TEST:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE_GEN="${LEARNING_RATE_GEN:-1e-5}"
ADAM_WEIGHT_DECAY="${ADAM_WEIGHT_DECAY:-0.0}"
VAE_DTYPE="${VAE_DTYPE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USER_NUM_PROCS="${NUM_PROCS:-}"
MAX_GPUS_PER_CLASS="${MAX_GPUS_PER_CLASS:-4}"

count_class_images() {
    local class_dir="$1"
    local image_dir="${class_dir}/target_images"
    local image_path
    local count=0

    if [[ ! -d "${image_dir}" ]]; then
        if [[ -f "${class_dir}/data.jsonl" ]]; then
            wc -l < "${class_dir}/data.jsonl"
        else
            printf '%s\n' "0"
        fi
        return 0
    fi

    for image_path in "${image_dir}"/*; do
        [[ -f "${image_path}" ]] || continue
        case "${image_path,,}" in
            *.jpg|*.jpeg|*.png|*.webp|*.bmp)
                count=$((count + 1))
                ;;
        esac
    done

    printf '%s\n' "${count}"
}

classes=()
if [[ "$#" -gt 0 ]]; then
    classes=("$@")
else
    while IFS= read -r jsonl_path; do
        class_name="${jsonl_path#${DATASET_DIR}/}"
        class_name="${class_name%/data.jsonl}"
        classes+=("${class_name}")
    done < <(find -L "${DATASET_DIR}" -mindepth 2 -maxdepth 2 -name data.jsonl -print | sort)
fi

if [[ "${#classes[@]}" -eq 0 ]]; then
    echo "No classes selected for training." >&2
    exit 1
fi

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT="${#VISIBLE_GPU_ARRAY[@]}"

echo "Sequential Vanilla SFT-LoRA direct-distilled 4B training classes:"
echo "Dataset dir: ${DATASET_DIR}"
printf ' - %s\n' "${classes[@]}"

for class_name in "${classes[@]}"; do
    class_dir="${DATASET_DIR}/${class_name}"
    data_jsonl="${class_dir}/data.jsonl"
    if [[ ! -f "${data_jsonl}" ]]; then
        echo "Missing dataset jsonl: ${data_jsonl}" >&2
        exit 1
    fi

    image_count="$(count_class_images "${class_dir}")"
    if (( image_count < 1 )); then
        echo "No target images found under: ${class_dir}/target_images" >&2
        exit 1
    fi
    if [[ -n "${USER_NUM_PROCS}" ]]; then
        class_num_procs="${USER_NUM_PROCS}"
    else
        class_num_procs="${image_count}"
        if (( class_num_procs > MAX_GPUS_PER_CLASS )); then
            class_num_procs="${MAX_GPUS_PER_CLASS}"
        fi
    fi
    if (( class_num_procs < 1 )); then
        echo "NUM_PROCS must be positive, got ${class_num_procs}." >&2
        exit 1
    fi
    if (( class_num_procs > MAX_GPUS_PER_CLASS )); then
        echo "NUM_PROCS=${class_num_procs} exceeds MAX_GPUS_PER_CLASS=${MAX_GPUS_PER_CLASS}." >&2
        exit 1
    fi
    if (( class_num_procs > VISIBLE_GPU_COUNT )); then
        echo "NUM_PROCS=${class_num_procs} requires ${class_num_procs} visible GPUs, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${VISIBLE_GPU_COUNT}." >&2
        exit 1
    fi

    exp_prefix="${EXP_PREFIX:-vanilla_sft_lora_direct_distilled_${NUM_TRAINING_STEPS}steptrain_${MODEL_SIZE_TAG}}"
    exp_name="${EXP_NAME:-${exp_prefix}_${class_name}_bsz${BATCH_SIZE}_lora_lr${LEARNING_RATE_GEN}}"

    cmd=(
        "${ACCELERATE_BIN}" launch
        --config_file "${CONFIG_FILE}"
        --main_process_port "${MAIN_PORT}"
        --num_processes "${class_num_procs}"
        "${PYTHON_SCRIPT}"
        --deepspeed-config "${DEEPSPEED_CONFIG}"
        --output-dir "${OUTPUT_DIR}"
        --exp-name "${exp_name}"
        --sample-steps "${SAMPLE_STEPS}"
        --checkpoint-steps "${CHECKPOINT_STEPS}"
        --epochs "${EPOCHS}"
        --max-train-steps "${MAX_TRAIN_STEPS}"
        --pretrained_model "${PRETRAINED_MODEL}"
        --num-training-steps "${NUM_TRAINING_STEPS}"
        --use-lora "${USE_LORA}"
        --lora-rank "${LORA_RANK}"
        --lora-alpha "${LORA_ALPHA}"
        --data-path-train-jsonl "${data_jsonl}"
        --data-path-test-jsonl "${data_jsonl}"
        --seed "${SEED}"
        --mixed-precision "${MIXED_PRECISION}"
        --batch-size "${BATCH_SIZE}"
        --batch-size-test "${BATCH_SIZE_TEST}"
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
        --learning-rate-gen "${LEARNING_RATE_GEN}"
        --adam-weight-decay "${ADAM_WEIGHT_DECAY}"
        --enable-gc
        --vae-dtype "${VAE_DTYPE}"
        --num-workers "${NUM_WORKERS}"
    )

    echo "============================================================"
    echo "Class: ${class_name}"
    echo "Pretrained model: ${PRETRAINED_MODEL}"
    echo "Target images: ${image_count}"
    echo "Num processes: ${class_num_procs}"
    echo "Dataset: ${data_jsonl}"
    echo "Experiment: ${exp_name}"
    echo "============================================================"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'Dry run command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
done
