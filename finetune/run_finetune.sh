#!/usr/bin/env bash
# run_finetune.sh
#
# Fine-tunes a causal LM on every JSON task file found under data/cot/
# using Accelerate + LoRA (PEFT).
#
# Usage:
#   export HF_TOKEN="<your-huggingface-token>"
#   export WANDB_API_KEY="<your-wandb-key>"
#   bash run_finetune.sh
#
# Each task produces a checkpoint under:
#   models/base_ckpt_<MODEL>_<TASK_NAME>/

set -euo pipefail


# Configuration — edit these to match your setup

MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"   # or a local path
DATA_DIR="data/cot"                               # directory containing per-task JSON files
OUTPUT_ROOT="models"
LOG_FILE="cot_all_tasks_runs.log"
MAX_RETRIES=5
ACCELERATE_CONFIG="configs/acc_tuner_config.yaml"
ACCELERATE_PORT=29501


# WandB login (skipped if WANDB_API_KEY is unset)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --cloud --relogin "$WANDB_API_KEY"
fi


# Helpers

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG_FILE"; }

run_task() {
    local task_name="$1"
    local data_path="$2"
    local output_dir="$3"
    local retries_left="$4"

    if [[ "$retries_left" -eq 0 ]]; then
        log "ERROR: Max retries exceeded for task: $task_name"
        return 1
    fi

    log "Fine-tuning task: $task_name  (attempt $((MAX_RETRIES - retries_left + 1))/$MAX_RETRIES)"

    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --main_process_port "$ACCELERATE_PORT" \
        src/instruction_tuner.py \
            --dataset_name_or_path "$data_path" \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --load_data_from_disk \
            --hf_access_token "${HF_TOKEN:-}" \
            --torch_dtype bfloat16 \
            --max_seq_length 4096 \
            --learning_rate 2e-5 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --preprocessing_num_workers 12 \
            --use_peft \
            --peft_lora_r 8 \
            --peft_lora_alpha 32 \
            --peft_lora_dropout 0.1 \
            --peft_target_modules "q_proj,k_proj,v_proj,o_proj" \
            --seed 23 \
            --num_train_epochs 1 \
            --gradient_accumulation_steps 1 \
            --gradient_checkpointing \
            --weight_decay 0.01 \
            --lr_scheduler_type cosine \
            --with_tracking \
            --report_to wandb \
            --output_dir "$output_dir"

    local exit_code=$?
    if [[ "$exit_code" -ne 0 ]]; then
        log "WARNING: Fine-tuning failed for task: $task_name — retrying..."
        run_task "$task_name" "$data_path" "$output_dir" "$((retries_left - 1))"
    else
        log "Done: $task_name -> $output_dir"
    fi
}


# Main loop — iterate over every JSON task file

log "--------------------------------------------"
log "Started CoT Fine-Tuning Stage"

for task_file in "$DATA_DIR"/*.json; do
    base_name="$(basename "$task_file")"
    task_name="${base_name%.json}"
    output_dir="${OUTPUT_ROOT}/base_ckpt_$(basename "$MODEL_NAME_OR_PATH")_${task_name}"

    if [[ -d "$output_dir" ]]; then
        log "Skipping (already done): $output_dir"
        continue
    fi

    run_task "$task_name" "$task_file" "$output_dir" "$MAX_RETRIES"
done

log "All tasks complete."
