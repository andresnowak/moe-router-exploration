#!/bin/bash
#SBATCH --job-name=moe_router_distribution_eval
#SBATCH --time=05:00:00
#SBATCH --nodes=1                # total number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus-per-task=4        # Use 4 GPUs for multi-GPU acceleration
#SBATCH --output=eval_logs/router_distribution_eval_-%x-%j.log
#SBATCH --error=eval_logs/router_distribution_eval_-%x-%j.err

# Initialization.
set -x

export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

declare -A MODELS=(
  # ["olmoe"]="allenai/OLMoE-1B-7B-0924"
  ["deepseek-moe"]="deepseek-ai/deepseek-moe-16b-base"
  # ["trinity"]="arcee-ai/Trinity-Nano-Base"
  # ["gptoss"]="openai/gpt-oss-20b"
)

PROB_THRESHOLDS=(0.0 0.1 0.01)

for key in "${!MODELS[@]}"; do
  for prob_threshold in "${PROB_THRESHOLDS[@]}"; do
    export MODEL_TYPE=${key}
    export MODEL_ID=${MODELS[$key]}

    export BATCH_SIZE=16
    export TASKS="hellaswag,arc_easy,winogrande,mmlu"
    export PROB_THRESHOLD=${prob_threshold}
    export OUTPUT_PATH="eval_results/moe_router_distribution_eval/${MODEL_TYPE}/${MODEL_TYPE}_eval_results_${PROB_THRESHOLD}.json"

    export MASTER_PORT=6850

    srun --environment=pytorch2506 bash -lc "

      START_TIME=\$(date +%s)

      cd $HOME/developer/moe-router-exploration

      if [[ \$(uname -m) == \"aarch64\" ]]; then
          source venv-arm/bin/activate
      else
          source venv-amd/bin/activate
      fi

      accelerate launch \
        --num_processes 4 \
        --multi_gpu \
        eval.py \
        --model_path ${MODEL_ID} \
        --model_type ${MODEL_TYPE} \
        --batch_size ${BATCH_SIZE} \
        --tasks ${TASKS} \
        --output_path ${OUTPUT_PATH} \
        --prob_threshold ${PROB_THRESHOLD}
    "
  done
done