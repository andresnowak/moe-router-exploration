#!/bin/bash

ACCOUNT_NAME=${1:-"def-ai"}

declare -A MODEL_TYPES=(
    # ["deepseek_moe"]="deepseek-ai/deepseek-moe-16b-base"
    ["gpt_oss"]="openai/gpt-oss-20b"
    # ["olmoe"]="allenai/OLMoE-1B-7B-0125-Instruct"
    # ["trinity"]="arcee-ai/Trinity-Nano-Base"
)

DATASETS=(
    "cais/mmlu"
    "TIGER-Lab/MMLU-Pro"
)

for MODEL_KEY in "${!MODEL_TYPES[@]}"; do
    MODEL_NAME="${MODEL_TYPES[$MODEL_KEY]}"

    # Generate dataset commands
    DATASET_COMMANDS=""
    for DATASET in "${DATASETS[@]}"; do
        DATASET_COMMANDS+="accelerate launch \\
        --num_processes=4 \\
        --mixed_precision=bf16 \\
        main_router_prob_distribution.py \\
        --model_name \"${MODEL_NAME}\" \\
        --out_data_dir \"\$SCRATCH/moe-router-exploration-data/router_prob_distribution\" \\
        --data_name \"${DATASET}\" \\
        --overwrite

"
    done

    # Submit job for the current model
    SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash

#SBATCH --job-name=${MODEL_KEY}_router_distribution
#SBATCH --time=05:00:00
#SBATCH --account=${ACCOUNT_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --output=logs/${MODEL_KEY}_router_distribution-%x-%j.log
#SBATCH --error=logs/${MODEL_KEY}_router_distribution-%x-%j.err

# Initialization
export UV_LINK_MODE=copy
export UV_CACHE_DIR=\$SCRATCH/uv_cache

export HF_CACHE=/\$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=/\$SCRATCH/hf_cache

export USE_HUB_KERNELS=OFF # So as to not use the megablocks kernel for GPTOss (this was added on version 5.* it seems, so it doing nothing for us now)

export PYTHONBUFFERED=1

srun --environment=pytorch2506 --export=all bash -c "

START_TIME=\\\$(date +%s)

cd \$HOME/developer/moe-router-exploration

if [[ \$(uname -m) == \"aarch64\" ]]; then
    source venv-arm/bin/activate
else
    source venv-amd/bin/activate
fi

source set_threads.sh

${DATASET_COMMANDS}
END_TIME=\\\$(date +%s)
ELAPSED=\\\$((END_TIME - START_TIME))
echo \\"\\"
echo \\"======================================\\"
python -c 'print(f\\"Total job time: '\\\$ELAPSED' seconds ({'\\\$ELAPSED' // 60} minutes)\\")'
echo \\"======================================\\"
"
EOF
)

    echo "Generated SBATCH script for ${MODEL_KEY}:"
    echo "----------------------------------------"
    echo "$SBATCH_SCRIPT"
    echo "----------------------------------------"

    echo "$SBATCH_SCRIPT" | sbatch

    echo "Submitted job for ${MODEL_KEY}: ${MODEL_NAME}"
done