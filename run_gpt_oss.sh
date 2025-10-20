#!/bin/bash
#SBATCH --job-name=gpt_oss_router_statistics
#SBATCH --time=03:00:00
#SBATCH --nodes=1                # total number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus-per-task=4        # Use 4 GPUs for multi-GPU acceleration
#SBATCH --output=logs/slurm-%x-%j.log  # if #SBATCH --error=... is not specified,
                                 # this will also contain stderr (error messages)

# Initialization.
export UV_LINK_MODE=copy
export UV_CACHE_DIR=$SCRATCH/uv_cache

export HF_CACHE=/$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=/$SCRATCH/hf_cache

export PYTHONBUFFERED=1

srun --environment=pytorch2506 bash -c "

START_TIME=\$(date +%s)

cd $HOME/developer/moe-router-exploration

if [[ $(uname -m) == "aarch64" ]]; then
    source venv-arm/bin/activate
else
    source venv-amd/bin/activate
fi

source set_threads.sh


accelerate launch --num_processes=4 --mixed_precision=bf16 main.py \
    --model_name "openai/gpt-oss-20b" \
    --out_data_dir "$SCRATCH/moe-router-exploration-data" \
    --data_name "cais/mmlu" \

accelerate launch --num_processes=4 --mixed_precision=bf16 main.py \
    --model_name "openai/gpt-oss-20b" \
    --out_data_dir "$SCRATCH/moe-router-exploration-data" \
    --data_name "openai/MMMLU" \

accelerate launch --num_processes=4 --mixed_precision=bf16 main.py \
    --model_name "openai/gpt-oss-20b" \
    --out_data_dir "$SCRATCH/moe-router-exploration-data" \
    --data_name "li-lab/MMLU-ProX" \

END_TIME=\$(date +%s)
ELAPSED=\$((END_TIME - START_TIME))
echo \"\"
echo \"======================================\"
echo \"Total job time: \$ELAPSED seconds (\$((ELAPSED / 60)) minutes)\"
echo \"======================================\"
"