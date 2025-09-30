#!/bin/bash
#SBATCH --job-name=deepseek_moe_router_statistics
#SBATCH --time=05:00:00
#SBATCH --nodes=1                # total number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/slurm-%x-%j.log  # if #SBATCH --error=... is not specified,
                                 # this will also contain stderr (error messages)

# Initialization.
export UV_LINK_MODE=copy
export UV_CACHE_DIR=$SCRATCH/uv_cache

export HF_CACHE=/$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=/$SCRATCH/hf_cache

srun --environment=pytorch2506 bash -c "

cd $HOME/developer/moe-router-exploration

if [[ $(uname -m) == "aarch64" ]]; then
    source venv-arm/bin/activate
else
    source venv-amd/bin/activate
fi

source set_threads.sh

python main.py \
    --model_name "deepseek-ai/deepseek-moe-16b-base" \
    --out_data_dir "$SCRATCH/moe-router-exploration-data" \
    --data_name "cais/mmlu" \

python main.py \
    --model_name "deepseek-ai/deepseek-moe-16b-base" \
    --out_data_dir "$SCRATCH/moe-router-exploration-data" \
    --data_name "TIGER-Lab/MMLU-Pro" \

python main.py \
    --model_name "deepseek-ai/deepseek-moe-16b-base" \
    --out_data_dir "$SCRATCH/moe-router-exploration-data" \
    --data_name "openai/MMMLU" \
"