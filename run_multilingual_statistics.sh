source set_threads.sh

export CUDA_VISIBLE_DEVICES=0

python main_multilingual.py \
    --model_name "openai/gpt-oss-20b" \
    --out_data_dir "$SCRATCH/moe-router-exploration-data" \
    --data_name "cais/mmlu" \
