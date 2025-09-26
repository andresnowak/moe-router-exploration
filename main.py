import os
import argparse
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from torch.utils.data import DataLoader

from src.router_logger import DeepSeekMoELogger, RoutingStatisticsTracker, GPTOssMoELogger, MoELogger
from src.utils import mmlu_loader, mmlu_pro_loader


def get_router_statistics(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    routing_logger: MoELogger,
    loader: DataLoader,
) -> RoutingStatisticsTracker:
    """
    Log routing stats for a dataset.

    Args:
        model:               the MoE model (with output_router_logits=True)
        tok:                 its tokenizer
        routing_logger:      DeepSeekMoELogger attached to model
        loader:              the DataLoader

    Returns:
        A RoutingStatisticsTracker with counts filled in.
    """
    tracker = RoutingStatisticsTracker(model, tok, routing_logger.num_layers, routing_logger.num_experts)

    for batch in loader:
        batch = {
            k: v.to(model.device) for k, v in batch.items()
        }  # Move tensors to device

        with torch.no_grad():
            _ = model(**batch, use_cache=False, return_dict=True)

        # Update tracker with one batch at a time
        tracker.update(batch, routing_logger.routing_logs)

    return tracker


def main(model_name: str, data_name: str, max_examples: int | None, out_dir: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 1) load config & model with router‐logits enabled
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    tok = AutoTokenizer.from_pretrained(model_name)

    routing_logger = None
    if "deepseek-ai/deepseek-moe" in model_name:
        routing_logger = DeepSeekMoELogger(model, tok)
    elif "openai/gpt-oss-20b":
        routing_logger = GPTOssMoELogger(model, tok)
    else:
        raise Exception(f"{model_name} is not a valid model name")

    loader_dict = None
    if "cais/mmlu" in data_name:
        loader_dict = mmlu_loader(tok, max_examples, 16)
    elif "TIGER-Lab/MMLU-Pro" in data_name:
        loader_dict = mmlu_pro_loader(tok, max_examples, 16)
    else:
        raise Exception(f"{data_name} is not a valid data name")

    print("Starting...")

    for (subject, language), loader in loader_dict.items():
        start_time = time.time()

        # 2) gather routing statistics
        tracker = get_router_statistics(model, tok, routing_logger, loader)

        # 3) save results
        counter_path = os.path.join(
            out_dir,
            f"{model_name.replace('/', '-')}/{data_name.replace('/', '-')}/{language}/{subject}/routing_counter.pt",
        )
        sparse_path = os.path.join(
            out_dir,
            f"{model_name.replace('/', '-')}/{data_name.replace('/', '-')}/{language}/{subject}/routing_sparse.pt",
        )
        os.makedirs(os.path.dirname(counter_path), exist_ok=True)
        os.makedirs(os.path.dirname(sparse_path), exist_ok=True)

        tracker.save_counter(counter_path)
        tracker.save_sparse(sparse_path)

        print(f"Done. Processed {len(loader.dataset) or 'all'} examples in {time.time() - start_time} seconds.")
        print(f" • raw Counter → {counter_path}")
        print(f" • sparse COO  → {sparse_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log MoE routing stats")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="huggingface model path or path",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="If set, only process this many dataset examples",
    )
    parser.add_argument(
        "--out_data_dir",
        type=str,
        required=True,
        default="./data",
        help="path to output directory",
    )
    parser.add_argument(
        "--data_name", type=str, required=True, help="Huggingface data path or path"
    )

    args = parser.parse_args()
    main(args.model_name, args.data_name, args.max_examples, args.out_data_dir)
