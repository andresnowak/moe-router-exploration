import os
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from torch.utils.data import DataLoader

from src.router_logger import DeepSeekMoELogger, RoutingStatisticsTracker
from src.utils import mmlu_loader


def get_router_statistics(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    routing_logger: DeepSeekMoELogger,
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
    tracker = RoutingStatisticsTracker(model, tok)

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
    else:
        raise Exception(f"{model_name} is not a valid model name")

    loader = None
    if "cais/mmlu" in data_name:
        loader = mmlu_loader(tok, max_examples, 8)
    else:
        raise Exception(f"{data_name} is not a valid data name")

    # 2) gather routing statistics over MMLU
    tracker = get_router_statistics(model, tok, routing_logger, loader)

    # 3) save results
    counter_path = os.path.join(out_dir, "mmlu_routing_counter.pt")
    sparse_path = os.path.join(out_dir, "mmlu_routing_sparse.pt")

    tracker.save_counter(counter_path)
    tracker.save_sparse(sparse_path)

    print(f"Done. Processed {max_examples or 'all'} examples.")
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
