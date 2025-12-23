import os
import argparse
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig
)
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.router_logger import DeepSeekMoELogger, RoutingDistributionTracker, GPTOssMoELogger, MoELogger
from src.utils import mmlu_loader, mmlu_pro_loader, mmmlu_loader, mmlu_pro_x_loader


def get_router_statistics(
    model: PreTrainedModel,
    config: PretrainedConfig,
    tok: PreTrainedTokenizer,
    routing_logger: MoELogger,
    loader: DataLoader,
    accelerator: Accelerator,
) -> RoutingDistributionTracker:
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
    tracker = RoutingDistributionTracker(model, config, tok, routing_logger.num_layers, routing_logger.num_experts)

    for batch in loader:
        # Move batch to device since we're not using accelerator.prepare(loader)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        with torch.no_grad():
            _ = model(**batch, use_cache=False, return_dict=True)

        # Update tracker with one batch at a time
        tracker.update(batch, routing_logger.routing_logs)

    return tracker


def main(model_name: str, data_name: str, max_examples: int | None, out_dir: str, overwrite: bool = False):
    # 1) load config & model with router‐logits enabled
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
    ).eval()

    tok = AutoTokenizer.from_pretrained(model_name)

    model_config = model.config
    model = accelerator.prepare(model)

    routing_logger = None
    if "deepseek-ai/deepseek-moe" in model_name:
        routing_logger = DeepSeekMoELogger(model, model_config, tok)
    elif "openai/gpt-oss-20b" in model_name:
        routing_logger = GPTOssMoELogger(model, model_config, tok)
    else:
        raise Exception(f"{model_name} is not a valid model name")

    loader_dict = None
    if "cais/mmlu" in data_name:
        loader_dict = mmlu_loader(tok, max_examples, 16)
    elif "TIGER-Lab/MMLU-Pro" in data_name:
        loader_dict = mmlu_pro_loader(tok, max_examples, 16)
    elif "openai/MMMLU" in data_name:
        loader_dict = mmmlu_loader(tok=tok, max_examples=max_examples, batch_size=16)
    elif "li-lab/MMLU-ProX" in data_name:
        loader_dict = mmlu_pro_x_loader(
            tok=tok, max_examples=max_examples, batch_size=16
        )
    else:
        raise Exception(f"{data_name} is not a valid data name")

    if accelerator.is_main_process:
        print(f"Starting on {accelerator.num_processes} processes...")

    items = list(loader_dict.items())
    for i, ((subject, language), loader) in enumerate(items):
        if i % accelerator.num_processes != accelerator.process_index:
            print(f"Not my part {accelerator.process_index}")
            continue

        # Check if output folder already exists
        output_folder = os.path.join(
            out_dir,
            f"{model_name.replace('/', '-')}/{data_name.replace('/', '-')}/{language}/{subject}",
        )
        if os.path.exists(output_folder) and not overwrite:
            print(f"Skipping {subject} ({language}) - output folder already exists: {output_folder}")
            continue

        # Don't use accelerator.prepare on loader - we're manually splitting by subject
        # We move them manually instead
        # loader = accelerator.prepare(loader)

        start_time = time.time()

        # 2) gather routing statistics
        tracker = get_router_statistics(model, model_config, tok, routing_logger, loader, accelerator)

        # 3) save results
        distributions_path = os.path.join(
            out_dir,
            f"{model_name.replace('/', '-')}/{data_name.replace('/', '-')}/{language}/{subject}/routing_distributions.pt",
        )
        os.makedirs(os.path.dirname(distributions_path), exist_ok=True)

        tracker.save_distributions(output_folder)

        print(f"Done. Processed {len(loader.dataset) or 'all'} examples in {time.time() - start_time} seconds.")
        print(f" • raw distributions saved to {distributions_path}")


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders",
    )

    args = parser.parse_args()
    main(args.model_name, args.data_name, args.max_examples, args.out_data_dir, args.overwrite)
