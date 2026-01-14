import argparse
import numpy as np
import os
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import json

from src.router_intervention import create_router_intervention, MODEL_INTERVENTION_CLASSES


# Monkey-patch AutoModelForCausalLM.from_pretrained to skip quantization_config
_original_from_pretrained = AutoModelForCausalLM.from_pretrained

@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    kwargs.pop('quantization_config', None) # Remove quantization_config if present just to fix the problem with initializing gpt_oss in the lm-eval-harness HFLM _create_model function, as gptoss is already an mxfp4 model but then we are also passing the dict quantization config to the from_pretrained method which causes an error
    return _original_from_pretrained.__func__(cls, *args, **kwargs)

AutoModelForCausalLM.from_pretrained = _patched_from_pretrained


MODEL_LIST = list(MODEL_INTERVENTION_CLASSES.keys())

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Llama models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (HF name)")
    parser.add_argument("--model_type", type=str, choices=MODEL_LIST, default=None,
                       help="Model type for router intervention, if None, no intervention is applied")
    parser.add_argument("--prob_threshold", type=float, default=0.0, help="Probability threshold for router intervention, if 0.0, no intervention is applied")
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy,winogrande", help="Comma-separated tasks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default="eval_results.json")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results")

    return parser.parse_args()

def main():
    args = parse_args()

    # Pass model path as string to HFLM for proper accelerate support
    lm_obj = HFLM(
        pretrained=args.model_path,
        # dtype="bfloat16",
        dtype="auto",
        batch_size=args.batch_size,
        trust_remote_code=True,
        # parallelize=True # This is for dividing the model across gpus naively
    )

    if args.model_type and args.prob_threshold > 0.0:
        router_intervention = create_router_intervention(
            model=lm_obj.model, # modified in place
            model_type=args.model_type,
            prob_threshold=args.prob_threshold
        )
    # Run tasks
    tasks = args.tasks.split(",")

    if "babilong" in tasks:
        tasks.remove("babilong")
        tasks.extend(["babilong_qa1", "babilong_qa2", "babilong_qa3", "babilong_qa4", "babilong_qa5"]) # 6 to 20 only have 0k length
    
    task_metadata = {
        "ruler": {"pretrained": args.model_path, "max_seq_lengths": [4096]},
        "babilong": {"pretrained": args.model_path, "max_seq_lengths": "2k"},
    }

    # Evaluate tasks with different metadata
    all_results = {"results": {}, "configs": {}}

    if os.path.exists(args.output_path) and not args.overwrite:
        with open(args.output_path, "r") as f:
            all_results = json.load(f)
        
        if "model_info" not in all_results:
            if lm_obj.accelerator.is_main_process:
                print("Warning: Loaded results do not contain model info, terminating to avoid confusion.")
            return
        elif all_results["model_info"]["model_path"] != args.model_path or all_results["model_info"]["model_type"] != args.model_type:
            if lm_obj.accelerator.is_main_process:
                print("Warning: Loaded results are for a different model or model type than the current evaluation. Terminating to avoid confusion.")
            return

        print(f"Loaded existing results from {args.output_path}, use --overwrite to re-evaluate.")


    for task in tasks:
        if task in all_results["results"]:
            if lm_obj.accelerator.is_main_process:
                print(f"Skipping {task}, already evaluated.")
            continue
        
        if lm_obj.accelerator.is_main_process:
            print(f"Evaluating task: {task}")

        metadata = task_metadata.get(task, {"pretrained": args.model_path})
        
        results = evaluator.simple_evaluate(
            model=lm_obj,
            tasks=[task],  # Evaluate one task at a time
            batch_size=args.batch_size,
            metadata=metadata,
            # apply_chat_template="gptoss"==args.model_type,
        )
        
        if results:
            all_results["results"].update(results.get("results", {}))
            all_results["configs"].update(results.get("configs", {}))

    if lm_obj.accelerator.is_main_process:
        if not all_results["results"]:
            print("Warning: No results returned from evaluation")
            return

        # Save and print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for task in tasks:
            if task in all_results["results"]:
                print(f"\n{task}:")
                for metric, value in all_results["results"][task].items():
                    if not metric.startswith("alias"):
                        print(f"  {metric}: {value}")

        def handle_non_serializable(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        all_results["model_info"] = {"model_type": args.model_type, "model_path": args.model_path, "prob_threshold": args.prob_threshold}

        with open(args.output_path, "w") as f:
            json.dump(
                all_results, f, indent=2, default=handle_non_serializable, ensure_ascii=False
            )
        print(f"\nResults saved to: {args.output_path}")

if __name__ == "__main__":
    main()