# MoE Router Exploration

Utilities for inspecting Mixture-of-Experts router behavior, aggregating routing-score distributions, and measuring the downstream effect of zeroing low-probability expert contributions.

The repo centers on two workflows:

1. Collect router statistics from MoE language models across evaluation datasets.
2. Re-run evaluation with router interventions that suppress experts below a probability threshold.

Supported model families in the current codebase:

- `deepseek-ai/deepseek-moe-16b-base`
- `openai/gpt-oss-20b`
- `allenai/OLMoE-1B-7B-0125`
- `arcee-ai/Trinity-Nano-Base` and `arcee-ai/Trinity-Mini`

## What is here

- `main_router_prob_distribution.py`: logs per-token router probabilities and saves compact distribution files.
- `main_multilingual.py`: older routing-statistics path for multilingual datasets.
- `eval.py`: runs `lm-eval` with optional router-threshold interventions.
- `generate_router_distribution_plots.py`: batch-generates plots from saved router distributions.
- `src/router_logger.py`: model-specific hook logic for extracting routing decisions.
- `src/router_intervention.py`: model-specific forward patches for zeroing low-probability experts.
- `src/router_distribution_analysis.py` and `src/visualize.py`: aggregation and plotting helpers.
- `run_router_distribution.sh` and `router_zero_experts_eval.sh`: SLURM-oriented launchers.
- `MoE_Router_Distribution_Exploration_revised.pdf`: accompanying write-up.

## Environment

Python target:

- `>=3.12,<3.13`

This repo is set up for `uv`, and the project intentionally avoids pinning `torch` in `pyproject.toml` because the expected runtime is usually a prebuilt container or system environment with PyTorch already installed.

Basic setup:

```bash
uv sync
```

If you are working outside the intended container/HPC environment, make sure a compatible `torch` build is installed before running the scripts.

Useful development commands:

```bash
uv run ruff check .
uv run ty check
```

## Main workflow

### 1. Collect router probability distributions

`main_router_prob_distribution.py` loads a supported MoE model, attaches routing hooks, iterates over a dataset, and stores per-layer/per-expert probability tensors for the experts that made the model top-k.

Example:

```bash
accelerate launch --num_processes=4 main_router_prob_distribution.py \
  --model_name openai/gpt-oss-20b \
  --data_name cais/mmlu \
  --out_data_dir ./data/router_prob_distribution \
  --max_examples 128 \
  --overwrite
```

Supported datasets in the script today:

- `cais/mmlu`
- `TIGER-Lab/MMLU-Pro`
- `openai/MMMLU`
- `li-lab/MMLU-ProX`
- `Rowan/hellaswag`
- `allenai/ai2_arc`
- `allenai/winogrande`

Output layout:

```text
<out_data_dir>/
  <model-name-with-slashes-replaced>/
    <dataset-name-with-slashes-replaced>/
      <subject>/
        <language>/
          routing_distributions.pt
```

Those files are later consumed by `src/router_loading_utils.py` and `generate_router_distribution_plots.py`.

### 2. Generate plots

Once distributions are saved, generate the standard analysis figures:

```bash
uv run python generate_router_distribution_plots.py
```

By default the script writes plots under `./plots` when run as checked into the repo. Internally it expects routing data under:

```text
$SCRATCH/moe-router-exploration-data/router_prob_distribution
```

unless you edit the helper paths in the script.

Generated figures include:

- global router-score histograms
- per-layer router-score distributions
- per-expert distributions for selected layers
- deactivated-expert counts across threshold sweeps

### 3. Evaluate router-threshold interventions

`eval.py` runs `lm-eval` and optionally patches the MoE routing path so any selected expert whose weight falls below `--prob_threshold` contributes `0`.

Example:

```bash
accelerate launch --num_processes 4 --multi_gpu eval.py \
  --model_path openai/gpt-oss-20b \
  --model_type gptoss \
  --prob_threshold 0.05 \
  --tasks hellaswag,arc_easy,winogrande,mmlu \
  --batch_size 64 \
  --output_path eval_results/moe_router_distribution_eval/gptoss/results_0.05.json
```

Supported `--model_type` values come from `src/router_intervention.py`:

- `deepseek-moe`
- `gptoss`
- `olmoe`
- `trinity`

Behavior notes:

- If `--prob_threshold` is `0.0`, no intervention is applied.
- Existing result files are reused unless `--overwrite` is passed.
- `babilong` is expanded into `babilong_qa1` through `babilong_qa5`.

## SLURM / cluster usage

The shell scripts in the repo are written for a multi-GPU SLURM environment and assume:

- `srun`/`sbatch` are available

Useful entry points:

- `run_router_distribution.sh`: submits routing-distribution jobs for several models and datasets.
- `router_zero_experts_eval.sh`: sweeps probability thresholds for evaluation.

## Notes on implementation

- Routing extraction is model-specific because each architecture exposes router outputs differently.
- `gpt-oss` support depends on hooking the `GptOssMLP` path; the code comments note that hub kernels should be disabled for this route with `USE_HUB_KERNELS=OFF`.
- The intervention code does not renormalize probabilities after thresholding; it zeros the selected experts below threshold and leaves the remaining weights unchanged to match the experiment design.
- `eval.py` monkey-patches `AutoModelForCausalLM.from_pretrained` to ignore `quantization_config` for the GPT-OSS loading path used by `lm-eval`.

## Repo outputs

Current committed outputs include example evaluation JSON files under:

- `eval_results/moe_router_distribution_eval/...`

These are useful as references for expected result format and naming.
