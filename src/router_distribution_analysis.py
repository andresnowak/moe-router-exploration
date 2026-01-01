import torch
from typing import List, Dict


def get_global_router_distribution(subject_list: List[Dict]) -> torch.Tensor:
    """
    Aggregate router probabilities across all layers, experts, and subjects.

    Args:
        subject_list: List of dicts containing router distribution data

    Returns:
        Concatenated tensor of all router probabilities
    """
    all_probs = []

    for subject in subject_list:
        dict_layer_probs = subject["distributions"]
        num_layers = subject["num_layers"]
        num_experts = subject["num_experts"]
        # print(f"Processing subject with {num_layers} layers and {num_experts} experts")
        for layer in range(num_layers):
            for expert in range(num_experts):
                values = dict_layer_probs[layer][expert]
                # Handle both cases: list of tensors or single tensor
                if isinstance(values, list):
                    all_probs.extend(values)
                elif len(values) > 0:  # non-empty tensor
                    all_probs.append(values)

    # Single concatenation at the end
    probs = torch.cat(all_probs, dim=0) if all_probs else torch.tensor([])
    return probs


def get_per_expert_router_distribution(subject_list: List[Dict]) -> List[List[torch.Tensor]]:
    """
    Aggregate router probabilities per expert across all subjects.

    Args:
        subject_list: List of dicts containing router distribution data

    Returns:
        List of tensors, one per expert, containing aggregated probabilities
    """
    # Across all layers
    expert_probs = [[[] for _ in range(subject_list[0]["num_experts"])] for _ in range(subject_list[0]["num_layers"])]

    for subject in subject_list:
        dict_layer_probs = subject["distributions"]
        num_layers = subject["num_layers"]
        num_experts = subject["num_experts"]
        expert_indices = subject
        # print(f"Processing subject with {num_layers} layers and {num_experts} experts")

        for layer in range(num_layers):
            for expert in range(num_experts):
                values = dict_layer_probs[layer][expert]
                # Handle both cases: list of tensors or single tensor
                if isinstance(values, list):
                    raise NotImplementedError("List of tensors not supported in per-expert distribution")
                elif len(values) > 0:  # non-empty tensor
                    expert_probs[layer][expert].append(values)

    expert_probs = [[torch.cat(probs, dim=0) if probs else torch.tensor([]) for probs in layer_probs] for layer_probs in expert_probs] # # we will basically concatenate the subjects here
    return expert_probs


def get_per_layer_router_distribution(subject_list: List[Dict]) -> List[torch.Tensor]:
    """
    Aggregate router probabilities per layer across all subjects and experts.

    Args:
        subject_list: List of dicts containing router distribution data

    Returns:
        List of tensors, one per layer, containing aggregated probabilities
    """
    layer_probs = [[] for _ in range(subject_list[0]["num_layers"])]

    for subject in subject_list:
        dict_layer_probs = subject["distributions"]
        num_layers = subject["num_layers"]
        num_experts = subject["num_experts"]
        # print(f"Processing subject with {num_layers} layers and {num_experts} experts")

        for layer in range(num_layers):
            for expert in range(num_experts):
                values = dict_layer_probs[layer][expert]
                if isinstance(values, list):
                    layer_probs[layer].extend(values)
                elif len(values) > 0:
                    layer_probs[layer].append(values)

    layer_probs = [torch.cat(probs, dim=0) if probs else torch.tensor([]) for probs in layer_probs]
    return layer_probs
