import os
import torch
from typing import Dict, List

def create_dataset_routing_statistics(root_path: str) -> List[Dict[str, List[torch.Tensor] | int]]:
    list_tensors = []

    for dirpath, _, filenames in os.walk(root_path):
        if "router_distributions.pt" in filenames:
            candidate = os.path.join(dirpath, "router_distributions.pt")

            rel_path = os.path.relpath(candidate, root_path)
            print(f"Found: {rel_path}")

            dirs = rel_path.split("/")[:-1]  # should have subject, language and file
            assert len(dirs) == 2

            print(os.path.join(root_path, rel_path))
            list_tensor = torch.load(os.path.join(root_path, rel_path)) # [L][E][tensor of probs (length for each expert is max total amount of tokens in dataset)]
            list_tensors.append(list_tensor)
        else:
            continue

    return list_tensors