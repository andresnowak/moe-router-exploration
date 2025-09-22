from typing import Dict, Tuple, List
import re

import torch
from transformers.modeling_utils import PreTrainedModel


class DeepSeekMoELogger:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,  # to get vocab_size
    ):
        self.model = model
        cfg = model.config

        # shapes for your stats tensor:
        self.vocab_size = tokenizer.vocab_size
        self.num_layers = cfg.num_hidden_layers
        self.top_k_experts = cfg.num_experts_per_tok
        self.num_experts = cfg.n_routed_experts

        # routing_logs: holds the last-forward for each gate path
        self.routing_logs: Dict[str, Dict[str, torch.Tensor | int]] = {}

        # register once
        self.hook_handles = self._register_hooks()

    def _make_hook(self, path: str, layer: int):
        def hook(module, inputs, outputs):
            # outputs[0] = expert_indices
            idx = outputs[0].detach().cpu()
            self.routing_logs[path] = {"indices": idx, "layer_num": layer}

        return hook

    def _register_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        handles = []
        # find all MoEGate modules and parse layer number from name
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "MoEGate":
                m = re.search(r"layers\.(\d+)", name)
                layer = int(m.group(1)) if m else -1
                h = module.register_forward_hook(self._make_hook(name, layer))
                handles.append(h)
        return handles

    def clear_logs(self):
        # just clear for the _next_ forward
        self.routing_logs.clear()

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


def update_routing_statistics(
    batch,
    routing_logs: Dict[str, Dict[str, torch.Tensor | int]],
    expert_routing: torch.Tensor,
):
    input_ids = batch[
        "input_ids"
    ]  # (B, S); assume no padding for simplicity, or mask below
    # _ = model(**batch, use_cache=False, return_dict=True)

    B, S = input_ids.shape

    for _, info in routing_logs.items():
        indices = info["indices"].detach().cpu()  # (B, S, K)
        layer_num = info["layer_num"]

        # 1) expand your input_ids to match (B, S, K)
        toks = batch["input_ids"].unsqueeze(-1).cpu()  # (B, S, 1)
        toks = (
            toks.expand(-1, -1, indices.size(-1)) - 1
        )  # (B, S, K) # so to go from 0 to N

        # 2) build a same-shape tensor of the layer index
        lays = torch.full_like(indices, fill_value=layer_num)

        # 3) now in one shot bump every (token, layer, expert)
        expert_routing[toks, lays, indices] += 1.0
