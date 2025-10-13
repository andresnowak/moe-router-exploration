from typing import Dict, Tuple, List
import re
from collections import Counter
from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel


class MoELogger(ABC):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,  # to get vocab_size
    ):
        self.model = model

        # Placeholders
        self.num_layers = 0
        self.top_k_experts = 0
        self.num_experts = 0

        # routing_logs: holds the last-forward for each gate path
        self.routing_logs: Dict[str, Dict[str, torch.Tensor | int]] = {}

        # register once
        self.hook_handles = self._register_hooks()

    @abstractmethod
    def _make_hook(self, path: str, layer: int):
        pass

    @abstractmethod
    def _register_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        pass

    def clear_logs(self):
        # just clear for the _next_ forward
        self.routing_logs.clear()

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


class DeepSeekMoELogger(MoELogger):
    def __init__(self, model: PreTrainedModel, tokenizer):
        super().__init__(model, tokenizer)

        # shapes for your stats tensor:
        cfg = model.config

        self.num_layers = cfg.num_hidden_layers
        self.top_k_experts = cfg.num_experts_per_tok
        self.num_experts = cfg.n_routed_experts

    def _make_hook(self, path: str, layer: int):
        def hook(module, inputs, outputs):
            idx = outputs[0].detach()
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


class GPTOssMoELogger(MoELogger):
    def __init__(self, model: PreTrainedModel, tokenizer):
        super().__init__(model, tokenizer)

        # shapes for your stats tensor:
        cfg = model.config

        self.num_layers = cfg.num_hidden_layers
        self.top_k_experts = cfg.num_experts_per_tok
        self.num_experts = cfg.num_local_experts

    def _make_hook(self, path: str, layer: int):
        def hook(module, inputs, outputs):
            _, topk_idx = torch.topk(outputs[1], k=self.top_k_experts, dim=-1)
            self.routing_logs[path] = {"indices": topk_idx.detach(), "layer_num": layer}

        return hook

    def _register_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        handles = []
        # find all MoEGate modules and parse layer number from name
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "GptOssMLP":
                m = re.search(r"layers\.(\d+)", name)
                layer = int(m.group(1)) if m else -1
                h = module.register_forward_hook(self._make_hook(name, layer))
                handles.append(h)
        return handles


class RoutingStatisticsTracker:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_layers: int,
        num_experts: int,
    ):
        self.model = model
        self.tok = tokenizer
        cfg = model.config
        self.vocab_size = cfg.vocab_size
        self.sequence_length = cfg.max_position_embeddings
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = cfg.num_experts_per_tok

        # Counter to store flattened (token,pos,layer,expert) -> count
        self.counts = Counter()

    @staticmethod
    def _flatten_idx(
        tok_id, pos_id, layer_id, expert_id, rank_id, seq_len, num_layers, num_experts, top_k
    ):
        """Flatten 5D indices into 1D long tensor."""
        return ((
            ((tok_id * seq_len) + pos_id) * num_layers + layer_id
        ) * num_experts + expert_id) * top_k + rank_id

    def update(self, batch: Dict[str, torch.Tensor], routing_logs: Dict):
        """
        Update internal counts from one model forward pass.

        Args:
            batch: Dict with "input_ids" [B, S]
            routing_logs: Dict[layer_num] -> {"indices": [B, S, K]}
        """
        input_ids = batch["input_ids"].to(torch.long)  # (B, S)
        device = input_ids.device
        B, S = input_ids.shape

        pos = torch.arange(S, device="cpu").repeat(B)  # [B*S]
        tok = input_ids.view(-1).cpu()  # [B*S]

        for _, info in routing_logs.items():
            indices = info["indices"].cpu()  # (B, S, K) [GPU]
            layer_num = info["layer_num"]
            K = indices.size(-1) # K = self.top_k experts

            # Loop over each rank position separately
            for rank in range(K):
                expert_at_rank = indices[:, rank].view(-1)  # [B*S]
                rank_tensor = torch.full_like(tok, rank, device="cpu")  # [B*S]
                lay_tensor = torch.full_like(tok, layer_num, device="cpu")  # [B*S]

                # Flatten to 1D
                flat = self._flatten_idx(
                    tok,
                    pos,
                    lay_tensor,
                    expert_at_rank, # only expert_ids with this rank in top_k
                    rank_tensor,
                    self.sequence_length,
                    self.num_layers,
                    self.num_experts,
                    self.top_k,
                )

                # Collapse duplicates
                uniq, cnts = flat.unique(return_counts=True)

                # Update Counter
                for idx, c in zip(uniq.tolist(), cnts.tolist()):
                    self.counts[idx] += c

    def decode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert internal Counter to sparse COO tensor.

        Returns:
            indices: [5, N] int64 tensor
            values:  [N] float32 tensor
        """
        if not self.counts:
            return torch.empty(5, 0, dtype=torch.long), torch.empty(0)

        keys = torch.tensor(list(self.counts.keys()), dtype=torch.long)
        vals = torch.tensor(list(self.counts.values()), dtype=torch.float32)  # counts

        # Decode flattened index
        rank = keys % self.top_k # get rank id
        keys = keys // self.top_k
        exp = keys % self.num_experts  # get expert id
        keys = keys // self.num_experts
        lay = keys % self.num_layers  # get layer id
        keys = keys // self.num_layers
        pos = keys % self.sequence_length  # get sequence id
        tok = keys // self.sequence_length  # this is already the token id

        indices = torch.stack([tok, pos, lay, exp, rank], dim=0)
        return indices, vals

    def to_sparse_tensor(self) -> torch.Tensor:
        """Return sparse COO tensor of shape [vocab_size, sequence_length, num_layers, num_routed_experts, top_k_experts]"""
        idxs, vals = self.decode()

        shape = (
            self.vocab_size,
            self.sequence_length,
            self.num_layers,
            self.num_experts,
            self.top_k,
        )
        return torch.sparse_coo_tensor(idxs, vals, shape).coalesce()

    def reset(self):
        """Clear all accumulated counts."""
        self.counts.clear()

    def save_counter(self, path: str):
        """
        Dump the raw Counter to disk. Later you can reload it with:
            counter = torch.load(path)
        """
        torch.save(self.counts, path)

    def save_sparse(self, path: str):
        """
        Dump the sparse COO Tensor to disk. Later you can reload it with:
            st = torch.load(path)       # sparse tensor
            idxs, vals = st.indices(), st.values()
        """
        sparse = self.to_sparse_tensor()
        torch.save(sparse, path)


# def update_routing_statistics(
#     batch,
#     routing_logs: Dict[str, Dict[str, torch.Tensor | int]],
#     expert_routing: torch.Tensor,
# ):
#     input_ids = batch[
#         "input_ids"
#     ]  # (B, S); assume no padding for simplicity, or mask below
#     # _ = model(**batch, use_cache=False, return_dict=True)

#     B, S = input_ids.shape

#     for _, info in routing_logs.items():
#         indices = info["indices"].detach().cpu()  # (B, S, K)
#         layer_num = info["layer_num"]

#         # 1) expand your input_ids to match (B, S, K)
#         toks = batch["input_ids"].unsqueeze(-1).cpu()  # (B, S, 1)
#         toks = (
#             toks.expand(-1, -1, indices.size(-1)) - 1
#         )  # (B, S, K) # so to go from 0 to N

#         # 2) build a same-shape tensor of the layer index
#         lays = torch.full_like(indices, fill_value=layer_num)

#         # 3) now in one shot bump every (token, layer, expert)
#         expert_routing[toks, lays, indices] += 1.0
