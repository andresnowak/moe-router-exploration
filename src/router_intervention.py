from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import re

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


class RouterIntervention(ABC):
    """Base class for router interventions."""

    def __init__(
        self,
        model: PreTrainedModel,
        prob_threshold: Optional[float] = None,
    ):
        """
        Args:
            model: The transformer model with MoE layers
            prob_threhold: Optional probability threshold where if the routing probability is below this value we make the value instead equal to zero (meaning the expert will not contribute anything in the weighted sum of experts)
        """
        self.model = model
        self.prob_threshold = prob_threshold
        self.original_forwards = {}

        if self.prob_threshold is not None:
            self._apply_interventions()

    @abstractmethod
    def _apply_interventions(self):
        """Apply the routing interventions to the model."""
        pass

    def restore(self):
        """Restore original forward methods."""
        for module_id, original_forward in self.original_forwards.items():
            # Find the module by id and restore
            for module in self.model.modules():
                if id(module) == module_id:
                    module.forward = original_forward
                    break
        self.original_forwards.clear()


class OLMoERouterIntervention(RouterIntervention):
    """Router intervention for OLMoE models."""

    def _apply_interventions(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "OlmoeSparseMoeBlock":
                m = re.search(r"layers\.(\d+)", name)
                if m:
                    layer = int(m.group(1))
                    if self.prob_threshold is not None:
                        self._patch_moe_block(module, layer)

    def _patch_moe_block(self, module, layer: int):
        """Patch the OlmoeSparseMoeBlock forward method."""
        original_forward = module.forward
        prob_threshold = self.prob_threshold
        num_experts_per_tok = module.top_k
        norm_topk_prob = module.norm_topk_prob
        num_experts = module.num_experts


        def patched_forward(hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = module.gate(hidden_states) # module. is self.

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)

            if norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            if prob_threshold is not None:
                # Make the routing probability 0 if the routing probability of that expert in the top-k is below the threshold
                routing_weights = torch.where(
                    routing_weights < prob_threshold,
                    torch.zeros_like(routing_weights),
                    routing_weights
                )
            
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be selected
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            for expert_idx in range(num_experts):
                expert_layer = module.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        module.forward = patched_forward
        self.original_forwards[id(module)] = original_forward


class DeepSeekRouterIntervention(RouterIntervention):
    """Router intervention for DeepSeek models."""

    def _apply_interventions(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "MoEGate":
                m = re.search(r"layers\.(\d+)", name)
                if m:
                    layer = int(m.group(1))
                    if self.prob_threshold is not None:
                        self._patch_gate(module, layer)

    def _patch_gate(self, module, layer: int):
        """Patch the MoEGate forward method."""
        original_forward = module.forward
        prob_threshold = self.prob_threshold

        def patched_forward(hidden_states):
            bsz, seq_len, h = hidden_states.shape        
            ### compute gating score
            hidden_states = hidden_states.view(-1, h)
            logits = F.linear(hidden_states, module.weight, None)
            if module.scoring_func == 'softmax':
                scores = logits.softmax(dim=-1)
            else:
                raise NotImplementedError(f'insupportable scoring function for MoE gating: {module.scoring_func}')
            
            ### select top-k experts
            topk_weight, topk_idx = torch.topk(scores, k=module.top_k, dim=-1, sorted=False)
            
            ### norm gate to sum 1
            if module.top_k > 1 and module.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator

            ### expert-level computation auxiliary loss
            if module.training and module.alpha > 0.0:
                scores_for_aux = scores
                aux_topk = module.top_k
                # always compute aux loss based on the naive greedy topk method
                topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
                if module.seq_aux:
                    scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                    ce = torch.zeros(bsz, module.n_routed_experts, device=hidden_states.device)
                    ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / module.n_routed_experts)
                    aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * module.alpha
                else:
                    mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=module.n_routed_experts)
                    ce = mask_ce.float().mean(0)
                    Pi = scores_for_aux.mean(0)
                    fi = ce * module.n_routed_experts
                    aux_loss = (Pi * fi).sum() * module.alpha
            else:
                aux_loss = None

            if prob_threshold is not None:
                # Make the routing probability 0 if the routing probability of that expert in the top-k is below the threshold
                topk_weight = torch.where(
                    topk_weight < prob_threshold,
                    torch.zeros_like(topk_weight),
                    topk_weight
                )

            return topk_idx, topk_weight, aux_loss

        module.forward = patched_forward
        self.original_forwards[id(module)] = original_forward


class TrinityRouterIntervention(RouterIntervention):
    """Router intervention for Trinity models."""

    def _apply_interventions(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "AfmoeTokenChoiceRouter":
                m = re.search(r"layers\.(\d+)", name)
                if m:
                    layer = int(m.group(1))
                    if layer in self.experts_to_zero:
                        self._patch_router(module, layer)

    def _patch_router(self, router_module, layer: int):
        """Patch the AfmoeTokenChoiceRouter forward method."""
        original_forward = router_module.forward
        experts_to_mask = self.experts_to_zero[layer]

        def patched_forward(hidden_states):
            pass

        router_module.forward = patched_forward
        self.original_forwards[id(router_module)] = original_forward


class GPTOssRouterIntervention(RouterIntervention):
    """Router intervention for GPT-OSS models."""

    def _apply_interventions(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "GptOssMLP":
                m = re.search(r"layers\.(\d+)", name)
                if m:
                    layer = int(m.group(1))
                    if layer in self.experts_to_zero:
                        self._patch_mlp(module, layer)

    def _patch_mlp(self, module, layer: int):
        """Patch the GptOssMLP forward method."""
        original_forward = module.forward
        prob_threshold = self.prob_threshold

        def patched_forward(hidden_states):
            pass

        module.forward = patched_forward
        self.original_forwards[id(module)] = original_forward


MODEL_INTERVENTION_CLASSES = {
    "olmoe": OLMoERouterIntervention,
    "deepseek-moe": DeepSeekRouterIntervention,
    "trinity": TrinityRouterIntervention,
    "gptoss": GPTOssRouterIntervention,
}

def create_router_intervention(
    model: PreTrainedModel,
    model_type: str,
    prob_threshold: Optional[float] = None,
) -> RouterIntervention:
    """
    Factory function to create the appropriate router intervention.

    Args:
        model: The transformer model
        model_type: Model type ("olmoe", "deepseek-moe", "trinity", "gptoss")
        prob_threshold: Optional probability threshold for routing modification

    Returns:
        RouterIntervention instance
    """
    interventions = MODEL_INTERVENTION_CLASSES

    if model_type.lower() not in interventions:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: {list(interventions.keys())}"
        )

    intervention_class = interventions[model_type.lower()]
    return intervention_class(model, prob_threshold=prob_threshold)
