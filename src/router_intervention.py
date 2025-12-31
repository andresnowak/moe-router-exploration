from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.integrations.mxfp4 import routing_torch_dist

# NOTE: here we don't need to do a renormalization after applying the threshold (as we want to use the same behavior as the original model, just seeing if in reality it is using the value of the experts with low score in the weighted sum of experts or not)

# Global variable for triton kernels hub (initialized on first use)
triton_kernels_hub = None


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
                # Make the routing probability 0 if the routing probability of that expert in the top-k is below the threshold (we do this based on the prob value that will be used in the weighted sum of experts (even if its not renormalized to 1 in the top-k))
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


class DeepSeekMoERouterIntervention(RouterIntervention):
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
            pass

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
                    if self.prob_threshold is not None:
                        self._patch_mlp(module, layer)

    def _patch_mlp(self, module, layer: int):
        """Patch the GptOssMLP forward method to apply threshold and also be able to bypass the megablocks kernel decorator (because there is not env_variable to do it unless you are in version 5.*) (https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py)."""
        original_forward = module.forward
        prob_threshold = self.prob_threshold

        def patched_forward(hidden_states):
            # NOTE: This is used when we don't have MXFP4 installed
            # # Call router directly (bypassing the decorator)
            # router_scores, router_indices = module.router(hidden_states)

            # # Apply probability threshold
            # if prob_threshold is not None and prob_threshold > 0.0:
            #     router_scores = torch.where(
            #         router_scores < prob_threshold,
            #         torch.zeros_like(router_scores),
            #         router_scores
            #     )

            # # Call experts with modified scores (positional args only for Mxfp4GptOssExperts)
            # routed_out = module.experts(hidden_states, router_indices, router_scores)
            # return routed_out, router_scores

            # NOTE:This one is used when we have MXFP4 installed for gptoss (this is copied from https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/integrations/mxfp4.py#L272)
            import torch.distributed as dist
            global triton_kernels_hub

            # Initialize triton_kernels_hub if not already done
            if triton_kernels_hub is None:
                from kernels import get_kernel
                triton_kernels_hub = get_kernel("kernels-community/triton_kernels")

            if dist.is_available() and dist.is_initialized() and hasattr(module, "_is_hooked"):
                routing = routing_torch_dist
            else:
                routing = triton_kernels_hub.routing.routing

            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(-1, module.router.hidden_dim)
            router_logits = nn.functional.linear(hidden_states, module.router.weight, module.router.bias) # [batch*seq_len, num_experts]

            with torch.cuda.device(router_logits.device):
                # routing data is from here https://huggingface.co/kernels-community/triton_kernels/blob/main/torch-ext/triton_kernels/routing.py
                routing_data, gather_idx, scatter_idx = routing(router_logits, module.router.top_k) # should be in bf16 or float32 dtype I think
    
            # Apply probability threshold if needed
            if prob_threshold is not None and prob_threshold > 0.0:
                # NOTE: routing_data contains the routing weights, need to modify them. gate_scal (torch.Tensor) has shape [n_tokens_pad * n_expts_act]
                gate_scores_ref: torch.Tensor = routing_data.gate_scal
                gate_scores_ref.masked_fill_(gate_scores_ref < prob_threshold, 0)

            routed_out = module.experts(hidden_states, routing_data, gather_idx, scatter_idx)
            routed_out = routed_out.reshape(batch_size, -1, module.router.hidden_dim)

            return routed_out, router_logits

        module.forward = patched_forward
        self.original_forwards[id(module)] = original_forward


MODEL_INTERVENTION_CLASSES = {
    "olmoe": OLMoERouterIntervention,
    "deepseek-moe": DeepSeekMoERouterIntervention,
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
