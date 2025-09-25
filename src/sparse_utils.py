from typing import Dict, List

import torch


def sum_by_expert(sparse_tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns a 1D tensor [num_experts] where entry e is
    the sum of all sparse.values() whose expert index == e.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        A COO tensor of shape
        (vocab_size, seq_len, num_layers, num_experts)
        where `values()` contains counts.

    Returns
    -------
    torch.Tensor
        A 1-D dense tensor of length `num_experts` giving
        the total summed counts for each expert.
    """
    # indices shape [4, N]: [tok, pos, layer, expert]
    idxs = sparse_tensor.coalesce().indices()
    counts = sparse_tensor.coalesce().values()
    experts = idxs[3]  # shape [N]
    num_experts = sparse_tensor.shape[3]  # the expert-dimension size

    # Sum counts for each expert
    return torch.bincount(experts, weights=counts, minlength=num_experts)


def decode_sparse_indices(
    sparse_tensor: torch.Tensor, vocab
) -> List[Dict[str, str | int]]:
    """
    Converts sparse tensor indices into readable labels.
    Returns: list of dicts with keys 'token', 'position', 'layer', 'expert', 'count'
    """
    indices = sparse_tensor.indices()  # [4 x N]
    values = sparse_tensor.values()  # [N]

    data = []
    for i in range(indices.size(1)):
        tok_id, pos, lay, exp = indices[:, i].tolist()
        count = values[i].item()
        token = vocab[tok_id] if tok_id < len(vocab) else f"<unk:{tok_id}>"
        data.append(
            {
                "token": token,
                "position": pos,
                "layer": lay,
                "expert": exp,
                "count": count,
            }
        )
    return data
