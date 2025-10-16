from typing import Dict, List

import torch
import pandas as pd
import os


def create_dataframe(sparse_tensor: torch.Tensor) -> pd.DataFrame:
    """
    Convert a sparse COO tensor of shape
    (vocab_size, seq_len, num_layers, num_experts)
    into a DataFrame with all indices and counts.

    Each row corresponds to a nonzero entry.

    Columns: ['tok', 'pos', 'layer', 'expert', 'rank', 'count']
    """
    # Ensure COO format
    st = sparse_tensor.coalesce() # to remove duplicate indices

    # indices: [5, N] -> transpose to [N, 5]
    idxs = st.indices().t().cpu().numpy()
    counts = st.values().cpu().numpy()

    # Build DataFrame
    df = pd.DataFrame(idxs, columns=["tok_id", "pos_id", "layer_id", "expert_id", "rank_id"])
    df["count"] = counts

    return df


def create_dataset_routing_statisitcs_dataframe(root_path: str) -> pd.DataFrame:
    dfs = []

    for dirpath, _, filenames in os.walk(root_path):
        if "routing_sparse.pt" in filenames:
            candidate = os.path.join(dirpath, "routing_sparse.pt")

            rel_path = os.path.relpath(candidate, root_path)
            print(f"Found: {rel_path}")

            dirs = rel_path.split("/")[:-1]  # should have subject, language and file
            assert len(dirs) == 2

            print(os.path.join(root_path, rel_path))
            sparse_tensor = torch.load(os.path.join(root_path, rel_path))

            df = create_dataframe(sparse_tensor)

            df["subject"] = dirs[0]
            df["language"] = dirs[1]

            dfs.append(df)

    if dfs:
        results = pd.concat(dfs, ignore_index=True)
    else:
        results = pd.DataFrame()

    return results


def sum_by_expert(sparse_tensor: torch.Tensor, layer: int = 0) -> torch.Tensor:
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
    # indices shape [5, N]: [tok, pos, layer, expert, rank]
    idxs = sparse_tensor.coalesce().indices()
    counts = sparse_tensor.coalesce().values()
    experts = idxs[3]  # shape [N]
    num_experts = sparse_tensor.shape[3]  # the expert-dimension size
    layer_mask = idxs[2] == layer
    counts[layer_mask]
    experts[layer_mask]

    # Sum counts for each expert
    return torch.bincount(experts, weights=counts, minlength=num_experts)


def decode_sparse_indices(
    sparse_tensor: torch.Tensor, vocab
) -> List[Dict[str, str | int]]:
    """
    Converts sparse tensor indices into readable labels.
    Returns: list of dicts with keys 'token', 'position', 'layer', 'expert', 'rank', 'count'
    """
    indices = sparse_tensor.indices()  # [5 x N]
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
