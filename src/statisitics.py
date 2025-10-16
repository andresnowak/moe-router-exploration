import pandas as pd
from transformers import AutoTokenizer
from typing import Optional, Dict


def get_sum_subject_per_expert(df: pd.DataFrame, layer: int = 0):
    df_by_expert = df[df["layer_id"] == layer].pivot_table(
        index="expert_id",  # rows
        columns="subject",  # cols
        values="count",  # values to aggregate
        aggfunc="sum",  # sum the counts
        fill_value=0,  # replace NaN with 0
    )

    df_by_expert.index = df_by_expert.index.map(lambda x: f"E_{x}")
    df_by_expert.index.name = None

    return df_by_expert

def get_domain_specialization(df: pd.DataFrame, rank: int, layer: int = 0):
    # N^(k)_{E_i, D} / N_D 
    df_filtered = df[(df["layer_id"] == layer) & (df["rank_id"] < rank)]
    df_by_expert = df_filtered.pivot_table(
        index="expert_id",  # rows
        columns="subject",  # cols
        values="count",  # values to aggregate
        aggfunc="sum",  # sum the counts
        fill_value=0,  # replace NaN with 0
    )

    total_per_subject = df_filtered.groupby("subject")["count"].sum()
    df_by_expert  = df_by_expert.div(total_per_subject, axis=1)

    df_by_expert.index = df_by_expert.index.map(lambda x: f"E_{x}")
    df_by_expert.index.name = None

    return df_by_expert


def get_vocabulary_specialization(df: pd.DataFrame, rank: int, layer: int = 0):
    # N^(k)_{x, E_i} / N_x
    df_filtered = df[(df["layer_id"] == layer) & (df["rank_id"] < rank)]
    df_by_expert = df_filtered.pivot_table(
        index="expert_id",  # rows
        columns="tok_id",  # cols
        values="count",  # values to aggregate
        aggfunc="sum",  # sum the counts
        fill_value=0,  # replace NaN with 0
    )

    total_per_token = df_filtered.groupby("tok_id")["count"].sum()
    df_by_expert = df_by_expert.div(total_per_token, axis=1)

    df_by_expert.index = df_by_expert.index.map(lambda x: f"E_{x}")
    df_by_expert.index.name = None

    return df_by_expert


def get_language_specialization(df: pd.DataFrame, rank: int, layer: int = 0):
    # N^(k)_{E_i, L} / N_L 
    df_filtered = df[(df["layer_id"] == layer) & (df["rank_id"] < rank)]
    df_by_expert = df_layer.pivot_table(
        index="expert_id",  # rows
        columns="language",  # cols
        values="count",  # values to aggregate
        aggfunc="sum",  # sum the counts
        fill_value=0,  # replace NaN with 0
    )

    total_per_subject = df_filtered.groupby("language")["count"].sum()
    df_by_expert  = df_by_expert.div(total_per_subject, axis=1)

    df_by_expert.index = df_by_expert.index.map(lambda x: f"E_{x}")
    df_by_expert.index.name = None

    return df_by_expert


def get_top_n_tokens_per_expert(
    model_name: str,
    df: pd.DataFrame,
    n: int = 10,
    layer: int = 0,
) -> pd.DataFrame:
    """
    Return a DataFrame of shape (num_experts, 2*n) with columns
    tok_1, count_1, …, tok_n, count_n.

    tok_i is the i-th most frequent **decoded** token for each expert,
    count_i is its total count.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1) Sum counts per (expert, token)
    df_tok = df[df["layer_id"] == layer].groupby(["expert_id", "tok_id"], as_index=False)["count"].sum()

    # 2) Rank tokens inside each expert
    df_tok["rank"] = (
        df_tok.groupby("expert_id")["count"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # 3) Keep only the top-n
    df_topn = df_tok[df_tok["rank"] <= n].copy()

    # 4) Decode token IDs once
    unique_ids = df_topn["tok_id"].unique()
    id_to_str = {
        int(tid): tokenizer.convert_ids_to_tokens(int(tid)) for tid in unique_ids
    }
    df_topn["token"] = df_topn["tok_id"].map(id_to_str)

    # 5) Pivot to wide format
    tok_wide = df_topn.pivot(
        index="expert_id", columns="rank", values="token"
    ).add_prefix("tok_")
    count_wide = df_topn.pivot(
        index="expert_id", columns="rank", values="count"
    ).add_prefix("count_")

    # 6) Combine and enforce full column set
    df_out = pd.concat([tok_wide, count_wide], axis=1)

    full_tok  = [f"tok_{i}"   for i in range(1, n + 1)]
    full_cnt  = [f"count_{i}" for i in range(1, n + 1)]
    full_cols = full_tok + full_cnt

    df_out = df_out.reindex(columns=full_cols)

    # 6b) Fill missing values
    df_out[full_tok] = df_out[full_tok].fillna("")   # or pd.NA
    df_out[full_cnt] = df_out[full_cnt].fillna(0).astype(int)

    # 7) Nice index
    df_out.index = df_out.index.map(lambda x: f"E_{x}")
    df_out.index.name = None

    return df_out


def get_top_tokens_by_expert_layer_proportion(
    model_name: str,
    df: pd.DataFrame,
    n: int = 10,
    tokenizer_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    For each (expert_id, layer_id), picks the top-n tokens by:
      ratio = count(expert,layer,tok) / sum_{tok'} count(expert,layer,tok')
    Returns a DataFrame indexed by (expert_id, layer_id) with columns
      tok_1, prop_1, tok_2, prop_2, …, tok_n, prop_n
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # 1) Sum counts per (expert,layer,tok)
    df_el = (
        df.groupby(["expert_id", "layer_id", "tok_id"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "tok_count"})
    )

    # 2) Compute total traffic per (expert,layer)
    totals = (
        df_el.groupby(["expert_id", "layer_id"])["tok_count"]
        .sum()
        .rename("layer_total")
        .reset_index()
    )

    # 3) Merge and compute prop = tok_count / layer_total
    df_prop = df_el.merge(totals, on=["expert_id", "layer_id"])
    df_prop["prop"] = df_prop["tok_count"] / df_prop["layer_total"]

    # 4) Rank within each (expert,layer) by descending prop
    df_prop["rank"] = (
        df_prop.groupby(["expert_id", "layer_id"])["prop"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # 5) Keep top-n
    df_top = df_prop[df_prop["rank"] <= n].copy()

    # 6) Decode tok_id → clean token string
    unique_ids = df_top["tok_id"].unique().tolist()
    id2str = {int(tid): tokenizer.convert_ids_to_tokens(int(tid)) for tid in unique_ids}
    df_top["token"] = df_top["tok_id"].map(id2str)

    # 7) Pivot wide: tokens and proportions
    tok_wide = df_top.pivot(
        index=["expert_id", "layer_id"], columns="rank", values="token"
    ).add_prefix("tok_")
    prop_wide = df_top.pivot(
        index=["expert_id", "layer_id"], columns="rank", values="prop"
    ).add_prefix("prop_")

    # 8) Interleave columns tok_i, prop_i
    df_out = pd.concat([tok_wide, prop_wide], axis=1)
    cols = []
    for i in range(1, n + 1):
        cols += [f"tok_{i}", f"prop_{i}"]
    df_out = df_out[cols]

    # 9) Pretty up the index
    df_out = df_out.rename_axis(index=["E_index", "layer_id"])
    df_out = df_out.rename(index=lambda x: f"E_{x}", level="E_index")

    return df_out


def get_amount_unique_tokens_per_expert(df: pd.DataFrame, layer: int = 0):
    unique_per_expert = (
        df[df["layer_id"] == layer].groupby("expert_id")["tok_id"]
        .nunique()
        .reset_index(name="distinct_token_count")
    )

    unique_per_expert["expert_id"] = unique_per_expert["expert_id"].map(
        lambda x: f"E_{x}"
    )
    unique_per_expert = unique_per_expert.set_index("expert_id")

    return unique_per_expert


def get_total_tokens_per_expert(df: pd.DataFrame, layer: int = 0):
    total_per_expert = (
        df[df["layer_id"] == layer].groupby("expert_id")["count"]
        .sum()
        .reset_index(name="total_token_count")
    )

    total_per_expert["proportion"] = total_per_expert["total_token_count"] / total_per_expert["total_token_count"].sum()

    total_per_expert["expert_id"] = total_per_expert["expert_id"].map(
        lambda x: f"E_{x}"
    )
    total_per_expert = total_per_expert.set_index("expert_id")

    return total_per_expert


def get_amount_unique_tokens_per_expert_per_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes, for each expert and each layer, the number of distinct tokens
    that expert used in that layer, then normalizes so each row sums to 1.

    Input df must have columns ['expert_id','layer_id','tok_id',…].

    Returns a DataFrame indexed by 'E_<expert_id>', columns = layer_id,
    and values = fraction of that expert's total distinct tokens in that layer.
    """
    # 1) Count distinct tokens per (expert, layer)
    unique_per_expert_layer = (
        df.groupby(["expert_id", "layer_id"])["tok_id"]
        .nunique()
        .reset_index(name="distinct_token_count")
    )

    # 2) Pivot to wide: rows = expert_id, cols = layer_id
    unique_wide = (
        unique_per_expert_layer.pivot(
            index="expert_id", columns="layer_id", values="distinct_token_count"
        )
        .fillna(0)
        .astype(int)
    )

    # 3) Normalize each row to sum to 1
    #    sum(axis=1) gives a Series of row‐sums
    row_sums = unique_wide.sum(axis=1)
    #    divide each row by its sum
    unique_prop = unique_wide.div(row_sums, axis=0)

    # 4) Prettify the expert index
    unique_prop.index = unique_prop.index.map(lambda x: f"E_{x}")
    unique_prop.index.name = None

    return unique_prop


def unique_top_tokens_per_expert(df_top: pd.DataFrame, n: int = 10) -> Dict:
    """
    Given a wide DataFrame df_top whose index is expert IDs (e.g. "E_0", "E_1", …)
    and whose columns are tok_1, count_1, tok_2, count_2, …, tok_n, count_n,
    return a dict mapping each expert → the set of tokens in their top-n that
    *no other* expert has in their top-n.
    """
    # 1) extract just the tok_i columns
    tok_cols = [f"tok_{i}" for i in range(1, n + 1)]
    # build a map expert → set(tokens)
    expert_tokens = {
        expert: set(row[tok_cols].tolist()) for expert, row in df_top.iterrows()
    }

    # 2) for each expert, compute tokens unique to them
    unique = {}
    all_experts = set(expert_tokens)
    for e in all_experts:
        others = all_experts - {e}
        union_others = set().union(*(expert_tokens[o] for o in others))
        unique[e] = expert_tokens[e] - union_others

    return unique