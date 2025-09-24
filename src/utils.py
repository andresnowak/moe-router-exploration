from typing import Dict, List, Any

from transformers import PreTrainedTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def mmlu_collate(batch_of_examples: List[Dict[str, Any]], tok: PreTrainedTokenizer):
    """
    `batch_of_examples`: list of dicts from the dataset
    returns: dict with keys ['input_ids', 'attention_mask'] of shape (B, T)
    """
    prompts = [
        ex["question"] + "\nChoices: " + " / ".join(ex["choices"])
        for ex in batch_of_examples
    ]

    # tokenise one-by-one first (fast enough)
    enc = tok(
        prompts, truncation=True, max_length=1024, padding=False
    )  # no padding here – we do it tensor-wise below

    # convert to tensors and pad
    input_ids = [torch.tensor(x) for x in enc["input_ids"]]
    attention_mask = [torch.tensor(x) for x in enc["attention_mask"]]

    return {
        "input_ids": pad_sequence(
            input_ids, batch_first=True, padding_value=tok.pad_token_id
        ),
        "attention_mask": pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        ),
    }


def mmlu_loader(
    tok: PreTrainedTokenizer, max_examples: int, batch_size: int = 8
) -> DataLoader:
    subjects = [
        "abstract_algebra",
        "anatomy",
        "business_ethics",
        "computer_security",
    ]

    datasets = [
        load_dataset("cais/mmlu", subject, split="test") for subject in subjects
    ]

    # 3) stitch them together
    ds = concatenate_datasets(datasets)

    if max_examples:
        ds = ds.select(range(min(len(ds), max_examples)))

    # Create a DataLoader with the custom collate function
    def _collate_fn(batch):
        return mmlu_collate(batch, tok)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,  # Keep in main thread to avoid tokenization errors
    )

    return loader
