from typing import Dict, List, Any, Tuple

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
    tok: PreTrainedTokenizer, max_examples: int | None, batch_size: int = 8
) -> Dict[Tuple[str, str], DataLoader]:
    """
    Load MMLU datasets split by (language, subject) as a dict of DataLoaders.

    Returns:
        Dict[Tuple[language, subject], DataLoader]
    """
    subjects = [
        "abstract_algebra",
        "anatomy",
        "business_ethics",
        "computer_security",
    ]
    language = "EN"

    result = {}
    for subj in subjects:
        # 1) Load test + validation
        splits = load_dataset("cais/mmlu", subj, split=["test", "validation"])

        # 2) Map both splits to inject subject/language
        ds_test = splits[0].map(lambda ex: {"subject": subj, "language": language})
        ds_val = splits[1].map(lambda ex: {"subject": subj, "language": language})

        # 3) Concatenate
        ds_subj = concatenate_datasets([ds_test, ds_val])

        # 4) Truncate if needed
        if max_examples is not None:
            ds_subj = ds_subj.select(range(min(len(ds_subj), max_examples)))

        # 5) Wrap in DataLoader
        def _collate_fn(batch):
            return mmlu_collate(batch, tok)

        loader = DataLoader(
            ds_subj,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=0,
        )

        # 6) Insert into return dict
        result[(language, subj)] = loader

    return result
