from typing import Dict, List, Any, Tuple, Optional

from transformers import PreTrainedTokenizer
from datasets import load_dataset
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

    # fmt: off
    categories = {"biology":["anatomy","college_biology","high_school_biology","medical_genetics","virology","human_aging","nutrition"],"business":["business_ethics","management","marketing","public_relations","professional_accounting"],"chemistry":["college_chemistry","high_school_chemistry"],"computer_science":["college_computer_science","high_school_computer_science","computer_security","machine_learning"],"economics":["econometrics","high_school_macroeconomics","high_school_microeconomics"],"engineering":["electrical_engineering"],"health":["college_medicine","clinical_knowledge","professional_medicine","human_sexuality"],"history":["high_school_european_history","high_school_us_history","high_school_world_history","prehistory"],"law":["international_law","jurisprudence","professional_law"],"math":["abstract_algebra","college_mathematics","elementary_mathematics","high_school_mathematics","high_school_statistics"],"philosophy":["philosophy","logical_fallacies","moral_disputes","moral_scenarios","world_religions"],"physics":["astronomy","college_physics","conceptual_physics","high_school_physics"],"psychology":["high_school_psychology","professional_psychology"]}
    subjects = [item for lst in categories.values() for item in lst]
    # fmt: on
    language = "EN"

    result = {}
    for subj in subjects:
        # 1) Load test split only
        ds_test = load_dataset("cais/mmlu", subj, split="test")

        # 2) Inject subject/language
        ds_subj = ds_test.map(lambda ex: {"subject": subj, "language": language})

        # 4) TODO: Have to change this (because we are doing it by subject instead of total here)
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


def mmlu_pro_collate(batch_of_examples: List[Dict[str, Any]], tok: PreTrainedTokenizer):
    """
    `batch_of_examples`: list of dicts from the dataset
    returns: dict with keys ['input_ids', 'attention_mask'] of shape (B, T)
    """
    prompts = [
        ex["question"] + "\nChoices: " + " / ".join(ex["options"])
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


def mmlu_pro_loader(
    tok: PreTrainedTokenizer, max_examples: int | None, batch_size: int = 8
) -> Dict[Tuple[str, str], DataLoader]:
    """
    Load the MMLU-Pro test set and return one DataLoader per subject.

    Returns:
        A dict mapping (language, subject) -> DataLoader over that subset.
    """
    language = "EN"

    # 1) Load the single Dataset; it has a 'subject' column
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # 2) Inject the language column
    ds = ds.map(lambda ex: {"language": language})

    if max_examples is not None:
        ds = ds.select(range(min(len(ds), max_examples)))

    # 3) Gather the set of all subjects
    subjects = ds.unique("category")

    result = {}

    # 4) For each subject, filter + truncate + wrap
    for subj in subjects:
        # filter to only this subject
        ds_subj = ds.filter(lambda ex, s=subj: ex["category"] == s)

        # collate fn (re‐use your existing mmlu_collate)
        def _collate_fn(batch):
            return mmlu_pro_collate(batch, tok)

        loader = DataLoader(
            ds_subj,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=0,
        )

        result[(language, subj)] = loader

    return result

def mmmlu_collate(batch_of_examples: List[Dict[str, Any]], tok: PreTrainedTokenizer):
    """
    `batch_of_examples`: list of dicts from the dataset
    returns: dict with keys ['input_ids', 'attention_mask'] of shape (B, T)
    """
    prompts = [
        ex["Question"] + "\nChoices: " + " / ".join([ex[letter_choice] for letter_choice in ["A", "B", "C", "D"]])
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


def mmmlu_loader(
    tok: PreTrainedTokenizer,
    languages_country: List[str] = ["DE_DE", "ES_LA", "FR_FR", "HI_IN", "JA_JP", "KO_KR", "PT_BR", "ZH_CN"], # fmt: skip
    max_examples: Optional[int] = None,
    batch_size: int = 8,
) -> Dict[Tuple[str, str], DataLoader]:
    """
    Load the MMLU test set for each language in `languages` and return
    one DataLoader per (language, subject).
    """
    result: Dict[Tuple[str, str], DataLoader] = {}

    categories = {"biology":["anatomy","college_biology","high_school_biology","medical_genetics","virology","human_aging","nutrition"],"business":["business_ethics","management","marketing","public_relations","professional_accounting"],"chemistry":["college_chemistry","high_school_chemistry"],"computer_science":["college_computer_science","high_school_computer_science","computer_security","machine_learning"],"economics":["econometrics","high_school_macroeconomics","high_school_microeconomics"],"engineering":["electrical_engineering"],"health":["college_medicine","clinical_knowledge","professional_medicine","human_sexuality"],"history":["high_school_european_history","high_school_us_history","high_school_world_history","prehistory"],"law":["international_law","jurisprudence","professional_law"],"math":["abstract_algebra","college_mathematics","elementary_mathematics","high_school_mathematics","high_school_statistics"],"philosophy":["philosophy","logical_fallacies","moral_disputes","moral_scenarios","world_religions"],"physics":["astronomy","college_physics","conceptual_physics","high_school_physics"],"psychology":["high_school_psychology","professional_psychology"]}
    subjects = [item for lst in categories.values() for item in lst]

    for language_country in languages_country:
        # 1) Load the per‐language split (hf “openai/MMMLU” uses the language
        #    as its config name)
        try:
            ds = load_dataset("openai/MMMLU", language_country, split="test")
        except Exception as e:
            raise ValueError(f"Could not load MMLU test for {language_country!r}: {e}")
        
        # 2) Inject the language column (so collate/future metrics can see it)
        language = language_country.split("_")[0]
        ds = ds.map(lambda ex: {"language": language})

        # 3) Optionally truncate overall
        if max_examples is not None:
            ds = ds.select(range(min(len(ds), max_examples)))

        # 4) Find all subjects in this language
        ln_subjects = ds.unique("Subject")
        ln_subjects = list(set(ln_subjects) & set(subjects))

        # 5) Build one DataLoader per category
        for category, category_subjects in categories.items():
            # Filter to only subjects in this category that exist in the dataset
            valid_subjects = list(set(category_subjects) & set(ln_subjects))
            if not valid_subjects:
                continue

            # Filter dataset to include all subjects in this category
            ds_category = ds.filter(lambda ex: ex["Subject"] in valid_subjects)

            def _collate_fn(batch):
                return mmmlu_collate(batch, tok)

            loader = DataLoader(
                ds_category,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_fn,
                num_workers=0,
            )
            result[(language, category)] = loader

    return result
