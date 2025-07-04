"""Utility helpers for loading data and tokenizing text."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

def read_data(filepath: str) -> pd.DataFrame:
    """Read a CSV file from ``filepath`` and return a DataFrame."""

    df = pd.read_csv(filepath, delimiter=",")
    logger.info("Loaded %d rows from %s", len(df.index), filepath)
    return df


def get_dataframe(
    training_filepath: str = "data/clean_training1.csv",
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """Return a cleaned ``DataFrame`` from ``training_filepath``.

    The data is optionally sampled using ``sample_frac`` and any rows with
    missing values are dropped.
    """

    if not os.path.exists(training_filepath):
        raise FileNotFoundError(f"Training file not found: {training_filepath}")

    logger.info("Reading training data from %s", training_filepath)
    df = read_data(training_filepath)

    if 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    df = df.dropna()
    return df

# Backwards compatibility
getDataFrame = get_dataframe


def create_tensor(padded: Iterable[Iterable[int]], model) -> np.ndarray:
    """Convert padded token ids to embeddings using ``model``."""

    start = datetime.now()
    logger.info("Creating tensor embeddings")
    input_ids = torch.tensor(np.array(list(padded)))

    with torch.no_grad():
        outputs = model(input_ids)
        features = outputs[0][:, 0, :].numpy()

    logger.info("Tensor creation finished in %s", datetime.now() - start)
    return features

# Backwards compatibility
createTensor = create_tensor


def padding(tokenized: Iterable[Iterable[int]]) -> np.ndarray:
    """Pad the provided token sequences with zeros to equal length."""

    sequences = list(tokenized)
    max_len = max(len(seq) for seq in sequences)
    padded = np.array([list(seq) + [0] * (max_len - len(seq)) for seq in sequences])
    return padded

def tokenize_text(
    df: pd.DataFrame,
    labels: Optional[Iterable[int]],
    text_column_name: str,
    model_class,
    tokenizer_class,
    pretrained_weights: str,
) -> np.ndarray:
    """Tokenize ``text_column_name`` and convert to embeddings using ``model_class``."""

    start = datetime.now()
    logger.info("Tokenizing column '%s'", text_column_name)

    try:
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))

        tokenized = df[text_column_name].apply(
            lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=511)
        )

        padded = padding(tokenized)
        tensor = create_tensor(padded, model)
        logger.info("Tokenization finished in %s", datetime.now() - start)
        return tensor
    except Exception as exc:
        logger.error("Exception while tokenizing: %s", exc)
        traceback.print_exc(file=sys.stdout)
        raise


# Backwards compatibility
tokenizeText1 = tokenize_text
