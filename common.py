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

def strip_outer_quotes(text: str) -> str:
    """Remove leading and trailing double quotes from ``text`` if present."""

    if isinstance(text, str) and text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text

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
    sample_frac: float = 0.5
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

# --- Core Encoding Logic ---

def create_embeddings(
    df: pd.DataFrame, text_column_name: str, model_name: str
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a text column in a DataFrame and converts it to contextual embeddings.

    This function loads a specified pre-trained model, processes each sentence,
    handles long texts via chunking, and returns a dictionary of embeddings.
    """
    start_time = datetime.now()
    print(f"Starting embedding generation with model '{model_name}'...")

    try:
        # Load Model and Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        max_length = model.config.max_position_embeddings
        print(f"Model and tokenizer loaded. Max sequence length: {max_length}")

        # Extract sentences from DataFrame
        if text_column_name not in df.columns:
            raise ValueError(f"Column '{text_column_name}' not found in DataFrame.")
        
        sentences = df[text_column_name].dropna().astype(str).tolist()
        embeddings = {}

        # Process each sentence
        for i, sentence in enumerate(sentences):
            print(f"  Processing sentence {i+1}/{len(sentences)}...")
            input_ids = tokenizer.encode(sentence, add_special_tokens=False)

            if len(input_ids) <= max_length:
                inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings[sentence] = outputs.last_hidden_state
            else:
                # Handle long sentences with chunking
                chunk_size = max_length - 2
                stride = chunk_size // 2
                all_chunk_embeddings = []
                for start in range(0, len(input_ids), stride):
                    end = start + chunk_size
                    chunk_ids = input_ids[start:end]
                    if not chunk_ids:
                        continue
                    
                    chunk_with_special_tokens = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
                    chunk_tensor = torch.tensor([chunk_with_special_tokens])
                    
                    with torch.no_grad():
                        outputs = model(chunk_tensor)
                    
                    chunk_embeddings = outputs.last_hidden_state[0, 1:-1, :]
                    all_chunk_embeddings.append(chunk_embeddings)
                
                mean_embedding = torch.cat(all_chunk_embeddings, dim=0).mean(dim=0)
                embeddings[sentence] = mean_embedding

        print(f"Embedding generation finished in {datetime.now() - start_time}.\n")
        return embeddings

    except Exception as exc:
        print(f"An error occurred during embedding creation: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stdout)
        raise

def run_encoding_pipeline(df: pd.DataFrame, text_column_name: str, model_name: str = "allenai/longformer-base-4096"):
    """
    Main pipeline to generate, save, and display embeddings from a DataFrame.
    """
    try:
        # --- Step 1: Generate Encodings ---
        sentence_embeddings = create_embeddings(df, text_column_name, model_name)

        # --- Step 2: Save Embeddings to File ---
        timestamp = datetime.now().strftime("%Y-%m-%d-%H")
        safe_model_name = model_name.replace('/', '-')
        output_filename = f"{safe_model_name}_{timestamp}.pt"
        
        print(f"Saving embeddings to '{output_filename}'...")
        torch.save(sentence_embeddings, output_filename)
        print("Embeddings saved successfully.\n")

        # --- Step 3: Display Results ---
        print("--- Embedding Results Preview ---")
        for i, (sentence, embedding) in enumerate(list(sentence_embeddings.items())[:5]): # Preview first 5
            print("-" * 80)
            print(f"Result for Sentence {i+1}: '{sentence[:120]}...'")
            if embedding is None:
                print("  Status: FAILED")
                continue
            
            print(f"  Status: SUCCESS")
            print(f"  Shape of output tensor: {embedding.shape}")
            if len(embedding.shape) > 1:
                vector_preview = ", ".join([f"{x:.2f}" for x in embedding[0, 0, :5]])
                print(f"  Sample Vector (first token): [{vector_preview}, ...]")
            else:
                vector_preview = ", ".join([f"{x:.2f}" for x in embedding[:5]])
                print(f"  Averaged Vector: [{vector_preview}, ...]")
        print("-" * 80 + "\n")

    except Exception as exc:
        print(f"A critical error occurred in the pipeline: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stdout)

