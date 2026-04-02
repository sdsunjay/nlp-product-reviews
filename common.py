"""Utility helpers for loading data and tokenizing text."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

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

# --- Model Persistence ---

def save_model(model, filename: str, directory: str = "models") -> None:
    """Persist a trained model to ``<directory>/<filename>.sav``."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{filename}.sav")
    logger.info("Saving model to %s", file_path)
    import pickle
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


def load_model(filename: str):
    """Load a pickled model from disk."""
    import pickle
    with open(filename, "rb") as f:
        return pickle.load(f)


# --- Classical ML Training ---

def train_classifier(model_name, model, X_train, X_test, y_train, y_test):
    """Train a single sklearn-compatible classifier, print metrics, and save it."""
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

    history = model.fit(X_train, y_train)
    print("Model: " + model_name)

    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))

    score = history.score(X_test, y_test)
    print("Score: " + str(score))

    predictions = [round(value) for value in y_test_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    save_model(model, model_name)

# Backwards compatibility
trainClassifier = train_classifier


# --- Tokenization helpers ---

def tokenize_text1(df, text_column_name, model_class, tokenizer_class, pretrained_weights):
    """Tokenize a text column using a pretrained model and return CLS embeddings."""
    try:
        logger.info("Starting to tokenize %s", text_column_name)
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokenized = df[text_column_name].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
        padded = padding(tokenized)
        return create_tensor(padded, model)
    except Exception:
        logger.exception("Exception in tokenize code")
        traceback.print_exc(file=sys.stdout)
        raise

# Backwards compatibility
tokenizeText1 = tokenize_text1


def tokenize_text2(df, text_column_name, model_class, tokenizer_class, pretrained_weights):
    """Tokenize a text column (token-level) using a pretrained model and return CLS embeddings."""
    try:
        logger.info("Starting to tokenize2 %s", text_column_name)
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained("distilbert-base-uncased", do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokens = df[text_column_name].apply(lambda x: tokenizer.tokenize(x)[:511])
        tokenized = tokenizer.convert_tokens_to_ids(tokens)
        padded = padding(tokenized[:511])
        return create_tensor(padded, model)
    except Exception:
        logger.exception("Exception in tokenize2 code")
        traceback.print_exc(file=sys.stdout)
        raise

# Backwards compatibility
tokenizeText2 = tokenize_text2


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
