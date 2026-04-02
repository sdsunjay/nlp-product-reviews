# main.py
import csv
import json
import os
import re
import sys
from datetime import datetime

try:
    import boto3  # type: ignore
    s3_client = boto3.client('s3')
except Exception:  # pragma: no cover - optional dependency
    boto3 = None
    s3_client = None

try:
    import pytz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytz = None

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments,
                          TrainerCallback)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from preprocess import clean_text, read_data, read_clean_data
from common import strip_outer_quotes

import logging
import time
import warnings

# Suppress expected fine-tuning warnings
warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*byte fallback option.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# AWS / logging setup
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def prepend_star_rating(text: str, star_rating) -> str:
    """Prepend the star rating to the review text so the model can use it."""
    try:
        stars = int(star_rating)
    except (ValueError, TypeError):
        stars = 0
    return f"Rating: {stars} stars. {text}"


# ---------------------------------------------------------------------------
# AWS / logging setup
# ---------------------------------------------------------------------------

BUCKET_NAME = config.S3_BUCKET_NAME

PATH_TO_TRAINING_CSV = config.PATH_TO_TRAINING_CSV
OUTPUT_MODEL_DIR = config.OUTPUT_MODEL_DIR

NUM_EXAMPLES = config.NUM_EXAMPLES
NUM_EPOCHS = config.NUM_EPOCHS
RANDOM_SEED = config.RANDOM_SEED


def _setup_logging():
    """Configure file logging (call once from main)."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(config.RESULTS_DIR, 'training_log.txt'),
                        filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the review dataset from ``file_path``."""
    preview = pd.read_csv(
        file_path,
        nrows=1,
        quoting=csv.QUOTE_NONE,
        escapechar='\\',
    )
    if "text" in preview.columns:
        df = read_data(file_path)
    else:
        df = read_clean_data(file_path)
    df = df.dropna(subset=["human_tag"])
    df["clean_text"] = df["clean_text"].map(strip_outer_quotes)
    return df


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Weighted Trainer (class-imbalance aware)
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def upload_directory_to_s3(bucket_name, directory_path, s3_folder):
    """Upload a whole directory to an S3 bucket."""
    if s3_client is None:
        logging.warning("boto3 not available; skipping S3 upload")
        return
    for root, _, files in os.walk(directory_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory_path)
            s3_path = os.path.join(s3_folder, relative_path)
            try:
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")
            except Exception as e:
                print(f"Error uploading {local_path}: {e}")


class UploadToS3Callback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        checkpoint_directory = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        s3_folder = f"results/checkpoints/epoch-{state.epoch}"
        upload_directory_to_s3(BUCKET_NAME, checkpoint_directory, s3_folder)
        print(f"Uploaded epoch {state.epoch} checkpoint to S3")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def output_eval_results(eval_result, model, model_name):
    """Formats and saves evaluation results."""
    eval_result['model_name'] = model_name
    if pytz is not None:
        est = pytz.timezone('US/Eastern')
        eval_time = datetime.now(est).strftime("%Y-%m-%d__%H_%M_%S")
    else:
        eval_time = datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
    eval_result['eval_time'] = eval_time
    if hasattr(model.config, 'to_dict'):
        eval_result['model_config'] = model.config.to_dict()

    eval_result_json_str = json.dumps(eval_result, indent=4)
    print(eval_result_json_str)

    safe_name = model_name.replace("/", "_")
    directory = os.path.join(config.RESULTS_DIR, safe_name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f'{eval_time}_eval_result.json'), 'w') as f:
        f.write(eval_result_json_str)


def compute_metrics(pred):
    """Compute metrics using both argmax and the configured decision threshold."""
    labels = pred.label_ids
    logits = torch.tensor(pred.predictions)
    probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
    prob_class1 = probs[:, 1]

    # Standard argmax predictions
    preds_argmax = pred.predictions.argmax(-1)

    # Threshold-based predictions (optimised for recall)
    threshold = config.DECISION_THRESHOLD
    preds_threshold = (prob_class1 >= threshold).astype(int)

    # --- Argmax metrics ---
    acc = accuracy_score(labels, preds_argmax)
    f1 = f1_score(labels, preds_argmax, average="weighted")
    precision_1 = precision_score(labels, preds_argmax, zero_division=0)
    recall_1 = recall_score(labels, preds_argmax, zero_division=0)

    # --- Threshold metrics ---
    precision_1_thr = precision_score(labels, preds_threshold, zero_division=0)
    recall_1_thr = recall_score(labels, preds_threshold, zero_division=0)
    f1_thr = f1_score(labels, preds_threshold, average="weighted")

    try:
        auc = roc_auc_score(labels, prob_class1)
    except ValueError:
        auc = float("NaN")

    conf_matrix = confusion_matrix(labels, preds_threshold)
    print(f"\n--- Threshold={threshold} ---")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(classification_report(labels, preds_threshold, target_names=["safe", "flagged"]))

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision_class1": precision_1,
        "recall_class1": recall_1,
        "auc": auc,
        "threshold": threshold,
        "threshold_precision_class1": precision_1_thr,
        "threshold_recall_class1": recall_1_thr,
        "threshold_f1": f1_thr,
    }
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")
    return metrics


def find_best_threshold(pred, min_recall=0.90):
    """Find the threshold that maximises F1 while keeping recall >= min_recall."""
    labels = pred.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1).numpy()[:, 1]

    best_threshold = 0.5
    best_f1 = 0
    for thr in np.arange(0.05, 0.95, 0.05):
        preds = (probs >= thr).astype(int)
        rec = recall_score(labels, preds, zero_division=0)
        if rec >= min_recall:
            f = f1_score(labels, preds, average="weighted")
            if f > best_f1:
                best_f1 = f
                best_threshold = thr

    print(f"\nOptimal threshold for recall >= {min_recall:.0%}: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


# ---------------------------------------------------------------------------
# Data splitting and tokenisation
# ---------------------------------------------------------------------------

def split_data(df):
    """Splits data into train/val, prepending star_rating to text."""
    df = df.dropna(subset=['star_rating'])
    df = df.reset_index(drop=True)

    # Prepend star rating to review text
    df["model_input"] = df.apply(
        lambda row: prepend_star_rating(row["clean_text"], row["star_rating"]),
        axis=1,
    )

    features = df[['model_input']]
    labels = df['human_tag'].astype(int)
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=config.TEST_SIZE,
        random_state=RANDOM_SEED, stratify=labels,
    )
    train_texts = train_features['model_input']
    val_texts = val_features['model_input']
    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)
    return train_texts, val_texts, train_labels, val_labels


def tokenize_data(texts, tokenizer):
    """Tokenize texts one-by-one with truncation and padding."""
    print(f"Tokenizing {len(texts)} texts (max_length={config.MAX_TOKENS})...")

    all_input_ids = []
    all_attention_mask = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        if len(text) > config.SAFE_CHAR_LIMIT:
            text = text[:config.SAFE_CHAR_LIMIT]

        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=config.MAX_TOKENS,
        )
        all_input_ids.append(encoding['input_ids'])
        all_attention_mask.append(encoding['attention_mask'])

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_class_weights(labels):
    """Compute class weights: [weight_class0, weight_class1]."""
    if config.CLASS_1_WEIGHT > 0:
        return [1.0, config.CLASS_1_WEIGHT]
    # Auto-compute from distribution
    counts = np.bincount(labels)
    total = len(labels)
    w0 = total / (2 * counts[0])
    w1 = total / (2 * counts[1])
    return [w0, w1]


def train_model(df, tokenizer, model, model_name, epochs):
    """Trains the model on the provided data."""
    run_start = time.time()
    train_texts, val_texts, train_labels, val_labels = split_data(df)

    # Use the fixed tokenize_data function
    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)

    train_dataset = ReviewDataset(train_encodings, train_labels.tolist())
    val_dataset = ReviewDataset(val_encodings, val_labels.tolist())

    class_weights = compute_class_weights(train_labels.values)
    print(f"Class weights: [0: {class_weights[0]:.2f}, 1: {class_weights[1]:.2f}]")

    training_args = TrainingArguments(
        output_dir=config.RESULTS_DIR,
        num_train_epochs=epochs,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        fp16=config.FP16,
        no_cuda=config.FORCE_CPU,
        load_best_model_at_end=True,
        metric_for_best_model="recall_class1",
        greater_is_better=True,
        weight_decay=config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=config.WARMUP_STEPS,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[UploadToS3Callback()],
    )

    trainer.train()

    # Find optimal threshold on validation set
    val_pred = trainer.predict(val_dataset)
    best_thr = find_best_threshold(val_pred, min_recall=0.90)
    print(f"Recommended DECISION_THRESHOLD: {best_thr:.2f}")

    eval_result = trainer.evaluate()
    total_seconds = time.time() - run_start
    eval_result["recommended_threshold"] = best_thr
    eval_result["total_runtime_seconds"] = round(total_seconds, 1)
    eval_result["total_runtime_human"] = f"{int(total_seconds // 3600)}h {int((total_seconds % 3600) // 60)}m {int(total_seconds % 60)}s"
    print(f"\nTotal run time: {eval_result['total_runtime_human']}")

    try:
        safe_name = model_name.replace("/", "_")
        model_dir = os.path.join(config.RESULTS_DIR, "model", safe_name)
        trainer.save_model(model_dir)
        upload_directory_to_s3(BUCKET_NAME, model_dir, "results/model")
    except Exception as e:
        print(f"An error occurred with uploading the model to s3: {e}")

    output_eval_results(eval_result, model, model_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main execution function."""
    _setup_logging()
    logging.info("Training process starts")
    print(f'This system has {os.cpu_count()} CPU cores.')
    print(f"This system has {torch.cuda.device_count()} GPUs.")

    if not os.path.exists(PATH_TO_TRAINING_CSV):
        print(f"Creating dummy data file at {PATH_TO_TRAINING_CSV}")
        dummy_df = pd.DataFrame({
            "text": ['"This product burned my skin badly"',
                     "Coffee tastes burnt, terrible roast.",
                     "Fire hazard — nearly caught fire!",
                     "Great product, works well.",
                     "Poor quality, waste of money."],
            "human_tag": [1, 0, 1, 0, 0],
            "star_rating": [1, 2, 1, 5, 2],
        })
        os.makedirs(os.path.dirname(PATH_TO_TRAINING_CSV), exist_ok=True)
        dummy_df.to_csv(PATH_TO_TRAINING_CSV, index=False)

    df = load_dataset(PATH_TO_TRAINING_CSV)
    df = df.head(NUM_EXAMPLES)

    print(f"\nLabel distribution:\n{df['human_tag'].value_counts().sort_index()}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Max tokens: {config.MAX_TOKENS}")
    print(f"Decision threshold: {config.DECISION_THRESHOLD}")
    print(f"Class 1 weight: {config.CLASS_1_WEIGHT}\n")

    for model_name in [config.MODEL_NAME]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device(config.DEVICE)
        print('Using device:', device)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config.NUM_LABELS,
        ).to(device)
        train_model(df, tokenizer, model, model_name, NUM_EPOCHS)

    logging.info("Training process has completed.")


if __name__ == "__main__":
    main()
