# main.py
import csv
import json
import os
from datetime import datetime

import boto3
import pytz

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments,
                          TrainerCallback)

# These would be in your separate module files
# For this script, we define them here.
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^A-Za-z0-9\s\.\?,!']+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def strip_outer_quotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text

def read_data(file_path: str) -> pd.DataFrame:
    print("Reading and cleaning raw data...")
    df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE, escapechar='\\')
    df["clean_text"] = df["text"].apply(clean_text)
    return df

def read_clean_data(file_path: str) -> pd.DataFrame:
    print("Reading pre-cleaned data...")
    df = pd.read_csv(file_path)
    # quoting=csv.QUOTE_NONE, escapechar='\\')
    return df
# End of placeholder functions

import logging

s3_client = boto3.client('s3')
BUCKET_NAME = 'dsunjay-bucket'

# Configure logging
logging.basicConfig(filename='./results/25_07_04_training_log.txt',
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Training process starts")

PATH_TO_TRAINING_CSV = './data/25_07_04_11_18_clean_training.csv'
OUTPUT_MODEL_DIR = "models"

NUM_EXAMPLES = 1000
NUM_EPOCHS = 8
RANDOM_SEED = 913

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the review dataset from ``file_path``."""
    preview = pd.read_csv(
        file_path,
        nrows=1)
    if "text" in preview.columns:
        df = read_data(file_path)
    else:
        df = read_clean_data(file_path)
    df = df.dropna(subset=["human_tag"])
    df["clean_text"] = df["clean_text"].map(strip_outer_quotes)
    return df


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels, star_ratings):
        self.encodings = encodings
        self.labels = labels
        self.star_ratings = star_ratings.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

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
                print(f"Uploaded file {local_path} to s3://{bucket_name}/{s3_path}")
            except Exception as e:
                print(f"Error uploading {local_path} to s3://{bucket_name}/{s3_path}: {e}")

class UploadToS3Callback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        checkpoint_directory = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        s3_folder = f"results/checkpoints/epoch-{state.epoch}"
        upload_directory_to_s3(BUCKET_NAME, checkpoint_directory, s3_folder)
        print(f"Uploaded epoch {state.epoch} checkpoint to S3")

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

    directory = f'./{OUTPUT_MODEL_DIR}/{model_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{directory}/{eval_time}_eval_result.json', 'w') as f:
        f.write(eval_result_json_str)

def compute_metrics(pred):
    """Computes and logs evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    try:
        probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("NaN")

    conf_matrix = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    metrics = {
        "accuracy": acc, "f1": f1, "precision": precision,
        "recall": recall, "auc": auc,
    }
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")
    return metrics

def split_data(df):
    """Splits the data into training and validation sets."""
    df = df.dropna(subset=['star_rating'])
    df = df.reset_index(drop=True)
    features = df[['clean_text', 'star_rating']]
    labels = df['human_tag']
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_SEED
    )
    train_texts = train_features['clean_text']
    train_star_ratings = train_features['star_rating']
    val_texts = val_features['clean_text']
    val_star_ratings = val_features['star_rating']
    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)
    return train_texts, val_texts, train_labels, val_labels, train_star_ratings, val_star_ratings

def tokenize_data1(texts, tokenizer):
    """
    Tokenizes the texts using the provided tokenizer.

    :param texts: The texts to tokenize.
    :param tokenizer: The tokenizer to use.
    :return: The tokenized texts.
    """
    return tokenizer(texts.tolist(), truncation=True, padding='max_length',max_length=4000)

def tokenize_data(texts, tokenizer):
    print("Tokenizing data one by one to handle long texts and non-string data safely...")

    SAFE_CHAR_LIMIT = 5000
    MAX_TOKENS = 4096  # explicitly set max_length

    all_input_ids = []
    all_attention_mask = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text)

        if len(text) > SAFE_CHAR_LIMIT:
            text = text[:SAFE_CHAR_LIMIT]

        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_TOKENS,  # explicitly defined max length
        )
        all_input_ids.append(encoding['input_ids'])
        all_attention_mask.append(encoding['attention_mask'])

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask
    }



def create_dataset(encodings, labels, star_ratings):
    """Creates a dataset from the encodings and labels."""
    return ReviewDataset(encodings, labels, star_ratings)

def train_model(df, tokenizer, model, model_name, epochs):
    """Trains the model on the provided data."""
    train_texts, val_texts, train_labels, val_labels, train_star_ratings, val_star_ratings = split_data(df)

    # Use the fixed tokenize_data function
    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)

    train_dataset = create_dataset(train_encodings, train_labels, train_star_ratings)
    val_dataset = create_dataset(val_encodings, val_labels, val_star_ratings)
    
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=500
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[UploadToS3Callback()]
    )

    trainer.train()
    eval_result = trainer.evaluate()

    try:
        trainer.save_model(f"./results/model/{model_name}")
        upload_directory_to_s3(BUCKET_NAME, "./results/model", "results/model")
    except Exception as e:
        print(f"An error occurred with uploading the model to s3: {e}")
    
    output_eval_results(eval_result, model, model_name)

def main():
    """Main execution function."""
    print(f'This system has {os.cpu_count()} CPU cores.')
    print(f"This system has {torch.cuda.device_count()} GPUs.")
    
    # Create dummy data file if it doesn't exist for demonstration
    if not os.path.exists(PATH_TO_TRAINING_CSV):
        print(f"Creating dummy data file at {PATH_TO_TRAINING_CSV}")
        dummy_df = pd.DataFrame({
            "text": ['"This is a great product!"', "This is a terrible product.", "It's okay."],
            "human_tag": [1, 0, 0],
            "star_rating": [5, 1, 3]
        })
        os.makedirs(os.path.dirname(PATH_TO_TRAINING_CSV), exist_ok=True)
        dummy_df.to_csv(PATH_TO_TRAINING_CSV, index=False)
    
    df = load_dataset(PATH_TO_TRAINING_CSV)
    
    df = df.head(NUM_EXAMPLES) # Uncomment to run on a smaller sample

    for model_name in ['allenai/longformer-base-4096']:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_labels = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        ).to(device)
        train_model(df, tokenizer, model, model_name, NUM_EPOCHS)

    logging.info("Training process has completed.")
    # upload_directory_to_s3(BUCKET_NAME, "./results/training_log.txt", "results/training_log.txt")

if __name__ == "__main__":
    main()
