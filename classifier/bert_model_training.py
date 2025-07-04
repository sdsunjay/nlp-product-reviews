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
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments,
                          TrainerCallback)

from preprocess import clean_text, read_data, read_clean_data
from common import strip_outer_quotes
import logging

s3_client = boto3.client('s3')
BUCKET_NAME = 'dsunjay-bucket'

# Configure logging
logging.basicConfig(filename='results/training_log.txt',
                    filemode='a',  # 'w' to write from scratch, 'a' to append
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Example to log some info
logging.info("Training process starts")

PATH_TO_TRAINING_CSV = 'data/clean_training3.csv'
OUTPUT_MODEL_DIR = "models"

NUM_EXAMPLES = 100
NUM_EPOCHS = 8
RANDOM_SEED = 913


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the review dataset from ``file_path``.

    If ``file_path`` points to raw data containing a ``text`` column, the
    data will be cleaned using :func:`preprocess.read_data`. If the file
    already contains ``clean_text`` it is loaded directly via
    :func:`preprocess.read_clean_data`.
    """

    preview = pd.read_csv(file_path, nrows=1)
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
        self.star_ratings = star_ratings.reset_index(drop=True)  # Reset index here

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['star_rating'] = torch.tensor(self.star_ratings[idx])
        return item

    def __len__(self):
        return len(self.labels)

def upload_directory_to_s3(bucket_name, directory_path, s3_folder):
    """
    Upload a whole directory to an S3 bucket

    :param bucket_name: Bucket to upload to
    :param directory_path: Directory to upload
    :param s3_folder: Folder path in the S3 bucket
    """
    for root, dirs, files in os.walk(directory_path):
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
        # Specify your checkpoint directory and the S3 bucket details
        checkpoint_directory = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        s3_folder = f"results/checkpoints/epoch-{state.epoch}"


        # Upload the checkpoint directory to S3
        upload_directory_to_s3(BUCKET_NAME, checkpoint_directory, s3_folder)
        print(f"Uploaded epoch {state.epoch} checkpoint to S3")


def generate_encodings(sentences, model, tokenizer):
    """
    Generates contextual embeddings for a list of sentences.

    Args:
        sentences (list[str]): A list of sentences to encode.
        model: The pre-trained transformer model.
        tokenizer: The pre-trained tokenizer.

    Returns:
        dict: A dictionary mapping each sentence to its embedding tensor.
    """
    print("Generating embeddings for all sentences...")
    start_time = datetime.now()
    
    max_length = model.config.max_position_embeddings
    embeddings = {}

    for sentence in sentences:
        try:
            input_ids = tokenizer.encode(sentence, add_special_tokens=False)

            if len(input_ids) <= max_length:
                inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings[sentence] = outputs.last_hidden_state
            else:
                print(f"Sentence with {len(sentence.split())} words is too long ({len(input_ids)} tokens). Applying chunking...")
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
        except Exception as exc:
            print(f"Error processing sentence: '{sentence[:100]}...'", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            embeddings[sentence] = None

    print(f"Finished generating embeddings in {datetime.now() - start_time}.\n")
    return embeddings


def run_encoding_pipeline(df, text_column_name, model_name="allenai/longformer-base-4096"):
    """
    Main pipeline to load model, read from a DataFrame, generate embeddings, save, and print results.
    """
    try:
        print(f"Loading model and tokenizer ('{model_name}')...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("Model and tokenizer loaded.\n")

        if text_column_name not in df.columns:
            print(f"Error: Column '{text_column_name}' not found in the DataFrame.", file=sys.stderr)
            return

        sentences = df[text_column_name].dropna().astype(str).tolist()
        
        sentence_embeddings = generate_encodings(sentences, model, tokenizer)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H")
        safe_model_name = model_name.replace('/', '-')
        output_filename = f"{safe_model_name}_{timestamp}.pt"
        
        print(f"Saving embeddings to '{output_filename}'...")
        torch.save(sentence_embeddings, output_filename)
        print("Embeddings saved successfully.\n")

        print("--- Embedding Results ---")
        for i, (sentence, embedding) in enumerate(sentence_embeddings.items()):
            print("-" * 80)
            print(f"Result for Sentence {i+1}: '{sentence[:120]}...'")
            if embedding is None:
                print("  Status: FAILED")
                continue
            print(f"  Status: SUCCESS")
            print(f"  Shape of output tensor: {embedding.shape}")
            if len(embedding.shape) > 1:
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, truncation=True, max_length=model.config.max_position_embeddings))
                for j, token in enumerate(tokens):
                    token_embedding = embedding[0, j, :]
                    vector_preview = ", ".join([f"{x:.2f}" for x in token_embedding[:5]])
                    print(f"    Token: {token:<15} | Vector (first 5 of 768 dims): [{vector_preview}, ...]")
            else:
                vector_preview = ", ".join([f"{x:.2f}" for x in embedding[:5]])
                print(f"  Averaged Vector (first 5 of 768 dims): [{vector_preview}, ...]")
        print("-" * 80 + "\n")

    except Exception as exc:
        print(f"A critical error occurred in the pipeline: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stdout)


def output_eval_results(eval_result, model, model_name):

    # Add additional information
    eval_result['model_name'] = model_name
    est = pytz.timezone('US/Eastern')
    eval_time = datetime.now(est).strftime("%Y-%m-%d__%H_%M_%S")
    eval_result['eval_time'] = eval_time
    # Include model hyperparameters if available
    if hasattr(model.config, 'to_dict'):
        eval_result['model_config'] = model.config.to_dict()

    # Convert the dictionary to a JSON string
    eval_result_json_str = json.dumps(eval_result, indent=4)

    # Print the result
    print(eval_result_json_str)

      # Check if the directory exists, if not, create it
    directory = f'./{OUTPUT_MODEL_DIR}/{model_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the result to a file
    with open(f'{directory}/{eval_time}_eval_result.json', 'w') as f:
        f.write(eval_result_json_str)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    # Handle cases where ROC AUC cannot be computed
    try:
        probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1) # Compute probabilities
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("NaN")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, preds)
    # Print confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }

    # Log metrics
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

    return metrics


def split_data(df):
    """
    Splits the data into training and validation sets.

    :param df: The dataframe containing the data.
    :return: The texts, labels, and star ratings for the training and validation sets.
    """
    # Drop rows where 'star_rating' is null
    # TODO: Fix this so we don't drop these rows
    df = df.dropna(subset=['star_rating'])

    # drop index column
    df = df.reset_index(drop=True)

    features = df[['clean_text', 'star_rating']]
    labels = df['human_tag']

    # Set a fixed random seed for reproducibility
    random_seed = RANDOM_SEED

    # Split the data into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=random_seed)

    train_texts = train_features['clean_text']
    train_star_ratings = train_features['star_rating']

    val_texts = val_features['clean_text']
    val_star_ratings = val_features['star_rating']

    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)

    return train_texts, val_texts, train_labels, val_labels, train_star_ratings, val_star_ratings


def tokenize_data(texts, tokenizer):
    """
    Tokenizes the texts using the provided tokenizer.

    :param texts: The texts to tokenize.
    :param tokenizer: The tokenizer to use.
    :return: The tokenized texts.
    """
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=tokenizer.model_max_length,
    )

def create_dataset(encodings, labels, star_ratings):
    """
    Creates a dataset from the encodings and labels.

    :param encodings: The tokenized texts.
    :param labels: The labels for the texts.
    :param star_ratings: the star rating for the review.
    :return: A dataset containing the encodings and labels.
    """
    return ReviewDataset(encodings, labels, star_ratings)

def train_model(df, tokenizer, model, model_name, epochs):
    """
    Trains the model on the provided data.

    :param df: The dataframe containing the data.
    :param tokenizer: The tokenizer to use.
    :param model: The model to train.
    :param model_name: The model to train.
    :param epochs: The number of epochs to train for.
    """
    # Assuming df['star_rating'] contains the star ratings
    train_texts, val_texts, train_labels, val_labels, train_star_ratings, val_star_ratings = split_data(df)

    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)
    train_dataset = create_dataset(train_encodings, train_labels, train_star_ratings)
    val_dataset = create_dataset(val_encodings, val_labels, val_star_ratings)
    model.resize_token_embeddings(len(tokenizer))
    train_batch_size = 8
    eval_batch_size = 8
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=500)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[UploadToS3Callback()]
    )


    # Start training
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()

    try:
        # Save the final model
        trainer.save_model(f"./results/model/{model_name}")
        upload_directory_to_s3(BUCKET_NAME, "./results/model", "results/model")
    except Exception as e:
        print(f"An error occurred with uploading the model to s3: {e}")
    # Output evaluation results
    output_eval_results(eval_result, model, model_name)

def main():
    # Get the number of CPU cores
    num_cores = os.cpu_count()

    # Print the number of CPU cores
    print(f'This system has {num_cores} CPU cores.')

    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()

    print(f"This system has {num_gpus} GPUs.")
    # Load data
    df = load_dataset(PATH_TO_TRAINING_CSV)
 
    # The pipeline now expects the 'clean_text' column created by load_dataset
    # run_encoding_pipeline(df=reviews_df, text_column_name='clean_text')
    # df = df.head(NUM_EXAMPLES)
    # Sample output:
    # id, text, star rating, label
    # 31679,mine almost burned house,1,1

    # for model_name in ['bert-base-uncased-emotion', 'bert-large-uncased']:
    for model_name in ['allenai/longformer-base-4096']:
        # Initialize tokenizer for Longformer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_labels = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        ).to(device)
        train_model(df, tokenizer, model, model_name, NUM_EPOCHS)

    # Log message after training completes
    logging.info("Training process has completed.")
    upload_directory_to_s3(BUCKET_NAME, "./results/training_log.txt", "results/training_log.txt")


if __name__ == "__main__":
    # Usage example
    main()
