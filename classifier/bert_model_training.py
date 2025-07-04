import json
import os
from datetime import datetime
import pytz
import boto3

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification,
                          BertTokenizer,
                          Trainer,
                          TrainingArguments,
                          TrainerCallback)

from common import getDataFrame
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

def padding(tokenized):
    begin_time_main = datetime.now()
    print("Padding begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    print(type(tokenized))
    print('tokenized length: ' + str(len(tokenized)))
    max_len = 512
    zeros = np.zeros((512,), dtype=int)
    index = 0
    for item in tokenized:
        zeros[index] = 1
        index = index + 1
        if index == max_len:
            break
    print('Padding end time: ' + str(datetime.now() - begin_time_main))
    return zeros

def createTensor(padded, model):
    begin_time_main = datetime.now()
    print("Create Tensor begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    # input_ids = torch.from_numpy(padded)
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    # attention_mask = torch.from_numpy(attention_mask)

    with torch.no_grad():
        # last_hidden_states = model(input_ids)
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        # Slice the output for the first position for all the sequences,
        # take all hidden unit outputs
        features = last_hidden_states[0][:,0,:].numpy()
        print('Create Tensor end time: ' + str(datetime.now() - begin_time_main))
        return features

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
    return tokenizer(texts.tolist(), truncation=True, padding=True)

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
    df = getDataFrame(PATH_TO_TRAINING_CSV)
    # df = df.head(NUM_EXAMPLES)
    # Sample output:
    # id, text, star rating, label
    # 31679,mine almost burned house,1,1

    # for model_name in ['bert-base-uncased-emotion', 'bert-large-uncased']:
    for model_name in ['bert-large-uncased']:
        # Initialize the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name)
        num_labels = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)
        model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))
        train_model(df, tokenizer, model, model_name, NUM_EPOCHS)

    # Log message after training completes
    logging.info("Training process has completed.")
    upload_directory_to_s3(BUCKET_NAME, "./results/training_log.txt", "results/training_log.txt")


if __name__ == "__main__":
    # Usage example
    main()
