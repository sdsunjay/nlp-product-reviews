import os
import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizer


from myClass import output_eval_results, compute_metrics, split_data, tokenize_data, create_dataset
from common import getDataFrame
import config

PATH_TO_TRAINING_CSV = config.PATH_TO_TRAINING_CSV
OUTPUT_MODEL_DIR = config.OUTPUT_MODEL_DIR
NUM_EXAMPLES = config.NUM_EXAMPLES
NUM_EPOCHS = config.NUM_EPOCHS

def load_model_from_checkpoint():
    # Directory where the checkpoints are stored
    checkpoint_dir = "./results"

    # List all subdirectories in the checkpoint directory
    subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]

    # Sort the subdirectories to find the most recent one
    # Assuming the naming convention includes step or epoch number
    latest_checkpoint = sorted(subdirs, key=lambda x: int(x.split('-')[-1]))[-1]

    # Load the model from the latest checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint)

    # Now the model variable contains the most recently trained model
def load_tokenizer_from_checkpoint():

    # Path to your checkpoint directory
    checkpoint_directory = "my_model"

    # Load the tokenizer from the checkpoint directory
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_directory)
    return tokenizer

def getDatasets(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # tokenizer = load_tokenizer_from_checkpoint()
    df = getDataFrame(PATH_TO_TRAINING_CSV)
    train_texts, val_texts, train_labels, val_labels, train_star_ratings, val_star_ratings = split_data(df)

    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)
    train_dataset = create_dataset(train_encodings, train_labels, train_star_ratings)
    val_dataset = create_dataset(val_encodings, val_labels, val_star_ratings)
    return train_dataset, val_dataset

def continue_from_checkpoint():

    model_name = 'bert-large-uncased'

    train_batch_size = 8
    eval_batch_size = 8
    # Load the model
    model_checkpoint_path = "results/checkpoint-9471"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)

    train_dataset, val_dataset = getDatasets(model_name)

    # Load the training arguments
    training_args = TrainingArguments(
        output_dir=config.RESULTS_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        fp16=config.FP16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        weight_decay=config.WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=config.WARMUP_STEPS)
    # disable_tqdm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Load the optimizer and scheduler
    # optimizer = torch.load(os.path.join(model_checkpoint_path, "optimizer.pt"))
    # scheduler = torch.load(os.path.join(model_checkpoint_path, "scheduler.pt"))

    # Load the RNG state
    rng_file_path = os.path.join(model_checkpoint_path, "rng_state.pth")
    print(f"rng file path: {rng_file_path}")
    rng_state = torch.load(rng_file_path, map_location=torch.device('cpu'))
    if isinstance(rng_state, torch.ByteTensor):
        torch.set_rng_state(rng_state)
    else:
        raise TypeError("rng_state must be a torch.ByteTensor")

    # Resume training
    trainer.train(resume_from_checkpoint=model_checkpoint_path)
    eval_result = trainer.evaluate()
    output_eval_results(eval_result, model, model_name)

continue_from_checkpoint()