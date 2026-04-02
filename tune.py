import json
import sys
import multiprocessing
import openai
import pandas as pd

# Set up your OpenAI API key
openai.api_key = "<YOUR KEY>"

def process_data(chunk):
    # Prepare your training data for fine-tuning
    formatted_training_data = []
    for index, row in chunk.iterrows():
        formatted_training_data.append({
            "prompt": row["clean_text"],
            "completion": [row["human_tag"]]
        })

    # Fine-tune the model on your training data
    model_engine = "text-ada-001"
    model = openai.Model(model_engine)
    model.fine_tune(
        examples=formatted_training_data,
        batch_size=16,
        epochs=5
        )


# Load your data in chunks
training_path = "data/clean_training2_prepared.jsonl"
with open(training_path) as f:
    dataset = [json.loads(line) for line in f]

formatted_training_data = []
for row in dataset:
    formatted_training_data.append({
        "text": row.get("clean_text", ""),
        "labels": [row.get("human_tag", 0)]
    })

# Fine-tune the model on your training data
model_engine = "text-curie-001"
model = openai.Model(model_engine)
model.fine_tune(
    examples=formatted_training_data,
    batch_size=16,
    epochs=5
    )

# Save the fine-tuned model
model.save("fine_tuned_model")
