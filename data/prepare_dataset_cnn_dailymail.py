# This script is used to prepare the CNN/DailyMail summarization dataset.
# It performs the following steps:
# 1. Load the dataset from the HuggingFace datasets library.
# 2. Convert the dataset to a standard {"instruction": str, "input": str, "output": str} format.
# 3. Save the dataset splits as JSON files.

import os
import json
import datasets
from datasets import load_dataset, load_from_disk

# load the dataset from the HuggingFace datasets library
# and save it to disk
# dataset = load_dataset("cnn_dailymail", "3.0.0")
# dataset.save_to_disk("cnn_dailymail_hf")

# Load the dataset from disk
print("Loading the dataset...")
dataset = load_from_disk("cnn_dailymail_hf")
print(dataset)

def convert_data_format(example):
    return {
        "instruction": "Summarize the following news article.",
        "input": example["article"].strip(),
        "output": example["highlights"].strip()
    }

# Convert the dataset to a standard format
print("Converting the dataset format...")
original_columns = dataset['train'].column_names
print(original_columns)
dataset = dataset.map(convert_data_format)
dataset = dataset.remove_columns(original_columns)
dataset.save_to_disk("cnn_dailymail_hf_instruct")

# Save the dataset splits as JSON files
print("Saving the dataset splits...")
os.makedirs("cnn_dailymail", exist_ok=True)
for split in dataset.keys():
    print(f"Saving {split} split...")
    data_df = dataset[split].to_pandas()[:]
    data_df.to_json(f"cnn_dailymail/{split}.json", orient="records", indent=4, force_ascii=False)