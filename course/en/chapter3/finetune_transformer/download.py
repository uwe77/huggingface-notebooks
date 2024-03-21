from datasets import load_dataset
from transformers import AutoTokenizer
import os

def download_dataset_and_tokenizer(checkpoint, dataset_name="glue", dataset_config="mrpc", tokenizer_name=None):
    tokenizer_name = tokenizer_name if tokenizer_name else checkpoint
    
    # Check and download the dataset
    dataset_path = os.path.join("./data", dataset_name, dataset_config)
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset {dataset_name} with config {dataset_config}...")
        load_dataset(dataset_name, dataset_config, cache_dir="./data")
        print("Dataset downloaded.")
    else:
        print("Dataset already exists locally.")

    # Check and download the tokenizer
    tokenizer_path = os.path.join("./data/tokenizers", checkpoint)
    if not os.path.exists(tokenizer_path):
        print(f"Downloading tokenizer for {checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="./data/tokenizers")
        tokenizer.save_pretrained(tokenizer_path)
        print("Tokenizer downloaded.")
    else:
        print("Tokenizer already exists locally.")

if __name__ == "__main__":
    checkpoint = "bert-base-uncased"
    download_dataset_and_tokenizer(checkpoint)
