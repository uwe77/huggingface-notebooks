import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import os
from typing import Optional

def download_dataset_and_tokenizer(
    checkpoint: str = "bert-base-uncased", 
    dataset_name: str = "glue", 
    dataset_config: str = "mrpc",
    tokenizer_name: Optional[str] = None
) -> None:
    """
    Downloads the specified dataset and tokenizer, avoiding re-download if they 
    already exist locally. It saves the dataset and tokenizer in separate 
    directories within the './data' folder. For the `dataset_config` argument, 
    besides "mrpc", other configurations like "sst2" can also be specified to 
    download different subsets of the GLUE benchmark.

    Args:
        checkpoint (str): The name of the model checkpoint. Defaults to "bert-base-uncased".
        dataset_name (str): The name of the dataset to download. Defaults to "glue".
        dataset_config (str): The specific configuration of the dataset to 
            download. Can be "mrpc", "sst2", etc. Defaults to "mrpc".
        tokenizer_name (Optional[str]): The name of the tokenizer to download. 
            If None, the checkpoint name is used. Defaults to None.

    Returns:
        None
    """

    # Use the checkpoint as the tokenizer name if no specific name is provided.
    tokenizer_name = tokenizer_name if tokenizer_name else checkpoint
    
    # Construct paths for dataset and tokenizer
    dataset_path = os.path.join("./data", dataset_name, dataset_config)
    tokenizer_path = os.path.join("./data/tokenizers", checkpoint)
    
    # Download dataset if not already present
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset {dataset_name} with config {dataset_config}...")
        load_dataset(dataset_name, dataset_config, cache_dir="./data")
        print("Dataset downloaded.")
    else:
        print("Dataset already exists locally.")

    # Download tokenizer if not already present
    if not os.path.exists(tokenizer_path):
        print(f"Downloading tokenizer for {checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, cache_dir="./data/tokenizers")
        tokenizer.save_pretrained(tokenizer_path)
        print("Tokenizer downloaded.")
    else:
        print("Tokenizer already exists locally.")

if __name__ == "__main__":
    # python3 download.py --checkpoint bert-base-uncased --dataset_name glue --dataset_config mrpc
    parser = argparse.ArgumentParser(description="Download datasets and tokenizer")
    parser.add_argument("--checkpoint", default="bert-base-uncased", type=str, help="Model checkpoint")
    parser.add_argument("--dataset_name", default="glue", type=str, help="Dataset name")
    parser.add_argument("--dataset_config", default="mrpc", type=str, help="Dataset configuration")
    
    args = parser.parse_args()
    download_dataset_and_tokenizer(args.checkpoint, args.dataset_name, args.dataset_config)
