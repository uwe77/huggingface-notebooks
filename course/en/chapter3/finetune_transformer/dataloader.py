from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import os
from typing import Tuple

def load_and_prepare_data(
    checkpoint: str = "bert-base-uncased", 
    dataset_config: str = "mrpc",
    batch_size: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads and prepares the specified GLUE dataset configuration (e.g., MRPC or SST-2) 
    for training and evaluation using a specified tokenizer. It handles tokenization, 
    sets the data format for PyTorch, and creates DataLoader instances for both 
    training and validation sets.

    Args:
        checkpoint (str): The model checkpoint for loading the tokenizer. 
            Defaults to "bert-base-uncased".
        dataset_config (str): The configuration of the GLUE dataset to load. 
            Can be "mrpc" or "sst2". Defaults to "mrpc".
        batch_size (int): The batch size for the DataLoader instances. 
            Defaults to 8.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the DataLoader instances 
        for the training and validation datasets, respectively.
    """

    dataset_name = "glue"
    dataset_path = f"./data/{dataset_name}/{dataset_config}"
    tokenizer_path = f"./data/tokenizers/{checkpoint}"
    
    # Load dataset and tokenizer from local cache
    raw_datasets = load_dataset(dataset_name, dataset_config, cache_dir=dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Determine the key names for sentences/text in the dataset
    if dataset_config == "mrpc":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif dataset_config == "sst2":
        sentence1_key, sentence2_key = "sentence", None
    else:
        raise ValueError(f"Unsupported dataset configuration: {dataset_config}")

    def tokenize_function(example):
        """
        Tokenizes sentences using the provided tokenizer. Adjusts for datasets 
        that may have one or two text fields.

        Args:
            example: A dictionary containing text fields to be tokenized.

        Returns:
            A dictionary containing the tokenized text.
        """
        # Handle datasets with single or paired sentences
        if sentence2_key:
            return tokenizer(
                example[sentence1_key], example[sentence2_key], truncation=True
            )
        else:
            return tokenizer(example[sentence1_key], truncation=True)

    # Tokenize all datasets and adjust dataset format
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    if dataset_config == "mrpc":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif dataset_config == "sst2":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence", "idx"]
        )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # Initialize data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare DataLoaders for training and evaluation
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, 
        batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size, 
        collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader
