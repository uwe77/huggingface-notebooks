import argparse
import torch
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
from accelerate import Accelerator
from dataloader import load_and_prepare_data
import os

def train_model(
    checkpoint: str = "bert-base-uncased", 
    dataset_config: str = "mrpc",
    saved_model_path: str = "saved_models/bert-base-uncased", 
    num_epochs: int = 3, 
    batch_size: int = 8
) -> None:
    """
    Trains a sequence classification model on a specified dataset, utilizing 
    the Hugging Face Transformers library and Accelerate for potential multi-GPU 
    training. Saves the trained model to a specified directory.

    Args:
        checkpoint (str): The model checkpoint to initialize the weights from.
        dataset_config (str): The configuration of the dataset to load.
        saved_model_path (str): Path to save the trained model. Directory is 
            created if it does not exist.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        None
    """

    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    
    train_dataloader, eval_dataloader = load_and_prepare_data(checkpoint, dataset_config, batch_size)

    accelerator = Accelerator()

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.save_pretrained(saved_model_path)

    print(f"Model saved to {saved_model_path}")

if __name__ == "__main__":
    # python3 train.py --checkpoint bert-base-uncased --dataset_config mrpc --saved_model_path saved_models/bert_mrpc --num_epochs 3 --batch_size 8
    parser = argparse.ArgumentParser(description="Train a Transformers model on a specified dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint for initialization")
    parser.add_argument("--dataset_config", type=str, required=True, help="Dataset configuration to use (e.g., 'mrpc', 'sst2')")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Path where the trained model will be saved")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")

    args = parser.parse_args()
    train_model(
        args.checkpoint, 
        args.dataset_config, 
        args.saved_model_path, 
        args.num_epochs, 
        args.batch_size
    )
