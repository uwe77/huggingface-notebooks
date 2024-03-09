# train.py
import torch
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
from accelerate import Accelerator
from dataloader import load_and_prepare_data
import os

def train_model(checkpoint, saved_model_path, num_epochs=3, batch_size=8):
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    
    train_dataloader, eval_dataloader = load_and_prepare_data(checkpoint, batch_size)

    accelerator = Accelerator()

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}  # Ensure batch is on the same device as model
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Save model weights
    model.save_pretrained(saved_model_path)

    print(f"Model saved to {saved_model_path}")

if __name__ == "__main__":
    checkpoint = "bert-base-uncased"
    saved_model_path = 'saved_models/bert-base-uncased'
    train_model(checkpoint, saved_model_path)
