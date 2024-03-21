# eval.py
import torch
from dataloader import load_and_prepare_data
from transformers import AutoModelForSequenceClassification
import evaluate
from tqdm.auto import tqdm

def evaluate_model(model_path, checkpoint, batch_size=8):
    _, eval_dataloader = load_and_prepare_data(checkpoint, batch_size)

    # Load the trained model
    if model_path == checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metric = evaluate.load("glue", "mrpc")
    model.eval()
    
    # Wrap eval_dataloader with tqdm for progress visualization
    progress_bar = tqdm(eval_dataloader, desc="Evaluating")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()

if __name__ == "__main__":
    checkpoint = "bert-base-uncased"
    model_path = "saved_models/bert-base-uncased"
    print("\nEvaluating pretrained model...")
    results = evaluate_model(checkpoint, checkpoint)
    print(results)
    print("\n\nEvaluating fine-tuned model...")
    results = evaluate_model(model_path, checkpoint)
    print(results)
    
