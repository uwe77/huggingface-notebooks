import argparse
import torch
from dataloader import load_and_prepare_data
from transformers import AutoModelForSequenceClassification
import evaluate
from tqdm.auto import tqdm
from typing import Dict, Any

def evaluate_model(
    model_path: str, 
    checkpoint: str, 
    dataset_config: str,
    batch_size: int
) -> Dict[str, Any]:
    """
    Evaluates a trained sequence classification model using the specified 
    evaluation dataset and metrics. The function supports evaluation using 
    models from a given checkpoint or a saved model path.

    Args:
        model_path (str): Path to the trained model directory. If the path is 
            the same as the checkpoint, the function loads the checkpoint model.
        checkpoint (str): The model checkpoint to use for loading default model 
            configurations.
        dataset_config (str): The configuration of the dataset to evaluate on.
        batch_size (int): Batch size used for the DataLoader during evaluation.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation metrics.
    """

    _, eval_dataloader = load_and_prepare_data(checkpoint, dataset_config, batch_size)

    # Load the model for evaluation
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint if model_path == checkpoint else model_path, num_labels=2
    )

    # Prepare device for model evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load evaluation metric
    metric = evaluate.load("glue", dataset_config)
    model.eval()

    # Iterate over evaluation data
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
    # python3 eval.py --checkpoint bert-base-uncased --dataset_config mrpc --model_path saved_models/bert-base-uncased --batch_size 8
    parser = argparse.ArgumentParser(description="Evaluate a Transformers model on a specified dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint for initialization")
    parser.add_argument("--dataset_config", type=str, required=True, help="Dataset configuration to use for evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")

    args = parser.parse_args()

    print("\nEvaluating pretrained model...")
    results = evaluate_model(
        model_path=args.checkpoint, 
        checkpoint=args.checkpoint, 
        dataset_config=args.dataset_config, 
        batch_size=args.batch_size
    )
    print(results)
    
    print("\n\nEvaluating fine-tuned model...")
    results = evaluate_model(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size
    )
    print(results)
    
