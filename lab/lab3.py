## 1. Tokenization
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate
from torch.utils.data import DataLoader
import torch as th
# from DataLoader import load_and_prepare_data
from tqdm import tqdm
from transformers import default_data_collator,BertTokenizer,BertForSequenceClassification

def training():
    raw_datasets = load_dataset("glue", "sst2")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(50))
    small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(50))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ## 2. Model
    training_args = TrainingArguments(output_dir="test-trainer",
                                    evaluation_strategy="epoch",
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    learning_rate=2e-5,
                                    weight_decay=1e-2,)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


    def compute_metrics(eval_p):
        metric = evaluate.load("glue", "sst2")
        logits, labels = eval_p
        preds = np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels)
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("test-trainer")

def eval(checkpoint = "bert-base-uncased",
    model_path = "test-trainer", batch_size = 8):
    def tokenize_function(examples):
        # Remove 'idx' from the examples
        sentence = examples["sentence"]
        # Tokenize the sentences
        return tokenizer(sentence, padding='max_length', truncation=True, return_tensors='pt')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Assuming you have a function `load_dataset` that returns the appropriate dataset
    eval_dataset = load_dataset("glue", "sst2")

    # Tokenize the dataset
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)["validation"].shuffle(seed=42).select(range(50))

    eval_dataloader = DataLoader(eval_dataset, batch_size, collate_fn=default_data_collator)
    # eval_dataloader = DataLoader(checkpoint, batch_size)
    # eval_dataloader = DataLoader(checkpoint, batch_size, collate_fn=default_data_collator)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)

    metric = evaluate.load("glue", "mrpc")
    model.eval()
    # progress_bar = tqdm(eval_dataloader, desc="Evaluating")
    progress_bar = tqdm(eval_dataloader, desc="Evaluating")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with th.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = th.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())
    
if __name__ == "__main__":
    i = input("1. Training\n2. Eval\n")
    training() if i == "1" else eval()
