import torch as th
from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
padding_id = 100
attention_mask = []
input_str = ["it's a pleasure to work with you"
            ,"I don't like you"]
ids = []
for s in input_str:
    tokens = tokenizer.tokenize(s)
    attention_mask.append([1 for _ in range(len(tokens))])
    ids.append(tokenizer.convert_tokens_to_ids(tokens))
max_len = max([len(r) for r in attention_mask])
for i in range(len(attention_mask)):
    attention_mask[i].extend([0 for _ in range(max_len - len(attention_mask[i]))])
    ids[i].extend([padding_id for _ in range(max_len - len(ids[i]))])
batched_ids = th.LongTensor(ids)
outputs = model(batched_ids, attention_mask=th.tensor(attention_mask))
predictions = th.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
