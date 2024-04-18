from datasets import load_dataset
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer

# Load the tweet_eval dataset's emoji subset
dataset = load_dataset("tweet_eval", "emoji")
texts = [x['text'] for x in dataset['train']]  # Collect texts for tokenizer training

# Initialize and train the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer(lowercase=True)
tokenizer.train_from_iterator(texts, vocab_size=30522)  # Same vocab size as BERT for consistency

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(tokenizer.encode("LOL, just saw this #EpicFail @ the gym!  Someone tried to deadlift way more than they could handle... smh  #gymLife #noPainNoGain").tokens)
print(bert_tokenizer.tokenize("LOL, just saw this #EpicFail @ the gym!  Someone tried to deadlift way more than they could handle... smh  #gymLife #noPainNoGain"))