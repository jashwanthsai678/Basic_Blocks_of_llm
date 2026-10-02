from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

print("Downloading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

print("Training tokenizer...")
raw_texts = (item["text"] for item in dataset.select(range(10000)))
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(raw_texts, vocab_size=8192, min_frequency=2)

# Save the trained tokenizer files locally
tokenizer.save_model(".", "tiny_tokenizer")
print("Saved tokenizer files locally!")