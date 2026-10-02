import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from model import TinyLLM, VOCAB_SIZE, BLOCK_SIZE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on device: {device}")

# 1. Load trained tokenizer
tokenizer = ByteLevelBPETokenizer("tiny_tokenizer-vocab.json", "tiny_tokenizer-merges.txt")

# 2. Load dataset and encode a text chunk into token IDs
print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Collect first 2,000 stories into a single token stream for fast local memory access
print("Tokenizing stories into memory...")
all_text = " ".join([item["text"] for item in dataset.select(range(2000))])
encoded = tokenizer.encode(all_text)
data_ids = torch.tensor(encoded.ids, dtype=torch.long)

print(f"Total training tokens: {len(data_ids):,}")

# Function to draw a random batch from real tokenized data
def get_batch(batch_size=16):
    ix = torch.randint(len(data_ids) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data_ids[i:i+BLOCK_SIZE] for i in ix]).to(device)
    y = torch.stack([data_ids[i+1:i+BLOCK_SIZE+1] for i in ix]).to(device)
    return x, y

# 3. Initialize Model & Optimizer
model = TinyLLM().to(device)
scaler = torch.amp.GradScaler('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# 4. Training Loop
model.train()
print("Starting real training on GPU...")

for step in range(1000):
    x, y = get_batch(batch_size=16)

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast('cuda'):
        logits, loss = model(x, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
        print(f"Step {step:04d} | Loss: {loss.item():.4f}")

# Save the trained model parameters
torch.save(model.state_dict(), "tiny_llm.pt")
print("Model saved to tiny_llm.pt!")