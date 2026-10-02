import torch
from tokenizers import ByteLevelBPETokenizer
from model import TinyLLM, BLOCK_SIZE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer and model weights
tokenizer = ByteLevelBPETokenizer("tiny_tokenizer-vocab.json", "tiny_tokenizer-merges.txt")
model = TinyLLM().to(device)
model.load_state_dict(torch.load("tiny_llm.pt"))
model.eval()

prompt = "Once upon a time"
encoded = tokenizer.encode(prompt)
context = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)

# Generate 100 new tokens
print(f"\nPrompt: {prompt}\n" + "-"*30)
with torch.no_grad():
    for _ in range(100):
        # Crop context if it exceeds max block size
        context_cond = context[:, -BLOCK_SIZE:]
        logits, _ = model(context_cond)
        
        # Focus on the last token prediction
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

# Decode output back to text
generated_text = tokenizer.decode(context[0].tolist())
print(generated_text)