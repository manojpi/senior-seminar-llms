import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ==========================
# Configuration
# ==========================
class Config:
    """Configuration class for model hyperparameters"""
    # Training parameters
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    
    # Model parameters
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    
    # System
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    seed = 1337

# ==========================
# Model Components
# ==========================
class Head(nn.Module):
    """Single head of self-attention"""
    
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Apply attention to values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention"""
    
    def __init__(self, num_heads, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple feed-forward network"""
    
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config.n_head, head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ==========================
# Main GPT Model
# ==========================
class GPTLanguageModel(nn.Module):
    """GPT Language Model"""
    
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        
        # Token and positional embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device))
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text from the model
        
        Args:
            idx: Initial context tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k predictions
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ==========================
# Data Processing
# ==========================
class TextDataset:
    """Simple character-level text dataset"""
    
    def __init__(self, text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create encoding/decoding mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def get_batch(self, split, config, train_data, val_data):
        """Get a batch of data"""
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+config.block_size] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y


# ==========================
# Training Functions
# ==========================
@torch.no_grad()
def estimate_loss(model, dataset, train_data, val_data, config):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = dataset.get_batch(split, config, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(model, dataset, train_data, val_data, config):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for iter in range(config.max_iters):
        # Evaluate loss periodically
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, dataset, train_data, val_data, config)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Get batch and compute loss
        xb, yb = dataset.get_batch('train', config, train_data, val_data)
        logits, loss = model(xb, yb)
        
        # Backprop and update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model


# ==========================
# Main Execution
# ==========================
def main(input_file='input.txt', output_file='generated_text.txt'):
    """Main function to train and generate text"""
    
    # Initialize configuration
    config = Config()
    torch.manual_seed(config.seed)
    
    # Load and process text
    print(f"Loading text from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        print("Please ensure you have a text file named 'input.txt' in the current directory.")
        print("You can download sample text with:")
        print("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        return
    
    print(f"Text loaded. Length: {len(text)} characters")
    
    # Create dataset
    dataset = TextDataset(text)
    print(f"Vocabulary size: {dataset.vocab_size}")
    
    # Prepare data
    data = torch.tensor(dataset.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Initialize model
    model = GPTLanguageModel(dataset.vocab_size, config)
    model = model.to(config.device)
    
    # Print model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")
    print(f"Using device: {config.device}")
    
    # Train model
    print("\nStarting training...")
    model = train_model(model, dataset, train_data, val_data, config)
    
    # Generate text
    print("\nGenerating text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated_ids = model.generate(context, max_new_tokens=2000, temperature=0.8, top_k=200)
    generated_text = dataset.decode(generated_ids[0].tolist())
    
    # Save generated text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")
    
    # Print sample
    print("\nSample of generated text:")
    print("="*50)
    print(generated_text[:500])
    print("="*50)
    
    return model, dataset


if __name__ == "__main__":
    # Run the training and generation
    model, dataset = main()
    
    # Optionally save the model
    save_model = input("\nSave trained model? (y/n): ").lower() == 'y'
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': dataset.vocab_size,
            'config': model.config,
            'stoi': dataset.stoi,
            'itos': dataset.itos
        }, 'gpt_model.pt')
        print("Model saved to gpt_model.pt")