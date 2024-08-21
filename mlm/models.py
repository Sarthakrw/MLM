import torch
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

from mlm.utils import estimate_loss, get_batch, decode, encode

class NGram:
    
    def __init__(self, dataset, num_neurons, context_len, num_embedding=3):
        self.dataset = dataset
        self.num_embedding = num_embedding
        self.context_len = context_len
        self.num_neurons = num_neurons
        
        self.unique_tokens = sorted(list(set(self.dataset)))
        self.n = len(self.unique_tokens)
        
        self.X, self.Y = self.tokenize(self.dataset)
        
        #initialising weights
        self.C = torch.randn((self.n, self.num_embedding), requires_grad=True)
        self.emb = self.C[self.X]
        self.W1 = torch.randn((self.emb.shape[1]*self.emb.shape[2], self.num_neurons), requires_grad=True)
        self.b1 = torch.randn((self.num_neurons), requires_grad=True)
        self.W2 = torch.randn((self.num_neurons, self.n), requires_grad=True)
        self.b2 = torch.randn(self.n, requires_grad=True)
        
        
    def tokenize(self, dataset):
        X, Y = [], []
        stoi = {s:i for i,s in enumerate(self.unique_tokens)}
        context = [stoi[dataset[i]] for i in range(self.context_len)]

        for ch in dataset[self.context_len:]:
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
         
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        
        return X, Y
    
    
    def train(self, iterations, learning_rate=0.1, batch_size=None):
        parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        self.history = []
        for i in range(iterations):
            
            #constructing mini-batch
            if batch_size!=None:
                ix_rand = torch.randint(0, self.X.shape[0], (batch_size,))
            else:
                ix_rand = None

            #forward pass
            self.emb = self.C[self.X[ix_rand]].squeeze()
            h = torch.tanh(self.emb.view(-1, self.emb.shape[1]*self.emb.shape[2]) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            self.loss = F.cross_entropy(logits, self.Y[ix_rand].squeeze())
            
            self.history.append(self.loss.item())
            if i%10==0:
                print(f"Iteration {i} :", self.loss.item())
            
            #backward pass
            for p in parameters:
                p.grad = None
            self.loss.backward()

            #weight update
            for p in parameters:
                p.data += -learning_rate * p.grad
        print(f"Iteration {i} :", self.loss.item())        
        return self.history, self.loss
    
                
    def predict(self, text: str, num_tokens):
        itos = {i:s for i,s in enumerate(self.unique_tokens)}
        stoi = {s:i for i,s in enumerate(self.unique_tokens)}
        
        out = []
        context = text[-self.context_len:]
        for _ in range(num_tokens):
            self.emb = self.C[torch.tensor([stoi[i] for i in context])]
            h = torch.tanh(self.emb.view(1, -1) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + itos[ix]
            out.append(itos[ix])
        return ''.join(out)
    
    
    def plot(self):
        plt.plot(torch.arange(len(self.history)), self.history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.show()
        return None
    

class Head(nn.Module):
    """ Single Self-Attention Head """

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-1, -2) * C**-0.5
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple Self-Attention running in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ Linear layer followed by non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)
    
# parameters
block_size = 8
batch_size = 4
n_embd = 256
head_size = 80
num_heads = 8
eval_iters = 100
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iter = 1000
lr = 0.001

class Block(nn.Module):
    """ Decoder block in the transformer: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*(Block(n_embd, n_head=num_heads) for _ in range(n_layer)))
        self.sa_heads = MultiHeadAttention(num_heads, n_embd // num_heads)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def predict(self, idx, num_tokens):
        for _ in range(num_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return decode(idx.squeeze().tolist())

    def train_model(self, iterations, learning_rate=0.001):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        for i in range(iterations):
            xb, yb = get_batch(train_data)
            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                out = estimate_loss(train_data)
                print(f"iteration {i}: {out['train_data'].item():.5f}")
