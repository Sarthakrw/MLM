import torch

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def encode(text, stoi):
    return [stoi[i] for i in text]

def decode(tokens, itos):
    out = [itos[i] for i in tokens]
    return ''.join(out)

@torch.no_grad()
def estimate_loss(model, data, eval_iters, train_data, val_data):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out['train_data' if data is train_data else 'val_data'] = losses.mean()
    model.train()
    return out
