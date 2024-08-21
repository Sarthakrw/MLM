# MLM
Micro Language Models(MLM) is a python framework for for pretraining and inferencing small sized language models.

The library currently includes the following architectures:

1. N-gram Architecture
2. Transformer Architecture

## Installation

```bash
pip install mlm
```

## Usage

Example [dataset](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data) is a collection of conversations between two humans. 
```
Human 1: Hi!
Human 2: What is your favorite holiday?
Human 1: one where I get to meet lots of different people.
Human 2: What was the most number of people you have ever met during a holiday?
...
```


### 1. N-gram Architecture
```python
from mlm import NGram

#Initialise the model
model = NGram(dataset=data, num_neurons=100, context_len=3, num_embedding=10)

#Train the model
history, loss = model.train(iterations=1000, learning_rate=0.1, batch_size=32)

#Inference
output = model.predict(text="Human 1: ", num_tokens=100)

print(output)
```
Output:
```
I thanled to kid.
Human 2: oht. Whate mice! What Cotly, sorses do ids. What?
Human 1: He😛%ukin cor y
```

### 2. Transformer Architecture
```python
from mlm import Transformer

#Default Parameters
block_size=8
batch_size=4
n_embd=256
head_size=80
num_heads=8
eval_iters=100
n_layer=6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iter=5000
lr = 0.001

#Initialise the model
model = Transformer(vocab_size)

#Train the model
model.train_model(iterations=max_iter, learning_rate=lr)

#Compute Loss
logits, loss = model.forward(x_val, y_val)
loss=estimate_loss(val_data)
print("validation loss: ", loss['val_data'].item())

#Inference
output = model.predict(idx=torch.zeros((1,10), dtype=torch.long, device=device), num_tokens=1000)
print(output)
```
Output:
```
Human 1: nes a dentencen Bre.
Human 1: dem the to ut kay!
Human 1: Oefllade vecent ehojring ould?
Human 2: I just to is thin dry heh i  laken :scice cam eveesh. I weeking sould ?
```
<br>

A big thanks to [Andrej Karpathy](https://github.com/karpathy/) for inspiring this project :)

## License

Apache License 2.0
