# MLM
Micro Language Models(MLM) is a python framework for for pretraining and inferencing small sized language models.

The library currently includes the following architectures:

1. N-gram language model

## Installation

```bash
pip install MLM
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


### 1. N-gram Language Model 
```python
from MLM import NGram

#Initialize the model
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

A big thanks to [Andrej Karpathy](https://github.com/karpathy/) for inspiring this project :)

## License

Apache License 2.0
