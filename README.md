# A python implementation of a Neural Network

A pure python NN library, used to showcase how they work.

## Documentation

1. [Introduction](https://htmlpreview.github.io/?https://github.com/gruberpatrick/nn/blob/master/documentation/Introduction.html)

## Run the library:

```
pip install -r requirements.txt
python model.py
```

## Use it:

The provided `model.py` file provides a sequential model format. You can use it like this:

```python
from activation import Sigmoid
from loss import CrossEntropy
from model import Sequential
from layer import Linear
from data.test_datasets import load_easy_dataset

# load a provided dataset;
X, Y = load_easy_dataset()

# create a sequential model structure;
model_definition = [
    Linear(2, 3),
    Sigmoid(),
    Linear(3, 1),
    Sigmoid(),
]

# initialize the model with the given model definition;
model = Sequential(model_definition)

# train the model on the data;
model.train(X, Y, n_h=4, loss=CrossEntropy(), num_iterations=100000, print_cost=True)
```
