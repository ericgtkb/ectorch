# EC Torch

A simple PyTorch library that provides a simple, but comprehensive training
loop to reduce boilerplate code.

## Installation

With `pip`:

```bash
pip install git+https://github.com/ericgtkb/ectorch.git
```

With `uv`:

```bash
uv add git+https://github.com/ericgtkb/ectorch.git
```


## Features

The `Trainer` class. Wrap your model, optimizer, and loss function. After that,
training is as simple as `trianer.train()`.

The `Callback` class. Callbacks can be used to add custom behavior to the
training loop. Two built-in callbacks are provided: `Checkpointer` and `EarlyStopping`.

## Usage

Example usage of the `Trainer` class with a simple classification model.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ectorch import Trainer


class SimpleClassification(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


iris = load_iris()

x_train, x_val, y_train, y_val = train_test_split(
    iris['data'],
    iris['target'],
    test_size=0.2,
)

x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
x_val = torch.from_numpy(x_val.astype(np.float32))
y_val = torch.from_numpy(y_val.astype(np.int64))

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)

model = SimpleClassification(input_size=4, num_classes=3)

optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_func = nn.CrossEntropyLoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_func=loss_func,
)

trainer.train(train_loader, val_data=val_loader, num_epochs=20)

y_pred = trainer.predict(x_val).argmax(dim=1)
print(f'Validation accuracy: {accuracy_score(y_val, y_pred) * 100:.2f} %.')
```

## TODO

More tests.
