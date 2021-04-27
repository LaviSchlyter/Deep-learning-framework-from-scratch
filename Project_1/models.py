from typing import List, Optional

import torch
from torch import nn


class ViewModule(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def dense_network(sizes: List[int], activation: Optional[nn.Module], last_activation: nn.Module = None):
    assert len(sizes) >= 0, "Dense network needs at least an input size"

    layers = [
        nn.Flatten()
    ]

    prev_size = sizes[0]
    for i, size in enumerate(sizes[1:]):
        if i != 0:
            assert activation is not None, f"Networks with sizes {sizes} needs activation function"
            layers.append(activation)
            layers.append(nn.BatchNorm1d(prev_size))

        layers.append(nn.Linear(prev_size, size))
        prev_size = size

    layers.append(last_activation or activation)
    return nn.Sequential(*layers)


def full_conv_network():
    return nn.Sequential(
        nn.Conv2d(2, 32, (5, 5)),
        nn.MaxPool2d(2),
        # nn.ReLU(), # TODO why does adding this lower performance?
        nn.Conv2d(32, 64, (5, 5)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


def shared_conv_network(last_activation: nn.Module):
    return nn.Sequential(
        ViewModule(-1, 1, 14, 14),
        nn.Conv2d(1, 32, (5, 5)),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, (5, 5)),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        nn.Linear(64, 50),
        nn.ReLU(),
        nn.BatchNorm1d(50),
        nn.Linear(50, 10),
        last_activation,
    )


class WeightShareModel(nn.Module):
    def __init__(self, input_module: nn.Module, output_module: nn.Module):
        super().__init__()
        self.input_module = input_module
        self.output_module = output_module

    def forward(self, input):
        hidden_a = self.input_module(input[:, 0])
        hidden_b = self.input_module(input[:, 1])

        hidden = torch.stack([hidden_a, hidden_b], dim=1)
        return self.output_module(hidden), hidden_a, hidden_b
