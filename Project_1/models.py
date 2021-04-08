from typing import List

import torch
from torch import nn


class ViewModule(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def dense_network(sizes: List[int]):
    assert len(sizes) >= 0, "Dense network needs at least an input size"

    layers = [
        nn.Flatten()
    ]

    prev_size = sizes[0]
    for size in sizes[1:]:
        layers.append(nn.Linear(prev_size, size))
        layers.append(nn.ReLU())
        prev_size = size

    if len(layers) > 1:
        layers[-1] = nn.Sigmoid()

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


def shared_conv_network():
    return nn.Sequential(
        ViewModule(-1, 1, 14, 14),
        nn.Conv2d(1, 32, (5, 5)),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(32, 64, (5, 5)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, 10),
        nn.ReLU(),
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
        return self.output_module(hidden)
