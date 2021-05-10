from typing import List, Optional

import torch
from torch import nn


class ViewModule(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


# TODO this function is getting pretty confusing, maybe just stop using it
def dense_network(sizes: List[int], activation: Optional[nn.Module], last_activation: nn.Module = None):
    assert len(sizes) >= 0, "Dense network needs at least an input size"

    layers = [
        nn.Flatten()
    ]

    prev_size = sizes[0]
    for i, size in enumerate(sizes[1:]):
        if i != 0:
            assert activation is not None, f"Network with sizes {sizes} needs activation function"
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


def shared_conv_network(last_activation: nn.Module, output_size: int):
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
        nn.Linear(50, output_size),
        last_activation,
    )


class WeightShareModel(nn.Module):
    def __init__(self, input_module: nn.Module, output_head: nn.Module, digit_head: Optional[nn.Module] = None):
        super().__init__()

        self.input_module = input_module
        self.digit_head = digit_head
        self.output_head = output_head

    def forward(self, input):
        hidden_a = self.input_module(input[:, 0])
        hidden_b = self.input_module(input[:, 1])

        hidden = torch.stack([hidden_a, hidden_b], dim=1)
        output = self.output_head(hidden)

        if self.digit_head:
            hidden_a = self.digit_head(hidden_a)
            hidden_b = self.digit_head(hidden_b)

        return output, hidden_a, hidden_b


# TODO use a generic "outer product layer" followed by a linear layer instead
class ProbOutputLayer(nn.Module):
    def forward(self, input):
        eq_mask = torch.eye(10)[None, :, :].to(input.device)
        lt_mask = torch.ones(10, 10).triu()[None, :, :].to(input.device)

        prob_a = input[:, 0, :, None]
        prob_b = input[:, 1, None, :]

        prob = prob_a * prob_b
        prob_eq = (eq_mask * prob).sum(axis=(1, 2))
        prob_lte = (lt_mask * prob).sum(axis=(1, 2))

        prob_lt = (prob_lte - prob_eq) / (1 - prob_eq)
        return prob_lt[:, None].clamp(0, 1)
