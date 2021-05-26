from typing import List, Optional

import torch
from torch import nn


class ViewModule(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def dense_network(
        sizes: List[int],
        activation: Optional[nn.Module], last_activation: nn.Module,
        batch_norm: bool = False
):
    """ Build a dense network

    :param sizes: List of the input and output sizes in consecutive order for the different layers used
    :param activation: Activation functions used throughout the network
    :param last_activation: Activation function used in the last layer
    :param batch_norm: Boolean on whether to add bath normalization
    :return: A dense network using the different variables given
    """
    assert len(sizes) >= 0, "Dense network needs at least an input size"

    layers = [
        nn.Flatten() # TODO: Why ?
    ]

    prev_size = sizes[0]
    for i, size in enumerate(sizes[1:]):
        if i != 0:
            if batch_norm:
                layers.append(nn.BatchNorm1d(prev_size))

            assert activation is not None, f"Network with sizes {sizes} needs activation function"
            layers.append(activation)

        layers.append(nn.Linear(prev_size, size))
        prev_size = size

    if batch_norm:
        layers.append(nn.BatchNorm1d(prev_size))

    layers.append(last_activation)
    return nn.Sequential(*layers)


def full_conv_network():
    return nn.Sequential(
        nn.Conv2d(2, 32, (5, 5)),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, (5, 5)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


def shared_conv_network(last_activation: nn.Module, output_size: int):
    """ Shared convolutional neural network

    :param last_activation: Last activation function module
    :param output_size: Output size of the shared network
    :return: Sequential
    """
    return nn.Sequential(
        nn.Conv2d(1, 32, (5, 5)),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, (5, 5)),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        # nn.Dropout(.5),
        nn.Linear(64, 50),
        nn.ReLU(),
        nn.BatchNorm1d(50),
        # nn.Dropout(.5),
        nn.Linear(50, output_size),
        last_activation,
    )


# Resnet code based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResnetBlock(nn.Module):
    def __init__(self, channels: int, res=True):
        """

        :param channels:
        :param res:
        """
        super().__init__()

        self.res = res

        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))

        self.relu = nn.ReLU()

    def forward(self, input):
        x = input

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if self.res:
            x = x + input
        x = self.relu(x)

        return x


def shared_resnet(output_size: int, res: bool):
    return nn.Sequential(
        nn.Conv2d(1, 32, (3, 3), padding=(1, 1)),
        nn.ReLU(),

        ResnetBlock(32, res),
        ResnetBlock(32, res),

        nn.MaxPool2d((2, 2)),

        ResnetBlock(32, res),
        ResnetBlock(32, res),

        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 50),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(50, output_size),
        nn.Softmax(),
    )


class PreprocessModel(nn.Module):
    def __init__(
            self,
            a_input_module: nn.Module, b_input_module: nn.Module,
            output_head: nn.Module, digit_head: Optional[nn.Module] = None
    ):
        super().__init__()
        self.a_input_module = a_input_module
        self.b_input_module = b_input_module
        self.output_head = output_head
        self.digit_head = digit_head

    def forward(self, input):
        # Keep the channel axis to make convolutional networks easier to implement
        hidden_a = self.a_input_module(input[:, 0, None])
        hidden_b = self.b_input_module(input[:, 1, None])

        hidden = torch.stack([hidden_a, hidden_b], dim=1)
        output = self.output_head(hidden)

        if self.digit_head:
            hidden_a = self.digit_head(hidden_a)
            hidden_b = self.digit_head(hidden_b)

        return output, hidden_a, hidden_b


class WeightShareModel(PreprocessModel):
    def __init__(self, input_module: nn.Module, output_head: nn.Module, digit_head: Optional[nn.Module] = None):
        super().__init__(input_module, input_module, output_head, digit_head)


class ProbOutputLayer(nn.Module):
    @staticmethod
    def forward(input):
        lte_mask = torch.ones(10, 10).triu()[None, :, :].to(input.device)

        prob_a = input[:, 0, :, None]
        prob_b = input[:, 1, None, :]

        prob = prob_a * prob_b
        prob_lte = (lte_mask * prob).sum(axis=(1, 2))

        return prob_lte[:, None].clamp(0, 1)
