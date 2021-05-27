from typing import List, Optional

import torch
from torch import nn


class ViewModule(nn.Module):
    """ A utility module that view the input tensor as the given shape. """

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ShapePrintModule(nn.Module):
    """ A utility module that prints the shape of the tensor it gets and just passes it along. """

    def forward(self, input):
        print(input.shape)
        return input


def dense_network(
        sizes: List[int],
        activation: Optional[nn.Module], last_activation: nn.Module,
        batch_norm: bool = False
):
    """
    Build a dense network.

    note: This function is slightly confusing to use so we stopped using it for the report plots. It was useful during
    early experimenting and is still used for the exploration plots.

    :param sizes: List of the input and output sizes in consecutive order for the different layers used
    :param activation: Activation functions used throughout the network
    :param last_activation: Activation function used in the last layer
    :param batch_norm: Boolean on whether to add bath normalization
    :return: A dense network using the different variables given
    """
    assert len(sizes) >= 0, "Dense network needs at least an input size"

    layers = [
        # start by flatting the input for if input is a 2D shape for an image or a 3D shape as output by a convolution
        nn.Flatten()
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


def basic_conv_network():
    """
    Build a simple convolutional network.

    note: this function is also not used anymore for report results.
    """

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
    """
    Build the convolutional network used for weight sharing.

    note: this function is also not used anymore for report results.
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


class ResnetBlock(nn.Module):
    """
    A single Residual block as described in https://arxiv.org/pdf/1512.03385.pdf.
    We only implement the simple variant with the same number of input and output channels.
    The residual connections can be turned off with `res=False`.
    """

    def __init__(self, channels: int, res=True):
        """
        :param channels The number of input and output channels.:
        :param res Whether to include the residual connection.
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


def build_resnet(output_size: int, res: bool):
    """
    Build a Resnet-based network for use in weight sharing models.
    Takes as input a single digit image and returns a vector of size `output_size`.
    `res` controls whether to enable the residual connections.
    """

    return nn.Sequential(
        nn.Conv2d(1, 32, (3, 3), padding=(1, 1)),
        nn.ReLU(),

        ResnetBlock(32, res),
        ResnetBlock(32, res),

        nn.MaxPool2d((2, 2)),

        ResnetBlock(32, res),
        ResnetBlock(32, res),

        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 32),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(32, output_size),
        nn.Softmax(-1),
    )


class PreprocessModel(nn.Module):
    """
    A module that preprocesses both input images, each with a possibly different network,
    before concatenating their outputs and passing that through a final network.
    This is module is used for the Shared/Duplicate comparison.
    """

    def __init__(
            self,
            a_input_module: nn.Module, b_input_module: nn.Module,
            output_head: nn.Module, digit_head: Optional[nn.Module] = None
    ):
        """
        :param a_input_module: The module that preprocesses the first digit.
        :param b_input_module: THe module that preprocesses
        :param output_head:
        :param digit_head:
        """

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
    """
    A variant of `PreprocessModel` where the same `input_module` is used for both digits, as a form of weight sharing.
    """

    def __init__(self, input_module: nn.Module, output_head: nn.Module, digit_head: Optional[nn.Module] = None):
        super().__init__(input_module, input_module, output_head, digit_head)


class ProbOutputLayer(nn.Module):
    """
    A custom output layer that looks at the predicted probabilities  for both digits and returns the correctly computed
    probability that the first is >= the second. This is meant as a replacement for the final prediction network.

    note: We chose not to include this is the report because this is not really machine learning any more, it's a fixed
    function designed by us.
    """

    @staticmethod
    def forward(input):
        lte_mask = torch.ones(10, 10).triu()[None, :, :].to(input.device)

        prob_a = input[:, 0, :, None]
        prob_b = input[:, 1, None, :]

        prob = prob_a * prob_b
        prob_lte = (lte_mask * prob).sum(axis=(1, 2))

        return prob_lte[:, None].clamp(0, 1)
