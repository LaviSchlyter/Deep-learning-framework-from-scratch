import math
from abc import abstractmethod, ABC
from typing import List

import torch

from hyper.core import HyperCube


class Module(ABC):
    @abstractmethod
    def __call__(self, input_: HyperCube) -> HyperCube:
        """ Compute the forward operation on `input_`, returning a new `HyperCube` with a correct `grad_fn` set. """
        pass

    @abstractmethod
    def param(self) -> List[HyperCube]:
        """ Return the list of parameters used by this `Module`. """
        pass


class Sequential(Module):
    """ Utility module to apply a list of submodules to a given input, in order. """

    def __init__(self, layers: List[Module]):
        self.layers = layers

    def __call__(self, input_):
        for layer in self.layers:
            input_ = layer(input_)
        return input_

    def param(self):
        all_param = []
        for layer in self.layers:
            all_param.extend(layer.param())
        return all_param


class Linear(Module):
    """
    A linear transformation with learnable weight and bias:
    y = W * x + b
    Where x is the input, y the output and W anb b the learnable weight and bias respectively.
    """

    def __init__(self, Din, Dout):
        """
        Randomly initializes the weights given the input and output dimension using the same initialization scheme
        used by PyTorch.
        :param Din: Input dimension of the layer
        :param Dout: Output dimension of the layer
        """
        bound = 1 / math.sqrt(Din)

        self.W = HyperCube(torch.empty([Din, Dout]).uniform_() * (2 * bound) - bound)
        self.b = HyperCube(torch.empty([Dout]).uniform_() * (2 * bound) - bound)

    def __call__(self, input_):
        # Takes input_ of shape NxD and W of DxM
        output = input_.value @ self.W.value + self.b.value
        return HyperCube(output, grad_fn=LinearGradFn(input_, self.W, self.b))

    def param(self):
        return [self.W, self.b]


class LinearGradFn:
    def __init__(self, input_, W, b):
        self.b = b
        self.W = W
        self.input_ = input_

    def backward(self, output_grad):
        self.b.backward(output_grad.sum(dim=0))
        self.W.backward((self.input_.value[:, :, None] @ output_grad[:, None, :]).sum(dim=0))
        input_grad = output_grad @ self.W.value.T
        self.input_.backward(input_grad)


class Tanh(Module):
    """
    The tanh activation function, defined as:
    `y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    This module has no learnable parameters.
    """

    def __call__(self, input_):
        output = 2 * Sigmoid.sigmoid(2 * input_.value) - 1
        return HyperCube(value=output, grad_fn=TanhGradFn(input_, output))

    def param(self):
        return []


class TanhGradFn:
    def __init__(self, input_, output):
        self.output = output
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = output_grad * (1 - pow(self.output, 2))
        self.input_.backward(input_grad)


class Sigmoid(Module):
    """
    The sigmoid activation function, defined as:
    `y = 1 / (1 + exp(-x))`
    This module has no learnable parameters.
    """

    def __call__(self, input_):
        output = Sigmoid.sigmoid(input_.value)
        return HyperCube(
            value=output,
            grad_fn=SigmoidGradFn(input_, output)
        )

    def param(self):
        return []

    @staticmethod
    def sigmoid(input_):
        # We would normally use (-input).exp() here, but that sporadically segfaults the VM.
        # This solution appears to work around that issues.
        return 1 / (1 + math.e ** (-input_))


class SigmoidGradFn:
    def __init__(self, input_, output):
        self.output = output
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = output_grad * self.output * (1 - self.output)
        self.input_.backward(input_grad)


class Relu(Module):
    """
    The (Leaky)ReLU activation function defined as:
    `y = max(x, a * x)` where `0 <= a <= 1`.
    If `a=0` this is the ordinary ReLU function.
    This module has no learnable parameters.
    """

    def __init__(self, alpha=0.):
        self.alpha = alpha
        assert (0 <= alpha <= 1)

    def __call__(self, input_):
        output = input_.value.maximum(input_.value * self.alpha)
        return HyperCube(value=output, grad_fn=ReluGradFn(input_, alpha=self.alpha))

    def param(self):
        return []


class ReluGradFn:
    def __init__(self, input_, alpha):
        self.alpha = alpha
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = output_grad * (self.input_.value >= 0) + output_grad * (self.input_.value < 0) * self.alpha
        self.input_.backward(input_grad)
