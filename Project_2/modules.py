import math
from abc import abstractmethod

import torch

from core import HyperCube


class Module:

    @abstractmethod
    def __call__(self, input_):
        pass

    @abstractmethod
    def param(self):
        pass

class WeightSharing(Module):

    def __init__(self, input_module, output_module):
        self.output_module = output_module
        self.input_module = input_module

    def __call__(self, input_):
        hidden_x = self.input_module(input_[0])
        hidden_y = self.input_module(input_[1])

        y_pred = self.output_module(hidden_x.cat(hidden_y))
        return y_pred

    def param(self):
        # Return the parameters for the optimizer
        return self.input_module.param() + self.output_module.param()


class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input_):
        for layer in self.layers:
            input_ = layer(input_)
        return input_

    def param(self):
        par = []
        for layer in self.layers:
            par.extend(layer.param())

        return par


# Layers
class Linear(Module):

    def __init__(self, Din, Dout):
        self.W = HyperCube(torch.empty([Din, Dout]).uniform_() * (2 / math.sqrt(Din)) - 1 / math.sqrt(Din))
        self.b = HyperCube(torch.empty([Dout]).uniform_())

    def __call__(self, input_):
        # Takes input_ of shape NxD and W of DxM
        return HyperCube(input_.value @ self.W.value + self.b.value, grad_fn=LinearGradFn(input_, self.W, self.b))

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

    def __call__(self, input_):
        output = 2 * Sigmoid.sigmoid(2 * input_.value) - 1
        return HyperCube(
            value=output,
            grad_fn=TanhGradFn(input_, output)
        )

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
        # Workaround to stop SegFaults in the VM from .exp
        return 1 / (1 + math.e ** (-input_))


class SigmoidGradFn:

    def __init__(self, input_, output):
        self.output = output
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = output_grad * self.output * (1 - self.output)
        self.input_.backward(input_grad)


class Relu(Module):
    # alpha is the negative slope, 0 is zero else leaky ReLu
    def __init__(self, alpha=0.):
        self.alpha = alpha
        assert (0 <= alpha <= 1)

    def __call__(self, input_):
        return HyperCube(
            value=input_.value.maximum(input_.value * self.alpha),
            grad_fn=ReluGradFn(input_, alpha=self.alpha)
        )

    def param(self):
        return []


class ReluGradFn:

    def __init__(self, input_, alpha):
        self.alpha = alpha
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = output_grad * (self.input_.value >= 0) + output_grad * (self.input_.value < 0) * self.alpha
        self.input_.backward(input_grad)


