
from hyper.modules import *
from hyper.modules import Sequential, Linear, Relu, Sigmoid


class WeightSharing(Module):
    """
    A utility module for weight sharing networks.
    Given two networks `input_module` and `output_module`, this module implements the model
    `output_module(input_module(input[0]).cat(input_module(input[1]))`.
    """

    def __init__(self, input_module, output_module):
        self.output_module = output_module
        self.input_module = input_module

    def __call__(self, input_):
        assert input_.shape[1] == 2
        hidden_0 = self.input_module(input_[0])
        hidden_1 = self.input_module(input_[1])
        return self.output_module(hidden_0.cat(hidden_1))

    def param(self):
        return self.input_module.param() + self.output_module.param()


def build_simple_model():
    return Sequential([
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Relu(),
        Linear(25, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ])


def build_shared_model():
    input_module = Sequential([
        Linear(1, 10),
        Relu(),
        Linear(10, 25),
        Relu(),
    ])
    output_module = Sequential([
        Linear(50, 40),
        Relu(),
        Linear(40, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ])

    return WeightSharing(input_module, output_module)
