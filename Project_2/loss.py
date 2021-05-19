import torch

from core import HyperCube, ones_like

class LossMSE:

    def __call__(self, input_, target):
        return HyperCube(1 / 2 * ((input_.value - target.value) ** 2).sum(dim=0), grad_fn=LossMSEGradFN(input_, target))

    @staticmethod
    def param():
        return []


class LossMSEGradFN:
    def __init__(self, input_, target):
        self.target = target
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = (self.input_.value - self.target.value) * output_grad
        self.input_.backward(input_grad)
        self.target.backward(-input_grad)


class LossBCE:

    def __call__(self, input_, target):
        assert torch.all(input_.value >=0) and torch.all(input_.value <= 1), "Input values must be between [0,1]"


        return HyperCube(-(target.value * input_.value.log().clamp(-100, float("inf")) + (1 - target.value) * (
                1 - input_.value).log().clamp(-100, float("inf"))).sum(dim=0),
                         grad_fn=LossBCEGradFN(input_, target))

    @staticmethod
    def param():
        return []


class LossBCEGradFN:

    def __init__(self, input_, target):
        self.input_ = input_
        self.target = target

    def backward(self, output_grad):
        x = self.input_.value
        y = self.target.value
        eps = ones_like(x) * 1e-12
        input_grad = ((x - y) / ((1 - x) * x).maximum(eps)) * output_grad
        self.input_.backward(input_grad)
        self.target.backward((-x.log() + (1 - x).log()) * output_grad)