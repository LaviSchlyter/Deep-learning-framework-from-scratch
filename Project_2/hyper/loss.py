from abc import ABC, abstractmethod

import torch

from hyper.core import HyperCube, ones_like, GradFn


class Loss(ABC):
    """ Base class for all loss functions. """

    @abstractmethod
    def __call__(self, input_: HyperCube, target: HyperCube):
        """
        Compute the loss between the network prediction `input_` and the training target `target`.
        `target` must not have a grad_fn set, it doesn't make sense to propagate the gradient to the target tensor.
        """
        pass


class LossMSE(Loss):
    """
    The Mean Squared Error (MSE) loss, defined as:
    `L = 1 / (2 * N) * sum((ŷ - y) ** 2)`
    """

    def __call__(self, input_, target):
        assert target.grad_fn is None, "The target HyperCube must not have a grad_fn"

        output = 1 / 2 * ((input_.value - target.value) ** 2).mean(dim=0)
        return HyperCube(output, grad_fn=LossMSEGradFn(input_, target))


class LossMSEGradFn(GradFn):
    def __init__(self, input_, target):
        self.target = target
        self.input_ = input_

    def backward(self, output_grad):
        batch_size = len(self.input_.value)
        input_grad = (self.input_.value - self.target.value) * output_grad / batch_size
        self.input_.backward(input_grad)


class LossBCE(Loss):
    """
    The Binary Cross Entropy (BCE) loss, defined as:
    `L = - 1 / N * sum((y * log(ŷ) + (1 - y) * log(1 - ŷ))`
    For numerical stability purposes the log output values are clamped between -100 and 100,
    the gradients are clamped similarly.
    """

    def __call__(self, input_, target):
        assert target.grad_fn is None, "The target HyperCube must not have a grad_fn"
        assert torch.all(input_.value >= 0) and torch.all(input_.value <= 1), "Input values must be between [0,1]"

        output = -(
                target.value * input_.value.log().clamp(-100, float("inf")) +
                (1 - target.value) * (1 - input_.value).log().clamp(-100, float("inf"))
        ).mean(dim=0)
        return HyperCube(output, grad_fn=LossBCEGradFn(input_, target))


class LossBCEGradFn(GradFn):
    def __init__(self, input_, target):
        self.input_ = input_
        self.target = target

    def backward(self, output_grad):
        x = self.input_.value
        y = self.target.value
        eps = ones_like(x) * 1e-12
        batch_size = len(self.input_.value)
        input_grad = ((x - y) / ((1 - x) * x).maximum(eps)) * output_grad / batch_size
        self.input_.backward(input_grad)
