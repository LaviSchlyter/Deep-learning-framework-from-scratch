from abc import ABC, abstractmethod

import torch
from torch import Tensor


class HyperCube:
    """ A tensor that remembers which operation computed it for the purposes of backpropagation. """

    def __init__(self, value, grad_fn=None):
        assert isinstance(value, torch.Tensor)
        self.value = value
        self.grad = zeros_like(value)
        self.grad_fn = grad_fn

    def backward(self, output_grad=None):
        """
        Recursively backpropagate gradients from this `HyperCube`.
        `output_grad` must be a `Tensor` of the same shape as this `HyperCube` representing the gradient for each value in
        this `HyperCube`. If `output_grad` is None assumes a gradient of 1.
        """

        if output_grad is None:
            output_grad = ones_like(self.value)
        assert output_grad.shape == self.shape

        if self.grad_fn is not None:
            self.grad_fn.backward(output_grad)

        self.grad += output_grad

    def zero_grad(self):
        """
        Clear the accumulated gradient
        """
        self.grad = zeros_like(self.value)

    def __getitem__(self, item):
        """
        Slicing (`[]`) operator overload.
        Only supports 2D `HyperCube`s and a single slice argument, the batch size is automatically kept.
        """
        assert len(self.shape) == 2
        if isinstance(item, int):
            item = slice(item, item + 1)
        return HyperCube(self.value[:, item], grad_fn=SliceGradFn(self, item))

    def cat(self, other):
        """
        Concatenate `self` and `other` together in the second dimension. Only supports 2D `HyperCube`s.
        """

        size_cat = [self.value.shape[0], self.value.shape[1] + other.value.shape[1]]
        concat = torch.empty(size_cat)
        concat[:, :self.shape[1]] = self.value
        concat[:, self.shape[1]:] = other.value
        return HyperCube(concat, grad_fn=CatGradFn(self, other))

    @property
    def shape(self):
        return self.value.shape

    def item(self):
        """ Extract the single value in this `HyperCube`, assuming it only contains one value.` """
        return self.value.item()


class GradFn(ABC):
    """ Base class for all gradient functions. """

    @abstractmethod
    def backward(self, output_grad):
        """ Propagate the output gradient to the operands of this operation. """
        pass


class SliceGradFn(GradFn):
    def __init__(self, input_, item):
        self.input_ = input_
        self.item = item

    def backward(self, output_grad):
        input_grad = zeros_like(self.input_.value)
        input_grad[:, self.item] = output_grad
        self.input_.backward(input_grad)


class CatGradFn(GradFn):
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

    def backward(self, output_grad):
        input_grad_1 = output_grad[:, :(self.input1.shape[1])]
        input_grad_2 = output_grad[:, (self.input1.shape[1]):]
        self.input1.backward(input_grad_1)
        self.input2.backward(input_grad_2)


def zeros_like(tensor: Tensor):
    """ Create a new `Tensor` with the same shape as `tensor` filled with zeros. """
    return torch.empty(tensor.shape).zero_()


def ones_like(tensor: Tensor):
    """ Create a new `Tensor` with the same shape as `tensor` filled with ones. """
    return zeros_like(tensor) + 1
