import math
from abc import abstractmethod, ABC

import torch


class Module:

    @abstractmethod
    def __call__(self, input_):
        pass

    @abstractmethod
    def param(self):
        pass


class WeightSharing(Module):
    # model_ = Sequential([Linear(2 * 10, 50), Relu(), Linear(50, 25), Relu(), Linear(25, 1), Sigmoid()])
    def __init__(self, input_module, output_module):
        # TODO assert the dimensions
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


class Sequential:
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


class Tensor:

    def __init__(self, value, grad_fn=None):

        assert not isinstance(value, Tensor)
        self.value = value
        self.grad = zeros_like(value)
        self.grad_fn = grad_fn

    def backward(self, output_grad=None):

        if output_grad is None:
            output_grad = ones_like(self.value)
        if self.grad_fn is not None:
            self.grad_fn.backward(output_grad)

        self.grad += output_grad

    def zero_grad(self):
        self.grad = zeros_like(self.value)

    def __getitem__(self, item):
        # [] overloading
        if isinstance(item, int):
            item = slice(item, item + 1)
        return Tensor(self.value[:, item], grad_fn=SliceGradFn(self, item))

    def cat(self, tensor):
        # Concatenation
        size_cat = [self.value.shape[0], self.value.shape[1] + tensor.value.shape[1]]
        concat = torch.empty(size_cat)

        concat[:, :self.shape[1]] = self.value
        concat[:, self.shape[1]:] = tensor.value

        return Tensor(concat, grad_fn=CatGradFn(self, tensor))

    @property
    def shape(self):
        return self.value.shape


class CatGradFn:
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

    def backward(self, output_grad):
        input_grad_1 = output_grad[:, :(self.input1.shape[1])]
        input_grad_2 = output_grad[:, (self.input1.shape[1]):]
        self.input1.backward(input_grad_1)
        self.input2.backward(input_grad_2)


def zeros_like(tensor):
    return torch.empty(tensor.shape).zero_()


def ones_like(tensor):
    return zeros_like(tensor) + 1


# Layers
class Linear(Module):

    def __init__(self, Din, Dout):
        self.W = Tensor(torch.empty([Din, Dout]).uniform_() * (2 / math.sqrt(Din)) - 1 / math.sqrt(Din))
        self.b = Tensor(torch.empty([Dout]).uniform_())

    def __call__(self, input_):
        # Takes input_ of shape NxD and W of DxM
        return Tensor(input_.value @ self.W.value + self.b.value, grad_fn=LinearGradFn(input_, self.W, self.b))

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
        return Tensor(
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
        return Tensor(
            value=output,
            grad_fn=SigmoidGradFn(input_, output)
        )

    def param(self):
        return []

    @staticmethod
    def sigmoid(input_):
        return 1 / (1 + (-input_).exp())


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
        assert (0 <= alpha < 1)

    def __call__(self, input_):
        return Tensor(
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
        input_grad = output_grad * (self.input_.value >= 0) + output_grad * (self.input_.value < 0) * (- self.alpha)
        self.input_.backward(input_grad)


class SliceGradFn:
    def __init__(self, input_, item):
        self.input_ = input_
        self.item = item

    def backward(self, output_grad):
        input_grad = zeros_like(self.input_.value)
        input_grad[:, self.item] = output_grad
        self.input_.backward(input_grad)


# Optimizers (SGD, Adam)
class Optimizer(ABC):

    def __init__(self, params):
        self.params = params  # model parameters

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    @abstractmethod
    def step(self):
        pass


class Adam(Optimizer):
    def __init__(self, params, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """

        :param params: Parameters that will be updated
        :param alpha: The learning rate or step size
        :param beta1: The exponential decay rate for the first moment estimates
        :param beta2: The exponential decay rate for the second moment estimates
        :param epsilon: Small value to prevent division by zero
        """
        super().__init__(params)
        self.epsilon = epsilon
        self.beta2 = beta2
        self.beta1 = beta1
        self.alpha = alpha
        self.t = 0
        self.m = [0] * len(self.params)
        self.v = [0] * len(self.params)

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            ok = self.alpha * m_hat / (v_hat.sqrt() + self.epsilon)
            param.value -= self.alpha * m_hat / (v_hat.sqrt() + self.epsilon)


class SGD(Optimizer):
    def __init__(self, params, lr, lambda_=0):
        """

        :param params: Parameters that will be updated
        :param lr: The learning rate
        """
        super().__init__(params)
        self.lambda_ = lambda_
        self.lr = lr

    def step(self):
        for param in self.params:
            param.value -= self.lr * (param.grad + self.lambda_ * param.value)


# Loss functions (MSE, cross entropy)

class LossMSE:

    def __call__(self, input_, target):
        return Tensor(1 / 2 * ((input_.value - target.value) ** 2).sum(dim=0), grad_fn=LossMSEGradFN(input_, target))

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
        return Tensor(-(target.value * input_.value.log().clamp(-100, float("inf")) + (1 - target.value) * (
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
