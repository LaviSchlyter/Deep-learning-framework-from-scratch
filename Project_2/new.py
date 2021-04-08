import math

import torch

class Sequential:
    pass

#### Tensor
class Tensor:

    def __init__(self, value, grad_fn=None):

        self.value = value
        self.grad = torch.zeros_like(value)
        self.grad_fn = grad_fn
        self.shape = value.size()



    def backward(self, output_grad=None):
        # tensor.backward() accumulates the  gradients before calling it so you need to set to zero before calling
        if output_grad is None:
            output_grad = torch.ones_like(self.value)

        self.grad = torch.zeros_like(self.value)

        if self.grad_fn is not None:
            self.grad_fn.backward(output_grad)

    # Write here a zero grad function
    def zero_grad(self):
        return torch.zeros_like(self.value)



#### Layers

class Linear:
    # You need to initialize the weights  and biases

    def __init__(self, Din, Dout):
        # Check on pytorch
        self.W = Tensor(torch.randn([Din, Dout]))
        self.b = Tensor(torch.rand(([Dout])))


    def forward(self, input_):
        # Takes input_ of shape NxD and W of DxM
        return Tensor(input_.value * self.W.value + self.b.value)


    def backward(self, gradwrtoutput):

        raise NotImplementedError

    def param(self):
        return [self.W, self.b]


class Tanh:

    def __call__(self, input_):

        return Tensor(
            value = torch.tanh(input_),
            grad_fn = TanhGradFn
        )

class Sigmoid:

    def __call__(self, input_):

        return Tensor(
            value = torch.sigmoid(input_),
            grad_fn = SigmoGradFn
        )



class Relu:

    def __call__(self, input_):

        return Tensor(
            value = torch.maximum(torch.zeros_like(input_), input_),
            grad_fn = ReluGradFn(input_)
        )

    # Same but to be able to write in the two notations
    def forward(self, input_):
        return Tensor(
            value=torch.maximum(torch.zeros_like(input_), input_),
            grad_fn=ReluGradFn(input_)
        )

#### GradFN
class LinearGradFn:
    pass

class ReluGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        input__grad = output_grad * (self.input_ > 0)
        self.input_.backward(input__grad)
        self.input_.grad += input__grad
        return input__grad

class SigmoGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        #input__grad = output_grad * self.input_ * (1 - self.input_)
        input__grad = output_grad * (1 - output_grad)
        self.input_.backward(input__grad)
        self.input_.grad += input__grad
        return input__grad

class TanhGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        input__grad = output_grad * (1 - pow(self.input_, 2))
        self.input_.backward(input__grad)
        self.input_.grad += input__grad
        return input__grad



#### Optimizers (SGD, GD, Adam)
class optim:

    def __init__(self, params, lr = 0.01):
        self.params = params # model parameters
        self.lr = lr # learning rate


    def zero_grad(self):
        return Tensor(torch.zeros_like(self.params))


class SGD(optim):

    # Need to update the weights here....
    def step(self):




#### Loss functions (MSE, cross entropy)

class LossMSE:


    def __call__(self, input_, target):
        self.input_ = input_
        self.target = target
        return 1 / 2 * ((input_ - target) ** 2).sum()


    def backward(self, grad_output):
        return (self.input_ - self.target) * grad_output


def main():


    torch.set_grad_enabled(True)
    torch.manual_seed(19)



    x = Tensor(torch.tensor([-.5, 1.5]))
    #x = torch.tensor([-.5, 1.5])

    relu = Relu()
    loss = LossMSE()

    y = relu(x)

    cost = loss(y)

    # Here call zeros grad
    cost.backward()
    print(x.grad)


if __name__ == '__main__':
    main()
