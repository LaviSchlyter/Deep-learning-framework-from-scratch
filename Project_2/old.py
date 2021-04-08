import torch
import numpy as np


class Relu:
    def forward(self, input):
        self.input = input
        return input.maximum(torch.zeros_like(input))

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


class LossMSE:
    def forward(self, input):
        self.input = input
        return 1 / 2 * (input ** 2).sum()

    def backward(self, grad_output):
        return self.input * grad_output


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(19)

    x = torch.tensor([-.5, 1.5])
    print(x)

    relu = Relu()
    relu1 = Relu()
    loss = LossMSE()


    y = relu.forward(x)
    z = relu1.forward(y)
    cost = loss.forward(z)

    cost_grad = 1
    z_grad = loss.backward(cost_grad)
    y_grad = relu1.backward(z_grad)
    x_grad = relu.backward(y_grad)

    print(x_grad)


if __name__ == '__main__':
    main()
