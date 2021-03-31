import torch


class Tensor:
    def __init__(self, value, grad_fn=None):
        self.value = value
        self.grad = torch.zeros_like(value)
        self.grad_fn = grad_fn

    def backward(self, output_grad=None):
        if output_grad is None:
            output_grad = torch.ones_like(self.value)

        self.grad = torch.zeros_like(self.value)

        if self.grad_fn is not None:
            self.grad_fn.backward(output_grad)


class ReluGradFn:
    def __init__(self, input):
        self.input = input

    def backward(self, output_grad):
        input_grad = output_grad * (self.input > 0)
        self.input.backward(input_grad)
        self.input.grad += input_grad


class Relu:
    def __call__(self, input):
        return Tensor(
            value=input.maximum(torch.zeros_like(input)),
            grad_fn=ReluGradFn(input)
        )


class LossMSE:
    def __call__(self, input):
        self.input = input
        return 1 / 2 * (input ** 2).sum()

    def backward(self, grad_output):
        return self.input * grad_output


def main():
    torch.set_grad_enabled(True)
    torch.manual_seed(19)

    x = Tensor(torch.tensor([-.5, 1.5]))

    relu = Relu()
    loss = LossMSE()

    y = relu(x)
    cost = loss(y)

    cost.backward()
    print(x.grad)


if __name__ == '__main__':
    main()
