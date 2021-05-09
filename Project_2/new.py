import math
from abc import abstractmethod

import torch
from matplotlib import pyplot as plt
# TODO How to make this without torch.tensor
        # TODO Check with Karel assisant message on Slack


class Module:

    @abstractmethod
    def __call__(self, input_):
        pass

    @abstractmethod
    def param(self):
        pass


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input_):
        self.intermediates = []

        for layer in self.layers:
            self.intermediates.append(input_)
            input_ = layer(input_)

        self.intermediates.append(input_)

        return input_

    def param(self):
        par = []
        for layer in self.layers:
            par.extend(layer.param())

        return par




# Tensor
class Tensor:

    def __init__(self, value, grad_fn=None):

        assert not isinstance(value, Tensor)
        self.value = value
        # self.grad = torch.zeros_like(value)
        self.grad = Tensor.zeros_like(value)
        self.grad_fn = grad_fn

    def backward(self, output_grad=None):

        if output_grad is None:
            # output_grad = torch.ones_like(self.value)
            output_grad = Tensor.ones_like(self.value)
        if self.grad_fn is not None:
            self.grad_fn.backward(output_grad)

        self.grad += output_grad

    def zero_grad(self):
        #self.grad = torch.zeros_like(self.value)
        self.grad = Tensor.zeros_like(self.value)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = slice(item, item + 1)
        return Tensor(self.value[:, item], grad_fn=SliceGradFn(self, item))

    def cat(self, tensor):

        size_cat = [self.value.shape[0], self.value.shape[1] + tensor.value.shape[1]]
        concat = torch.empty(size_cat)
        concat[:, 0] = self.value[:, 0]
        concat[:, 1] = tensor.value[:, 0]

        # damn = [self.value, tensor.value]

        # return Tensor(torch.cat((self.value, tensor.value), 1), grad_fn= CatGradFn(self, tensor))
        return Tensor(concat, grad_fn=CatGradFn(self, tensor))

    def zeros_like(self):
        return torch.empty(self.shape).zero_()

    def ones_like(self):
        return Tensor.zeros_like(self) + 1

    @property
    def shape(self):
        return self.value.shape


# Layers

class Linear(Module):

    def __init__(self, Din, Dout):
        self.W = Tensor(torch.empty([Din, Dout]).uniform_() * (2 / math.sqrt(Din)) - 1 / math.sqrt(Din))
        # self.W = Tensor(torch.rand([Din, Dout]) * (2/math.sqrt(Din)) - 1/math.sqrt(Din))
        self.b = Tensor(torch.empty([Dout]).uniform_())
        # self.b = Tensor(torch.rand(([Dout])))

    def __call__(self, input_):
        # Takes input_ of shape NxD and W of DxM
        return Tensor(input_.value @ self.W.value + self.b.value, grad_fn=LinearGradFn(input_, self.W, self.b))

    def param(self):
        return [self.W, self.b]


class Tanh(Module):

    def __call__(self, input_):
        return Tensor(
            value=2 * Sigmoid.sigmoid(input_.value) - 1,
            grad_fn=TanhGradFn(input_)
        )

    def param(self):
        return []


class Sigmoid(Module):

    def __call__(self, input_):
        return Tensor(
            value=Sigmoid.sigmoid(input_.value),
            grad_fn=SigmoidGradFn(input_)
        )

    def param(self):
        return []

    @staticmethod
    def sigmoid(input_):
        # return 1 / (1 + torch.exp(-input_))
        return 1 / (1 + (-input_).exp())


class Relu(Module):

    def __call__(self, input_):
        return Tensor(
            # value=torch.maximum(torch.zeros_like(input_.value), input_.value),
            # TODO Fix maximum, am I allowed to do this
            value=Tensor.zeros_like(input_).maximum(input_.value),
            grad_fn=ReluGradFn(input_)
        )

    # Same but to be able to write in the two notations
    @staticmethod
    def forward(input_):
        return Tensor(

            value=Tensor.zeros_like(input_).maximum(input_),
            grad_fn=ReluGradFn(input_)
        )

    def param(self):
        return []


# GradFN

class SliceGradFn:
    def __init__(self, input_, item):
        self.input_ = input_
        self.item = item

    def backward(self, output_grad):
        # input_grad = torch.zeros_like(self.input_.value)
        input_grad = Tensor.zeros_like(self.input_.value)
        input_grad[:, self.item] = output_grad
        self.input_.backward(input_grad)


class CatGradFn:
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

    def backward(self, output_grad):
        input_grad_1 = output_grad[:, :(self.input1.shape[1])]
        input_grad_2 = output_grad[:, (self.input1.shape[1]):]
        self.input1.backward(input_grad_1)
        self.input2.backward(input_grad_2)


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


class ReluGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = output_grad * (self.input_.value > 0)
        self.input_.backward(input_grad)
        # self.input_.grad += input_grad


class SigmoidGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        output = Sigmoid.sigmoid(self.input_.value)
        input_grad = output_grad * output * (1 - output)
        self.input_.backward(input_grad)


class TanhGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = output_grad * (1 - pow(self.input_.value, 2))
        self.input_.backward(input_grad)
        self.input_.grad += input_grad
        return input_grad


# Optimizers (SGD, Adam)
class Optim:

    def __init__(self, params):
        self.params = params  # model parameters

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class Adam(Optim):
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

    def step(self):
        m = [0] * len(self.params)
        v = [0] * len(self.params)

        for i, param in enumerate(self.params):
            # gt = param.grad
            m[i] = self.beta1 * m[i] + (1 - self.beta1) * param.grad
            v[i] = self.beta2 * v[i] + (1 - self.beta2) * param.grad ** 2
            m_hat = m[i] / (1 - self.beta1)
            v_hat = v[i] / (1 - self.beta2)
            param.value -= self.alpha * m_hat / (v_hat.sqrt() + self.epsilon)


class SGD(Optim):
    def __init__(self, params, lr):
        """

        :param params: Parameters that will be updated
        :param lr: The learning rate
        """
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad


# Loss functions (MSE, cross entropy)

class LossMSE:

    def __call__(self, input_, target):
        return Tensor(1 / 2 * ((input_.value - target.value) ** 2).sum(dim=0), grad_fn=LossMSEGradFN(input_, target))

    @staticmethod
    def param():
        return []


def evaluate(pred, target):
    return 1 - ((pred.value > 0.5) == (target.value > 0.5)).sum() / len(pred.value)


def plot_performance(plot_data, plot_legend, print_loss = True):

    if not print_loss:
        plot_legend = plot_legend[:2]

    for i, legend in enumerate(plot_legend):

        plt.plot(plot_data[:, i], label=legend)
    plt.legend()
    plt.show()


class LossMSEGradFN:
    def __init__(self, input_, target):
        self.target = target
        self.input_ = input_

    def backward(self, output_grad):
        input_grad = (self.input_.value - self.target.value) * output_grad
        self.input_.backward(input_grad)
        self.target.backward(-input_grad)

# TODO Add cross entropy loss ....
class LossCrossEntropy:

    def __call__(self, input_, target):
        return -(input_.value*math.log(target)).sum()

    @staticmethod
    def param():
        return []




def generate_disc_set(nb):
    input_ = torch.empty(nb, 2).uniform_()
    target = ((input_ - 0.5).pow(2).sum(1) < 1 / (2 * math.pi)).float()
    return Tensor(input_), Tensor(target[:, None])


def main():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)
    # torch.manual_seed(19)

    model = Sequential([Linear(1, 50), Relu(), Linear(50, 25), Relu(), Linear(25, 1), Relu()])
    model_ = Sequential([Linear(2 * 1, 50), Relu(), Linear(50, 25), Relu(), Linear(25, 1), Sigmoid()])

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    mean, std = train_input.value.mean(), train_input.value.std()
    train_input.value.sub_(mean).div_(std)
    test_input.value.sub_(mean).div_(std)

    train_input_x = train_input[0]
    train_input_y = train_input[1]
    test_input_x = test_input[0]
    test_input_y = test_input[1]

    train_mask = train_target.value[:, 0] > 0.5
    plt.scatter(train_input.value[train_mask, 0], train_input.value[train_mask, 1])
    plt.scatter(train_input.value[~train_mask, 0], train_input.value[~train_mask, 1])
    plt.show()

    loss = LossMSE()
    criterion = Adam(model.param() + model_.param())
    # criterion = SGD(model.param(), 0.05/n)

    epoch = 500
    plot_data = torch.empty(epoch, 4)
    plot_legend = ["train_error", "test_error", "train_loss", "test_loss"]
    for e in range(epoch):
        criterion.zero_grad()

        # Train evaluation

        y_train_x = model(train_input_x)
        y_train_y = model(train_input_y)

        y_train = model_(y_train_x.cat(y_train_y))

        cost_train = loss(y_train, train_target)
        error_train = evaluate(y_train, train_target)

        # Test evaluation
        y_test_x = model(test_input_x)
        y_test_y = model(test_input_y)

        y_test = model_(y_test_x.cat(y_test_y))
        # y_test = model(test_input)
        cost_test = loss(y_test, test_target)
        error_test = evaluate(y_test, test_target)

        # Save values for data plotting
        #plot_data[e,:] = [error_train, error_test, cost_train.value / n, cost_test.value / n]

        plot_data[e, :] = torch.tensor([error_train, error_test, cost_train.value / n, cost_test.value / n])

        cost_train.backward()
        criterion.step()
        # print(f"For epoch = {e} with MSE loss = {cost.value}")



    plot_performance(plot_data, plot_legend, False)
    print("test error: ", evaluate(y_test, test_target))

    print("train error: ", evaluate(y_train, train_target))

    test = torch.empty(3, 1)
    test2 = torch.empty(3, 1)
    print(test, test2)
    print(test2.maximum(test))


if __name__ == '__main__':
    main()
