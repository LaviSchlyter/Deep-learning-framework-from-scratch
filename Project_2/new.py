import math
from abc import abstractmethod

import torch


# TODO Adam
# TODO Make an exponential GradFN with overloaded division, sum and multiplication
# TODO why better with normalizing when circle
# TODO try weight sharing // Try the sum of Tensor // or try concatenation

from matplotlib import pyplot as plt


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

    def evaluate(self, prediction, labels):
        pass


# Tensor
class Tensor:

    def __init__(self, value, grad_fn=None):

        assert not isinstance(value, Tensor)
        self.value = value
        self.grad = torch.zeros_like(value)
        self.grad_fn = grad_fn


    def backward(self, output_grad=None):

        if output_grad is None:
            output_grad = torch.ones_like(self.value)

        if self.grad_fn is not None:
            self.grad_fn.backward(output_grad)

        self.grad += output_grad


    def zero_grad(self):
        self.grad = torch.zeros_like(self.value)

    @property
    def shape(self):
        return self.value.shape


# Layers

class Linear(Module):

    def __init__(self, Din, Dout):
        self.W = Tensor(torch.rand([Din, Dout]) * (2/math.sqrt(Din)) - 1/math.sqrt(Din))
        self.b = Tensor(torch.rand(([Dout])))

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
        return 1 / (1 + torch.exp(-input_))


class Relu(Module):

    def __call__(self, input_):
        return Tensor(
            value=torch.maximum(torch.zeros_like(input_.value), input_.value),
            grad_fn=ReluGradFn(input_)
        )

    # Same but to be able to write in the two notations
    @staticmethod
    def forward(input_):
        return Tensor(
            value=torch.maximum(torch.zeros_like(input_), input_),
            grad_fn=ReluGradFn(input_)
        )

    def param(self):
        return []


# GradFN
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
        input__grad = output_grad * (self.input_.value > 0)
        self.input_.backward(input__grad)
        #self.input_.grad += input__grad



class SigmoidGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        output = Sigmoid.sigmoid(self.input_.value)
        input__grad = output_grad * output * (1 - output)
        self.input_.backward(input__grad)


class TanhGradFn:

    def __init__(self, input_):
        self.input_ = input_

    def backward(self, output_grad):
        input__grad = output_grad * (1 - pow(self.input_.value, 2))
        self.input_.backward(input__grad)
        self.input_.grad += input__grad
        return input__grad


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
        m = 0
        v = 0

        for param in self.params:
            # gt = param.grad
            m = self.beta1 * m + (1 - self.beta1) * param.grad
            v = self.beta2 * v + (1 - self.beta2) * param.grad ** 2
            m_hat = m / (1 - self.beta1)
            v_hat = v / (1 - self.beta2)
            param.value -= self.alpha * m_hat / (math.sqrt(v_hat) + self.epsilon)


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
        return Tensor(1 / 2 * ((input_.value - target.value) ** 2).sum(dim = 0), grad_fn=LossMSEGradFN(input_, target))

    def param(self):
        return []

def evaluate(pred, target):
        return 1 - ( (pred.value > 0.5) == (target.value > 0.5)).sum() / len(pred.value)

def plot_performance(plot_data, plot_legend):

    pass

class LossMSEGradFN:
    def __init__(self, input_, target):
        self.target = target
        self.input_ = input_

    def backward(self, output_grad):
        input__grad = (self.input_.value - self.target.value) * output_grad
        self.input_.backward(input__grad)
        self.target.backward(-input__grad)



def generate_disc_set(nb):
    input_ = torch.empty(nb, 2).uniform_()
    target = ((input_ - 0.5).pow(2).sum(1) < 1 / (2 * math.pi)).float()
    return Tensor(input_), Tensor(target[:, None])


def main():

    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)
    #torch.manual_seed(19)

    model = Sequential([Linear(2, 25), Relu(), Linear(25,20), Relu(), Linear(20, 1), Sigmoid()])

    # Generate the set of size n
    n = 5000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    mean, std = train_input.value.mean(), train_input.value.std()
    train_input.value.sub_(mean).div_(std)
    test_input.value.sub_(mean).div_(std)

    train_mask = train_target.value[:, 0] > 0.5
    plt.scatter(train_input.value[train_mask, 0], train_input.value[train_mask, 1])
    plt.scatter(train_input.value[~train_mask, 0], train_input.value[~train_mask, 1])
    plt.show()

    loss = LossMSE()
    criterion = SGD(model.param(), 5/n)

    epoch = 1000
    plot_data = torch.zeros(epoch, 4)
    plot_legend = "train_loss", "train_error", "test_loss", "test_error"
    for e in range(epoch):
        criterion.zero_grad()

        # Train evaluation

        y_train = model(train_input)
        cost_train = loss(y_train, train_target)
        error_train = evaluate(y_train, train_target)

        # Test evaluation
        y_test = model(test_input)
        cost_test = loss(y_test, test_target)
        error_test = evaluate(y_test, test_target)

        # Save values for data plotting
        plot_data[e, :] = torch.tensor([cost_train.value, error_train, cost_test.value, error_test])

        cost_train.backward()
        # print(train_input.value[0:5, :])
        # print(train_input.grad[0:5, :])
        # print(train_input.grad.shape)

        wrong_index = ((y_train.value > 0.5) != (train_target.value > 0.5)).nonzero()

        i = wrong_index[0, 0]

        print("input & grad")
        print(train_input.value[i:i+4])
        print(train_input.grad[i:i+4])
        print(train_mask[i:i+4])
        print(y_train.value[i:i+4])


        criterion.step()
        #print(f"For epoch = {e} with MSE loss = {cost.value}")


    y_test = model(test_input)

    plt.plot(plot_data[:,0])
    plt.savefig("test.png")
    plt.show()
    print("test error", evaluate(y_test, test_target))

    y_train = model(train_input)
    print("train error: ",evaluate(y_train, train_target))



if __name__ == '__main__':
    main()