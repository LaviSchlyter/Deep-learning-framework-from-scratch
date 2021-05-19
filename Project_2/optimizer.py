from abc import ABC, abstractmethod

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