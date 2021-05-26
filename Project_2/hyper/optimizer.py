from abc import ABC, abstractmethod


class Optimizer(ABC):
    """ Base class for all optimizer implementations. """

    def __init__(self, params):
        """ Crate a new optimizer that will optimize the given parameters. """
        self.params = params

    def zero_grad(self):
        """ Clear the accumulated gradient for all parameters. """
        for param in self.params:
            param.zero_grad()

    @abstractmethod
    def step(self):
        """
        Perform a single optimization iteration,
        updating the parameter values using their accumulated gradients.
        """
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
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # initialize iteration counter and moment estimates
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
        :param params: Parameters of the model which are to be optimized
        :param lr: The learning rate
        :param lambda_: The L2 regularization weight used to penalize large weights.
        """
        super().__init__(params)
        self.lr = lr
        self.lambda_ = lambda_

    def step(self):
        for param in self.params:
            param.value -= self.lr * (param.grad + self.lambda_ * param.value)
