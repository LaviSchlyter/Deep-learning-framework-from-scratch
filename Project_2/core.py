import torch

# TODO: Verify the comments
class HyperCube:
    # "Our" Tensor class

    def __init__(self, value, grad_fn=None):
        assert not isinstance(value, HyperCube)
        self.value = value
        self.grad = zeros_like(value)
        self.grad_fn = grad_fn

    def backward(self, output_grad=None):
        """ Recursive calls back on the computational graph and accumulation of gradients

        :param output_grad: The loss w.r.t to the HyperCube we are interesting in
        """
        if output_grad is None:
            output_grad = ones_like(self.value)
        if self.grad_fn is not None:
            self.grad_fn.backward(output_grad)

        self.grad += output_grad

    def zero_grad(self):
        """ Set gradients to zero
        """
        self.grad = zeros_like(self.value)

    def __getitem__(self, item):
        """ [] overloading

        :param item: Column index at which slicing should end
        :return: Hypercube with the kept columns and a grad_fn field set to corresponding operation
        """
        if isinstance(item, int):
            item = slice(item, item + 1)
        return HyperCube(self.value[:, item], grad_fn=SliceGradFn(self, item))

    def cat(self, hypercube):
        """ Concatenation 

        :param hypercube: "tensor" from the HyperCube class
        :return: concatenation of two HyperCubes on the second dimension (dim = 1)
        """

        size_cat = [self.value.shape[0], self.value.shape[1] + hypercube.value.shape[1]]
        concat = torch.empty(size_cat)
        concat[:, :self.shape[1]] = self.value
        concat[:, self.shape[1]:] = hypercube.value
        return HyperCube(concat, grad_fn=CatGradFn(self, hypercube))

    @property
    def shape(self):
        return self.value.shape

    def item(self):
        # Return the number in a single element HyperCube
        return self.value.item()


class CatGradFn:
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

    def backward(self, output_grad):
        """ Backpropagation on each part of the HyperCube separately

        :param output_grad: Gradient of the loss w.r.t the output
        """
        input_grad_1 = output_grad[:, :(self.input1.shape[1])]
        input_grad_2 = output_grad[:, (self.input1.shape[1]):]
        self.input1.backward(input_grad_1)
        self.input2.backward(input_grad_2)


def zeros_like(tensor):
    """
    :param tensor:PyTorch tensor
    :return:PyTorch tensor of same shape as tensor but filled with zeros
    """
    return torch.empty(tensor.shape).zero_()


def ones_like(tensor):
    """
    :param tensor: PyTorch tensor
    :return: PyTorch tensor of same shape as tensor but filled with ones
    """
    return zeros_like(tensor) + 1


class SliceGradFn:
    def __init__(self, input_, item):
        self.input_ = input_
        self.item = item

    def backward(self, output_grad):
        input_grad = zeros_like(self.input_.value)
        input_grad[:, self.item] = output_grad
        self.input_.backward(input_grad)


