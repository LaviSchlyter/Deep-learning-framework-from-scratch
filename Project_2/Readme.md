This is our implementation of a deep learning framework.

The framework itself is located in the `hyper` folder:
* `core` are the core classes including `HyperCube`, our `Tensor` wrapper.
* `modules` contains various layers types, activation function, the `Sequential` module and their `GradFn` implementations.
* `loss` contains different loss functions and their `GradFn` implementations.
* `optimizer` contains different optimizer implementations. 

The rest of this project is demo code for the `hyper` framework:µ
* `test` is the main file that reproduces the figures and training shown in our report.
* `architectures` defines the actual networks used.
* `util` contains a number of utility functions for model training, data generation, plotting, ...
* `gradient_verification` is a small program we used to check our gradient implementations: 
  it compares the gradient computed during backpropagation to gradients calculated using numerical differentiation

# VM Bug

We found that sometimes running our code on the VM Segfaults during a `torch.exp(x)` operation. The VM is not running out of memory, and we don't really know what else could cause this. We found doing `math.e ** x` stops this issue from occuring for us, but we're not sure how consistent that is.
