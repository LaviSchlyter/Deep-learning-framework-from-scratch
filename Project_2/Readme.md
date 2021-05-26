This is our implementation of a deep learning framework.

The framework itself is located in the `hyper` folder:
* `core` are the core classes including `HyperCube`, our `Tensor` wrapper.
* `modules` contains various layers types, activation function, the `Sequential` module and their `GradFn` implementations.
* `loss` contains different loss functions and their `GradFn` implementations.
* `optimizer` contains different optimizer implementations. 

The rest of this project is demo code for the `hyper` framework:µ
* `test` is the main file that reproduces the figures and training shown in our report.
* `architectures` defines the actual networks used.
* `util` contains a bunch of utility functions for model training, data generation, plotting, ...
* `gradient_verification` is a small program we used to check our gradient implementations: 
  it compares the gradient computed during backpropagation to gradients calculated using numerical differentiation

