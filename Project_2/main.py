# Implementation of Autograd
import matplotlib.pyplot as plt

from core import *
from loss import *
from optimizer import *
from modules import *
from util import *
# Set to true for plots
VERBOSE_PLOTS = True
# TODO Video
# TODO make simple architectures
# TODO Output plot reports (labels, legends, no title)

def network_WS_1():
    return Sequential([
        Linear(1, 10),
        Relu(),
        Linear(10, 25),
        Relu(),
        Linear(25, 10),
    ])
def network_WS_2():
    return Sequential([
        Linear(20, 40),
        Relu(),
        Linear(40, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ])

def network_1():
    return Sequential([
        Linear(2, 25),
        Relu(),
        Linear(25, 25),
        Relu(),
        Linear(25,25),
        Relu(),
        Linear(25, 1),
        Sigmoid()

    ])

def network_2():
    return Sequential([
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25,1),
        Tanh(),
        Linear(25, 1),
        Sigmoid()

    ])

def network_3():
    return Sequential([
        Linear(2, 25),
        Sigmoid(),
        Linear(25, 25),
        Sigmoid(),
        Linear(25,1),
        Sigmoid(),
        Linear(25, 1),
        Sigmoid()

    ])

def main_WS_MSE_SGD():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model_1 = network_WS_1()
    model_2 = network_WS_2()
    # Combining the models
    model = WeightSharing(model_1, model_2)

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    loss = LossMSE()

    # Learning rate
    lr = 0.3 / n
    # L2 penalty
    lambda_ = 0
    optimizer = SGD(model.param(), lr=lr, lambda_=lambda_)
    # Training model
    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=250, log_loss=True)
    if (VERBOSE_PLOTS):
        plot_performance(plot_data, plot_legend, print_loss=True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error weight sharing: ", evaluate_error(y_test, test_target))
    print("Train error weight sharing: ", evaluate_error(y_train, train_target))


def main_MSE_SGD():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)
    model = network_1()

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    loss = LossMSE()
    optimizer = SGD(model.param(), 0.3 / n, lambda_= 0)

    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=300, log_loss=True)
    if (VERBOSE_PLOTS):
        plot_performance(plot_data, plot_legend, print_loss=True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error: ", evaluate_error(y_test, test_target))
    print("Train error: ", evaluate_error(y_train, train_target))
    train_mask = train_target.value[:, 0] > 0.5

    fig = plt.figure(figsize=(7, 7))
    plt.scatter(train_input.value[train_mask, 0], train_input.value[train_mask, 1], color="red")
    plt.scatter(train_input.value[~train_mask, 0], train_input.value[~train_mask, 1], color="blue")
    xmin, xmax, ymin, ymax = plt.axis()
    fig.savefig("distribution_points.png")
    # Generate the heatmap for uncertain predictions
    image_input_x, image_input_y = torch.meshgrid(torch.linspace(xmin, xmax, 1000), torch.linspace(ymin, ymax, 1000))
    image_input = torch.cat([image_input_x.reshape(-1, 1), image_input_y.reshape(-1, 1)], dim=1)

    image_output = model(HyperCube(image_input))
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "yellow", "red"])
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(image_output.value.reshape(-1, 1000), cmap=cmap)
    import numpy as np
    plt.xticks(np.linspace(100, 901, 7), ["-1.5", "-1", "-0.5", "0.0", "0.5", "1.0", "1.5"])
    plt.yticks(np.linspace(100, 901, 7), ["1.5", "1", "0.5", "0.0", "-0.5", "-1.0", "-1.5"])
    plt.show()
    fig.savefig("density_correct.png")

def main_BCE_SGD():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)
    model = network_1()

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    loss = LossBCE()
    optimizer = SGD(model.param(), 0.3 / n, lambda_=0)

    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=300, log_loss=True)
    if (VERBOSE_PLOTS):
        plot_performance(plot_data, plot_legend, print_loss=True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error: ", evaluate_error(y_test, test_target))
    print("Train error: ", evaluate_error(y_train, train_target))


def main_MSE_Adam():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)
    model = network_1()

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    loss = LossMSE()
    optimizer = Adam(model.param())

    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=300, log_loss=True)
    if (VERBOSE_PLOTS):
        plot_performance(plot_data, plot_legend, print_loss=True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error: ", evaluate_error(y_test, test_target))
    print("Train error: ", evaluate_error(y_train, train_target))



def main_BCE_Adam():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)
    model = network_1()

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    loss = LossBCE()
    optimizer = Adam(model.param())

    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=300, log_loss=True)
    if (VERBOSE_PLOTS):
        plot_performance(plot_data, plot_legend, print_loss=True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error: ", evaluate_error(y_test, test_target))
    print("Train error: ", evaluate_error(y_train, train_target))

def main_WS_MSE_Adam():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model_1 = network_WS_1()
    model_2 = network_WS_2()
    # Combining the models
    model = WeightSharing(model_1, model_2)

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    loss = LossMSE()
    optimizer = Adam(model.param())
    # Training model
    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=200, log_loss=True)
    if (VERBOSE_PLOTS):
        plot_performance(plot_data, plot_legend, print_loss=True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error weight sharing: ", evaluate_error(y_test, test_target))
    print("Train error weight sharing: ", evaluate_error(y_train, train_target))


if __name__ == '__main__':
    main_MSE_SGD()
    #main_BCE_SGD()
    #main_MSE_Adam()
    #main_BCE_Adam()
    #main_WS_MSE_SGD()
    #main_WS_MSE_Adam()






