# Implementation of Autograd

from Project_2.modules import *
from Project_2.util import *


# TODO Add different weight initializers


def basic_network_WS_1():
    return Sequential([
        Linear(1, 10),
        Relu(),
        Linear(10, 25),
        Relu(),
        Linear(25, 10),

    ])


def basic_network_WS_2():
    return Sequential([
        Linear(20, 40),
        Relu(),
        Linear(40, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ])


def basic_network_1():
    return Sequential([
        Linear(2, 10),
        Relu(),
        Linear(10, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()

    ])


def main_share():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model_1 = basic_network_WS_1()
    model_2 = basic_network_WS_2()

    model = WeightSharing(model_1, model_2)

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    # loss = LossBCE()
    loss = LossMSE()

    optimizer = Adam(model.param())
    # optimizer = SGD(model.param(), 0.03 / n)
    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=400, log_loss=True)

    plot_performance(plot_data, plot_legend, True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error weight sharing: ", evaluate_error(y_test, test_target))
    print("Train error weight sharing: ", evaluate_error(y_train, train_target))

def main():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model = basic_network_1()

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    # loss = LossBCE()
    loss = LossMSE()
    # optimizer = Adam(model.param(), alpha=0.0005)
    # optimizer = Adam(model.param())
    optimizer = SGD(model.param(), 1 / n)
    plot_data, plot_legend = train_model(model, optimizer, loss, data, epoch=400, log_loss=False)

    plot_performance(plot_data, plot_legend, True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error: ", evaluate_error(y_test, test_target))
    print("Train error: ", evaluate_error(y_train, train_target))


if __name__ == '__main__':
    main()
    main_share()


def plot_fig(train_target, train_input, model):
    train_mask = train_target.value[:, 0] > 0.5

    plt.figure(figsize=(7, 7))
    plt.scatter(train_input.value[train_mask, 0], train_input.value[train_mask, 1], color="coral")
    plt.scatter(train_input.value[~train_mask, 0], train_input.value[~train_mask, 1], color="c")
    xmin, xmax, ymin, ymax = plt.axis()
    # Generate the heatmap for uncertain predictions
    image_input_x, image_input_y = torch.meshgrid(torch.linspace(xmin, xmax, 1000), torch.linspace(ymin, ymax, 1000))
    image_input = torch.cat([image_input_x.reshape(-1, 1), image_input_y.reshape(-1, 1)], dim=1)

    image_output = model(Tensor(image_input))
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["c", "yellow", "coral"])
    plt.figure(figsize=(7, 7))
    plt.imshow(image_output.value.reshape(-1, 1000), cmap=cmap)
    import numpy as np
    plt.xticks(np.linspace(100, 901, 7), ["-1.5", "-1", "-0.5", "0.0", "0.5", "1.0", "1.5"])
    plt.yticks(np.linspace(100, 901, 7), ["1.5", "1", "0.5", "0.0", "-0.5", "-1.0", "-1.5"])
    plt.show()
