# Implementation of Autograd
from matplotlib import pyplot as plt

from Project_2.modules import *
from Project_2.utils import *


# TODO remove plt and other libraries before handing
# TODO Add a gradient tester
# TODO Organize the files
# TODO Add an optimizer

def plot_performance(plot_data, plot_legend, print_loss=True):
    if not print_loss:
        plot_legend = plot_legend[:2]

    for i, legend in enumerate(plot_legend):
        plt.plot(plot_data[:, i], label=legend)
    plt.legend()
    plt.show()


# This is not a usable function !!
# Just to not run it each time
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


def basic_network():
    return Sequential([
        Linear(2, 50),
        Relu(),
        Linear(50, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ]
    )


class Weight_sharing:

    def __init__(self, train_input, test_input):
        self.test_input = test_input
        self.train_input = train_input
        # self.train_input_x = train_input[0]
        # self.train_input_y = train_input[1]
        # self.test_input_x = test_input[0]
        # self.test_input_a = test_input[1]


def evaluate_model(pred_y, true_y, loss_func):
    cost = loss_func(pred_y, true_y)
    error = evaluate_error(pred_y, true_y)
    return cost, error


def train_model(model, optimizer, loss_func, epoch, data, log_loss=True):
    plot_data = torch.empty(epoch, 4)
    plot_legend = ["train_error", "test_error", "train_loss", "test_loss"]
    n = data.train_x.shape[0] + data.test_x.shape[0]

    for e in range(epoch):
        optimizer.zero_grad()

        # Train evaluation
        y_train = model(data.train_x)
        # y_train_x = model(train_input_x)
        # y_train_y = model(train_input_y)
        cost_train, error_train = evaluate_model(y_train, data.train_y, loss_func)
        # y_train = model_(y_train_x.cat(y_train_y))

        # Test evaluation
        # y_test_x = model(test_input_x)
        # y_test_y = model(test_input_y)

        # y_test = model_(y_test_x.cat(y_test_y))
        y_test = model(data.test_x)
        cost_test, error_test = evaluate_model(y_test, data.test_y, loss_func)

        # Save values for data plotting
        plot_data[e, 0] = error_train
        plot_data[e, 1] = error_test
        plot_data[e, 2] = cost_train.value / n
        plot_data[e, 3] = cost_test.value / n
        cost_train.backward()
        optimizer.step()

        if (log_loss == True):
            print(format(
                f"For epoch = {e} with {type(loss_func).__name__}  ||  Training loss = {format(cost_train.value[0], '.4f')}    ,   Test loss = {format(cost_test.value[0], '.4f')}"))

    return plot_data, plot_legend


class Data:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.test_y = test_y
        self.test_x = test_x
        self.train_y = train_y
        self.train_x = train_x


def normalize(train_x, test_x):
    mean, std = train_x.value.mean(), train_x.value.std()
    train_x.value.sub_(mean).div_(std)
    test_x.value.sub_(mean).div_(std)

    return train_x, test_x


def main():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model = basic_network()
    # model_ = Sequential([Linear(2 * 10, 50), Relu(), Linear(50, 25), Relu(), Linear(25, 1), Sigmoid()])

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # This is for weightsharing, put in the weightsharing class
    train_input_x = train_input[0]
    train_input_y = train_input[1]
    test_input_x = test_input[0]
    test_input_y = test_input[1]

    loss = LossBCE()
    optimizer = Adam(model.param(), alpha=0.0005)
    # criterion = Adam(model.param() + model_.param())
    # criterion = SGD(model.param(), 0.05/n)
    epoch = 400
    plot_data, plot_legend = train_model(model, optimizer, loss, epoch, data, log_loss=True)

    plot_performance(plot_data, plot_legend, True)
    y_test = model(data.test_x)
    y_train = model(data.train_x)
    print("Test error: ", evaluate_error(y_test, test_target))
    print("Train error: ", evaluate_error(y_train, train_target))


if __name__ == '__main__':
    main()
