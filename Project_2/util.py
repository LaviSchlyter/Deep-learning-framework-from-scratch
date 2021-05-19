import math
import matplotlib.pyplot as plt
import torch

from core import HyperCube


def generate_disc_set(nb):
    input_ = torch.empty(nb, 2).uniform_()
    target = ((input_ - 0.5).pow(2).sum(1) < 1 / (2 * math.pi)).float()
    return HyperCube(input_), HyperCube(target[:, None])


def evaluate_error(pred, target):
    return 1 - ((pred.value > 0.5) == (target.value > 0.5)).sum() / len(pred.value)


def normalize(train_x, test_x):
    mean, std = train_x.value.mean(), train_x.value.std()
    train_x.value.sub_(mean).div_(std)
    test_x.value.sub_(mean).div_(std)

    return train_x, test_x


class Data:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.test_y = test_y
        self.test_x = test_x
        self.train_y = train_y
        self.train_x = train_x


def plot_performance(plot_data, plot_legend, print_loss=True):
    if not print_loss:
        plot_legend = plot_legend[:2]

    for i, legend in enumerate(plot_legend):
        plt.plot(plot_data[:, i], label=legend)
    plt.legend()
    plt.show()


def train_model(model, optimizer, loss_func, data, epoch=400, log_loss=True):
    plot_data = torch.empty(epoch, 4)
    plot_legend = ["train_error", "test_error", "train_loss", "test_loss"]
    n = data.train_x.shape[0] + data.test_x.shape[0]

    for e in range(epoch):

        optimizer.zero_grad()
        # Train evaluation

        y_train = model(data.train_x)
        cost_train, error_train = evaluate_model(y_train, data.train_y, loss_func)

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


def evaluate_model(pred_y, true_y, loss_func):
    cost = loss_func(pred_y, true_y)
    error = evaluate_error(pred_y, true_y)
    return cost, error
