import math
import os

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

    @classmethod
    def generate(cls, n: int):
        train_input, train_target = generate_disc_set(n)
        test_input, test_target = generate_disc_set(n)

        train_input, test_input = normalize(train_input, test_input)

        return Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)


PLOT_LEGEND = ["train_error", "test_error", "train_loss", "test_loss"]


def train_model(model, optimizer, loss_func, data, epoch, log_epochs):
    plot_data = torch.empty(epoch, len(PLOT_LEGEND))
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

        if log_epochs:
            loss_name = type(loss_func).__name__
            print(
                f"For epoch = {e} with {loss_name}: "
                f"train_loss = {cost_train.item():.4f}, "
                f"train_error={error_train:.4f}, "
                f"test_loss = {cost_test.item():.4f}, "
                f"test_error={error_test:.4f}"
            )

    return plot_data


def evaluate_model(pred_y, true_y, loss_func):
    cost = loss_func(pred_y, true_y)
    error = evaluate_error(pred_y, true_y)
    return cost, error


def run_experiment(
        name: str, rounds: int, n: int,
        build_model, build_optimizer, loss_func, epochs, log_epochs: bool,
        extra_plots: bool
):
    print(f"Running experiment {name}")

    all_plot_data = torch.zeros(rounds, epochs, len(PLOT_LEGEND))

    model, data = None, None

    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")

        data = Data.generate(n)

        model = build_model()
        optimizer = build_optimizer(model.param())

        plot_data = train_model(model, optimizer, loss_func, data, epochs, log_epochs)
        all_plot_data[round, :, :] = plot_data

    plot_experiment(name, model, data, all_plot_data, extra_plots)


def plot_experiment(name: str, model, data, all_plot_data, extra_plots):
    os.makedirs(f"data/{name}", exist_ok=True)

    plot_data_std, plot_data_mean = torch.std_mean(all_plot_data, dim=0)
    plot_data_max = all_plot_data[:, -1, :].max(dim=0)
    plot_data_min = all_plot_data[:, -1, :].max(dim=0)

    # print the final test/train error
    print(f"Train error: {plot_data_mean[-1, 0]:.4f} +- {plot_data_std[-1, 0]:.4f}, min/max: {plot_data_min[0]}/{plot_data_max[0]}")
    print(f"Test error: {plot_data_mean[-1, 1]:.4f} +- {plot_data_std[-1, 1]:.4f}, min/max: {plot_data_min[1]}/{plot_data_max[1]}")

    # training plot
    fig, ax = plt.subplots(1)
    ax.plot(plot_data_mean)

    if len(all_plot_data) > 1:
        ax.set_prop_cycle(None)
        ax.plot(plot_data_mean + plot_data_std, '--', alpha=.5)
        ax.set_prop_cycle(None)
        ax.plot(plot_data_mean - plot_data_std, '--', alpha=.5)

    ax.legend(PLOT_LEGEND)
    ax.set_xlabel("epoch")
    ax.xaxis.get_major_locator().set_params(integer=True)

    fig.savefig(f"data/{name}/training.png")
    fig.show()

    if extra_plots:
        # scatter plot
        fig = plt.figure(figsize=(7, 7))

        y_train_pred = model(data.train_x)
        y_train_bool_pred = (y_train_pred.value > 0.5).squeeze(1)
        plt.scatter(data.train_x.value[y_train_bool_pred, 0], data.train_x.value[y_train_bool_pred, 1], color="red")
        plt.scatter(data.train_x.value[~y_train_bool_pred, 0], data.train_x.value[~y_train_bool_pred, 1], color="blue")

        xmin, xmax, ymin, ymax = plt.axis()
        fig.savefig(f"data/{name}/distribution_points.png")
        fig.show()

        # heatmap
        image_input_x, image_input_y = torch.meshgrid(
            torch.linspace(xmin, xmax, 1000),
            torch.linspace(ymin, ymax, 1000)
        )
        image_input = torch.cat([image_input_x.reshape(-1, 1), image_input_y.reshape(-1, 1)], dim=1)
        image_output = model(HyperCube(image_input))
        import matplotlib
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "yellow", "red"])
        fig = plt.figure(figsize=(7, 7))
        plt.imshow(image_output.value.reshape(-1, 1000), cmap=cmap)
        import numpy as np
        plt.xticks(np.linspace(100, 901, 7), ["-1.5", "-1", "-0.5", "0.0", "0.5", "1.0", "1.5"])
        plt.yticks(np.linspace(100, 901, 7), ["1.5", "1", "0.5", "0.0", "-0.5", "-1.0", "-1.5"])
        fig.savefig(f"data/{name}/density_correct.png")
        plt.show()
