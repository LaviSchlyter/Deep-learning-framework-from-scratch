import math
import os

import torch
from matplotlib import pyplot
from torch import nn, optim

from experiments import EXPERIMENTS, Experiment
from util import select_device, Data, load_data

DEVICE = select_device(debug_on_cpu=True)


def train_model(model: nn.Module, optimizer: optim.Optimizer, loss_func: nn.Module, data: Data, epochs: int):
    plot_data = torch.zeros(epochs, 4)
    plot_legend = "train_loss", "train_acc", "test_loss", "test_acc"

    for e in range(epochs):
        model.train()

        train_y_pred = model(data.train_x)[:, 0]
        train_loss = loss_func(train_y_pred, data.train_y_float)
        train_acc = ((train_y_pred > 0.5) == data.train_y).sum() / math.prod(data.train_y.shape)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        test_y_pred = model(data.test_x)[:, 0]
        test_loss = loss_func(test_y_pred, data.test_y_float)
        test_acc = ((test_y_pred > 0.5) == data.test_y).sum() / math.prod(data.test_y.shape)

        plot_data[e, :] = torch.tensor([train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item()])

    return plot_data, plot_legend


def run_experiment(experiment: Experiment, data_size: int, rounds: int):
    all_plot_data = None
    plot_legend = None

    for round in range(rounds):
        print(f"Round {round}/{rounds}")

        data = load_data(data_size)
        data.expand_train_data(factor=experiment.expand_factor)
        data.to(DEVICE)

        model, loss_func = experiment.build()
        model.to(DEVICE)

        optimizer = optim.Adam(model.parameters())

        plot_data, plot_legend = train_model(
            model=model, optimizer=optimizer, loss_func=loss_func, data=data,
            epochs=experiment.epochs
        )

        if all_plot_data is None:
            all_plot_data = torch.empty((rounds,) + plot_data.shape)
        all_plot_data[round] = plot_data

    plot_data_dev, plot_data_mean = torch.std_mean(all_plot_data, dim=0)
    plot_data_min, _ = torch.min(all_plot_data, dim=0)
    plot_data_max, _ = torch.max(all_plot_data, dim=0)

    fig, ax = pyplot.subplots(1)

    ax.plot(plot_data_mean)
    ax.set_prop_cycle(None)
    ax.plot(plot_data_mean + plot_data_dev, '--', alpha=.5)
    ax.set_prop_cycle(None)
    ax.plot(plot_data_mean - plot_data_dev, '--', alpha=.5)
    ax.set_prop_cycle(None)
    ax.plot(plot_data_min, '--', alpha=.2)
    ax.set_prop_cycle(None)
    ax.plot(plot_data_max, '--', alpha=.2)

    pyplot.title(experiment.name)
    ax.legend(plot_legend)

    ax.set_xlabel("epoch")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylim(0, 1)

    fig.savefig(f"output/{experiment.name}.png")
    fig.show()

    with open(f"output/{experiment.name}.txt", "w") as f:
        f.write("Final performance:\n")
        for i in range(len(plot_legend)):
            f.write(f"{plot_legend[i]} = {plot_data_mean[i, -1]:.3f} +- {plot_data_dev[i, -1]:.3f}"
                    f" (min={plot_data_min[i, -1]:.3f}, max={plot_data_max[i, -1]:.3f})\n")


def main():
    print(f"Running on device {DEVICE}")
    os.makedirs("output", exist_ok=True)

    rounds = 10
    data_size = 1000

    for ei, exp in enumerate(EXPERIMENTS):
        print(f"Running experiment {ei}/{len(EXPERIMENTS)}: {exp.name}")
        run_experiment(exp, data_size, rounds)


if __name__ == '__main__':
    main()
