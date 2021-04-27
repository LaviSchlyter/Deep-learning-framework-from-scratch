import os

import torch
from matplotlib import pyplot
from torch import nn, optim

from experiments import EXPERIMENTS, Experiment
from util import select_device, Data, load_data

DEVICE = select_device(debug_on_cpu=True)


def evaluate(
        model: nn.Module,
        x, y, y_float, y_digit,
        loss_func, aux_weight, aux_loss_func
):
    batch_size = len(x)

    output = model(x)

    if aux_weight != 0.0:
        assert aux_loss_func, "Aux weight != 0 but no aux loss func"

    if isinstance(output, tuple):
        y_pred, a_pred, b_pred = output
    else:
        y_pred = output
        a_pred = None
        b_pred = None

    y_pred = y_pred[:, 0]

    loss = loss_func(y_pred, y_float)
    if aux_loss_func is not None:
        loss += aux_weight * (
                aux_loss_func(a_pred, y_digit[:, 0]) +
                aux_loss_func(b_pred, y_digit[:, 1])
        )

        digit_acc = ((torch.argmax(a_pred, dim=1) == y_digit[:, 0]).sum() +
                     (torch.argmax(b_pred, dim=1) == y_digit[:, 1]).sum()).item() / (2 * batch_size)
    else:
        digit_acc = float("nan")

    acc = (((y_pred > 0.5) == y).sum() / batch_size).item()

    return loss, acc, digit_acc


def train_model(
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_func: nn.Module, aux_loss_func: nn.Module, aux_weight: float,
        data: Data, epochs: int
):
    plot_data = torch.zeros(epochs, 6)

    for e in range(epochs):
        model.train()
        train_loss, train_acc, train_digit_acc = evaluate(
            model,
            data.train_x, data.train_y, data.train_y_float, data.train_digit,
            loss_func, aux_weight, aux_loss_func
        )

        model.eval()
        test_loss, test_acc, test_digit_acc = evaluate(
            model,
            data.test_x, data.test_y, data.test_y_float, data.test_digit,
            loss_func, aux_weight, aux_loss_func
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plot_data[e, :] = torch.tensor([
            train_loss.item(), train_acc, train_digit_acc,
            test_loss.item(), test_acc, test_digit_acc,
        ])

    plot_legend = "train_loss", "train_acc", "train_digit_acc", "test_loss", "test_acc", "test_digit_acc"
    return plot_data, plot_legend


def run_experiment(experiment: Experiment, data_size: int, rounds: int, plot_loss: bool):
    all_plot_data = None
    plot_legend = None

    for round in range(rounds):
        print(f"Round {round}/{rounds}")

        data = load_data(data_size)
        if experiment.expand_flip:
            data.expand_train_flip()
        data.expand_train_transform(factor=experiment.expand_factor)
        data.shuffle_train()
        data.to(DEVICE)

        model, loss_func, aux_loss_func = experiment.build()
        model.to(DEVICE)

        optimizer = optim.Adam(model.parameters())

        plot_data, plot_legend = train_model(
            model=model, optimizer=optimizer, loss_func=loss_func, data=data,
            aux_loss_func=aux_loss_func, aux_weight=experiment.aux_weight,
            epochs=experiment.epochs
        )

        if all_plot_data is None:
            all_plot_data = torch.empty((rounds,) + plot_data.shape)
        all_plot_data[round] = plot_data

    # filter out loss data
    plot_mask = torch.tensor([plot_loss or not legend.endswith("_loss") for legend in plot_legend])
    all_plot_data = all_plot_data[:, :, plot_mask]
    plot_legend = [legend for i, legend in enumerate(plot_legend) if plot_mask[i]]

    plot_data_dev, plot_data_mean = torch.std_mean(all_plot_data, dim=0)
    plot_data_min, _ = torch.min(all_plot_data, dim=0)
    plot_data_max, _ = torch.max(all_plot_data, dim=0)

    # actually start plotting
    fig, ax = pyplot.subplots(1)
    ax.plot(plot_data_mean)

    if rounds > 1:
        ax.set_prop_cycle(None)
        ax.plot(plot_data_mean + plot_data_dev, '--', alpha=.5)
        ax.set_prop_cycle(None)
        ax.plot(plot_data_mean - plot_data_dev, '--', alpha=.5)
        # ax.set_prop_cycle(None)
        # ax.plot(plot_data_min, '--', alpha=.2)
        # ax.set_prop_cycle(None)
        # ax.plot(plot_data_max, '--', alpha=.2)

    pyplot.title(experiment.name)
    ax.legend(plot_legend)

    ax.set_xlabel("epoch")
    ax.xaxis.get_major_locator().set_params(integer=True)
    if not plot_loss:
        ax.set_ylim(0, 1)

    fig.savefig(f"output/{experiment.name}.png")
    fig.show()

    with open(f"output/{experiment.name}.txt", "w") as f:
        f.write("Final performance:\n")
        for i in range(len(plot_legend)):
            f.write(f"{plot_legend[i]} = {plot_data_mean[-1, i]:.3f} +- {plot_data_dev[-1, i]:.3f}"
                    f" (min={plot_data_min[-1, i]:.3f}, max={plot_data_max[-1, i]:.3f})\n")


def main():
    print(f"Running on device {DEVICE}")
    os.makedirs("output", exist_ok=True)

    rounds: int = 10
    data_size: int = 1000
    plot_loss: bool = False

    for ei, exp in enumerate(EXPERIMENTS):
        print(f"Running experiment {ei}/{len(EXPERIMENTS)}: {exp.name}")
        run_experiment(exp, data_size, rounds, plot_loss)


if __name__ == '__main__':
    main()
