import os

import torch
from matplotlib import pyplot
from torch import nn, optim

from core import train_model, evaluate_model
from run_experiments import DEVICE
from util import load_data, InputNormalization

nan = float("nan")


def build_network(dropout_p: float, depth: int, size: int):
    layers = [nn.Flatten()]
    prev_size = 2 * 14 * 14

    for i in range(depth):
        layers.append(nn.Linear(prev_size, size))
        layers.append(nn.ReLU())
        if dropout_p != 0.0:
            layers.append(nn.BatchNorm1d(size))
        prev_size = size

    layers.append(nn.Linear(prev_size, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def calculate_different_params(rounds: int, epochs: int, dropout_p: float, weight_decay: float, depths, sizes):
    assert len(depths) == len(sizes)
    param_count = len(depths)

    full_result_test_acc = torch.zeros(param_count, rounds)
    full_result_train_acc = torch.zeros(param_count, rounds)

    for i, (depth, size) in enumerate(zip(depths, sizes)):
        print(f"Trying size={size}, depth={depth}")
        for round in range(rounds):
            print(f"Starting round {round}")
            data = load_data(1000, InputNormalization.No)
            data.to(DEVICE)

            model = build_network(dropout_p, depth, size)
            model.to(DEVICE)

            optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
            loss_func = nn.BCELoss()

            train_model(
                model=model, optimizer=optimizer,
                loss_func=loss_func, aux_loss_func=None, aux_weight=nan,
                data=data, epochs=epochs
            )

            _, train_acc, _ = evaluate_model(
                model,
                data.train_x, data.train_y, data.train_y_float, data.train_digit,
                loss_func, nan, None
            )
            _, test_acc, _ = evaluate_model(
                model,
                data.test_x, data.test_y, data.test_y_float, data.test_digit,
                loss_func, nan, None
            )
            full_result_train_acc[i, round] = train_acc
            full_result_test_acc[i, round] = test_acc

    return full_result_train_acc, full_result_test_acc


def make_plot(result_train_acc, result_test_acc, x_label: str, x_values, title: str):
    test_acc_std, test_acc_mean = torch.std_mean(result_test_acc, dim=1)
    train_acc_std, train_acc_mean = torch.std_mean(result_train_acc, dim=1)
    fig, ax = pyplot.subplots(1)

    ax.plot(x_values, train_acc_mean, label="train_acc")
    ax.plot(x_values, test_acc_mean, label="test_acc")
    ax.set_prop_cycle(None)
    pyplot.plot(x_values, train_acc_mean - train_acc_std, '--', alpha=.5)
    pyplot.plot(x_values, test_acc_mean - test_acc_std, '--', alpha=.5)
    ax.set_prop_cycle(None)
    pyplot.plot(x_values, train_acc_mean + train_acc_std, '--', alpha=.5)
    pyplot.plot(x_values, test_acc_mean + test_acc_std, '--', alpha=.5)

    ax.set_xlabel(x_label)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 1)
    ax.legend()
    ax.set_title(title)

    os.makedirs("output/plots", exist_ok=True)
    pyplot.savefig(f"output/plots/acc_vs_{x_label}.png")
    pyplot.show()


def main():
    rounds = 10
    epochs = 100
    dropout = 0.0
    weight_decay = 0.0

    depths = [0, 1, 2, 3, 4]
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]

    train_acc, test_acc = calculate_different_params(
        rounds, epochs, dropout, weight_decay,
        depths, [100] * len(depths)
    )
    make_plot(train_acc, test_acc, "depth", depths, f"batch norm")

    # train_acc,    test_acc = calculate_different_params(
    #     rounds, epochs, dropout, weight_decay,
    #     [2] * len(sizes), sizes
    # )
    # make_plot(train_acc, test_acc, "size", sizes)


if __name__ == '__main__':
    main()
