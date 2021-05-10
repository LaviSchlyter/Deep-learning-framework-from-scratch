import math
from typing import Optional

import torch
from torch import nn, optim

from util import Data


def evaluate_model(
        model: nn.Module,
        x, y, y_float, y_digit,
        loss_func, aux_weight, aux_loss_func
):
    batch_size = len(x)

    output = model(x)

    if not math.isnan(aux_weight):
        assert aux_loss_func, "Aux weight is set but no aux loss func given"

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
        loss_func: nn.Module, aux_loss_func: Optional[nn.Module], aux_weight: float,
        data: Data, epochs: int
):
    plot_data = torch.zeros(epochs, 6)

    for e in range(epochs):
        model.train()
        train_loss, train_acc, train_digit_acc = evaluate_model(
            model,
            data.train_x, data.train_y, data.train_y_float, data.train_digit,
            loss_func, aux_weight, aux_loss_func
        )

        model.eval()
        test_loss, test_acc, test_digit_acc = evaluate_model(
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
