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
    """ Compute the losses, the digit and final accuracy

    :param model: Model used
    :param x: Tensor input into the network
    :param y: Prediction from the network
    :param y_float: Target value
    :param y_digit: Target digit value
    :param loss_func: Loss function used
    :param aux_weight: Importance weight on auxiliary loss
    :param aux_loss_func: Auxiliary loss function
    :return: Loss, digit and final accuracies
    """
    batch_size = len(x)
    output = model(x)

    if not math.isnan(aux_weight):
        assert aux_loss_func, "Aux weight is set but no aux loss func given"

    # separate the digit predictions if the network returns them
    if isinstance(output, tuple):
        y_pred, a_pred, b_pred = output

        assert not torch.any(torch.isnan(y_pred)), "found nan value in a_pred"
        assert not torch.any(torch.isnan(y_pred)), "found nan value in a_pred"
    else:
        y_pred = output
        a_pred = None
        b_pred = None

    # some sanity checks on the output
    assert not torch.any(torch.isnan(y_pred)), "found nan value in y_pred"
    assert y_pred.shape[1] == 1, f"final prediction should have size 1, was {y_pred.shape}"
    y_pred = y_pred[:, 0]

    loss = loss_func(y_pred, y_float)

    # acc the aux loss if applicable
    if aux_loss_func is not None:
        if isinstance(aux_loss_func, nn.NLLLoss):
            assert a_pred.shape[1] == 10, f"digit prediction should have size 10, was {a_pred.shape}"
            assert b_pred.shape[1] == 10, f"digit prediction should have size 10, was {b_pred.shape}"

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
        data: Data, epochs: int, batch_size: int,
):
    """ Training the model
    The model is trained and evaluated at each epoch

    :param model: Module model
    :param optimizer: Method to optimize the parameters
    :param loss_func: Loss function
    :param aux_loss_func: Auxiliary loss function
    :param aux_weight: Auxiliary weight
    :param data: Data structures containing the data
    :param epochs: Number of epochs for training
    :param batch_size: Number of samples processed before updating the model, or -1 to use the entire dataset at once.
    :return: Tensor containing the evaluation at each epoch + legends for plotting
    """
    if batch_size == -1:
        batch_size = len(data.train_x)
    batch_count = len(data.train_x) // batch_size

    plot_data = torch.zeros(epochs, 6)

    for e in range(epochs):
        # train step
        data.shuffle_train()

        for bi in range(batch_count):
            batch_range = slice(bi * batch_size, (bi + 1) * batch_size)

            model.train()
            batch_train_loss, _, _ = evaluate_model(
                model,
                data.train_x[batch_range], data.train_y[batch_range], data.train_y_float[batch_range],
                data.train_digit[batch_range],
                loss_func, aux_weight, aux_loss_func
            )

            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()

        # train evaluation
        model.eval()
        train_loss, train_acc, train_digit_acc = evaluate_model(
            model,
            data.train_x, data.train_y, data.train_y_float, data.train_digit,
            loss_func, aux_weight, aux_loss_func
        )

        # test evaluation
        model.eval()
        test_loss, test_acc, test_digit_acc = evaluate_model(
            model,
            data.test_x, data.test_y, data.test_y_float, data.test_digit,
            loss_func, aux_weight, aux_loss_func
        )

        plot_data[e, :] = torch.tensor([
            train_acc, train_digit_acc,
            test_acc, test_digit_acc,
            train_loss.item(), test_loss.item(),
        ])

        print(f"Epoch {e + 1}/{epochs}:"
              f" train_loss={train_loss:.4}, train_acc={train_acc:.4}"
              f" test_loss={test_loss:.4}, test_acc={test_acc:.4}")

    plot_legend = "train_acc", "train_digit_acc", "test_acc", "test_digit_acc", "train_loss", "test_loss"
    return plot_data, plot_legend
