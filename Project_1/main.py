import math
from dataclasses import dataclass, fields

import torch
from matplotlib import pyplot
from torch import nn, optim

from dlc_practical_prologue import generate_pair_sets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Data:
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_y_float: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_y_float: torch.Tensor

    def to(self, device):
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name).to(device))


def train_model(model: nn.Module, optimizer: optim.Optimizer, loss_func: nn.Module, data: Data, epochs: int):
    plot_data = torch.zeros(epochs, 4)
    plot_legend = "train_loss", "train_acc", "test_loss", "test_acc"

    for e in range(epochs):
        print(f"Starting epoch {e}")
        model.train()

        train_y_pred = model.forward(data.train_x)[:, 0]
        train_loss = loss_func(train_y_pred, data.train_y_float)
        train_acc = ((train_y_pred > 0.5) == data.train_y).sum() / math.prod(data.train_y.shape)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        test_y_pred = model.forward(data.test_x)[:, 0]
        test_loss = loss_func(test_y_pred, data.test_y_float)
        test_acc = ((test_y_pred > 0.5) == data.test_y).sum() / math.prod(data.test_y.shape)

        plot_data[e, :] = torch.tensor([train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item()])

    return plot_data, plot_legend


def main():
    print(f"Running on device {DEVICE}")

    train_x, train_y, _, test_x, test_y, _ = generate_pair_sets(1000)
    data = Data(
        train_x=train_x, train_y=train_y, train_y_float=train_y.float(),
        test_x=test_x, test_y=test_y, test_y_float=test_y.float(),
    )
    data.to(DEVICE)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2 * 14 * 14, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 1),
        nn.Sigmoid(),
    )
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters())
    loss_func = nn.MSELoss()

    plot_data, plot_legend = train_model(
        model=model, optimizer=optimizer, loss_func=loss_func, data=data,
        epochs=20
    )

    pyplot.figure()
    pyplot.plot(plot_data)
    pyplot.legend(plot_legend)
    pyplot.show()


if __name__ == '__main__':
    main()
