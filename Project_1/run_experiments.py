"""
The experiment runner infrastructure.
This enables us to test different architectures with different settings for many rounds in an easily reproducible way.
"""

import os
from dataclasses import dataclass
from math import prod
from typing import Callable, Optional

import torch
# from matplotlib import pyplot
from torch import optim, nn

from core import train_model
from util import load_data, DEVICE, InputNormalization  # , set_plot_font_size


@dataclass
class Experiment:
    """
    The settings for an experiment. This includes the model, the loss function, training parameters,
    data augmentation settings, ...
    The model and loss functions are functions that need to be called to crate new instances, this is to ensure that
    different rounds start with a newly initialized model.
    """

    name: str
    epochs: int

    build_model: Callable[[], nn.Module]  # TODO: what is this ?

    build_loss: Callable[[], nn.Module]
    weight_decay: float = 0
    aux_weight: float = float("nan")
    build_aux_loss: Optional[Callable[[], nn.Module]] = None

    expand_factor: int = 1
    expand_flip: bool = False
    input_normalization: InputNormalization = InputNormalization.No

    batch_size: int = -1

    def build(self):
        """Build a new model and loss function."""

        return self.build_model(), self.build_loss(), None if self.build_aux_loss is None else self.build_aux_loss()


def run_experiment(
        base_name: str,
        experiment: Experiment,
        data_size: int, rounds: int,
        plot_loss: bool, plot_titles: bool
):
    """
    Run a single `experiment` `rounds` times.

    :param base_name: Name of session
    :param experiment: Name of current experiment
    :param data_size: Size of the data
    :param rounds: Number of rounds to run the experiment for.
    :param plot_loss: Boolean on whether to plot the loss
    :param plot_titles: Boolean on whether to plot the titles
    """
    all_plot_data = None
    plot_legend = None

    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")

        data = load_data(data_size, experiment.input_normalization)
        if experiment.expand_flip:
            data.expand_train_flip()
        data.expand_train_transform(factor=experiment.expand_factor)
        data.shuffle_train()
        data.to(DEVICE)

        model, loss_func, aux_loss_func = experiment.build()
        model.to(DEVICE)

        if round == 0:
            weight_count = sum(prod(param.shape) for param in model.parameters() if param.requires_grad)
            print(f"Model has {weight_count} weights")

        optimizer = optim.AdamW(model.parameters(), weight_decay=experiment.weight_decay)

        plot_data, plot_legend = train_model(
            model=model, optimizer=optimizer, loss_func=loss_func, data=data,
            aux_loss_func=aux_loss_func, aux_weight=experiment.aux_weight,
            epochs=experiment.epochs, batch_size=experiment.batch_size
        )

        if all_plot_data is None:
            all_plot_data = torch.empty((rounds,) + plot_data.shape)
        all_plot_data[round] = plot_data

    # Filter out loss data
    plot_mask = torch.tensor([plot_loss or not legend.endswith("_loss") for legend in plot_legend])
    all_plot_data = all_plot_data[:, :, plot_mask]
    plot_legend = [legend for i, legend in enumerate(plot_legend) if plot_mask[i]]

    plot_data_dev, plot_data_mean = torch.std_mean(all_plot_data, dim=0)
    plot_data_min, _ = torch.min(all_plot_data, dim=0)
    plot_data_max, _ = torch.max(all_plot_data, dim=0)

    # Plots # TODO: Remove before submission
    # fig, ax = pyplot.subplots(1)
    # ax.plot(plot_data_mean)

    # if rounds > 1:
    #     ax.set_prop_cycle(None)
    #     ax.plot(plot_data_mean + plot_data_dev, '--', alpha=.5)
    #     ax.set_prop_cycle(None)
    #     ax.plot(plot_data_mean - plot_data_dev, '--', alpha=.5)

    # if plot_titles:
    #     pyplot.title(experiment.name)
    # ax.legend(plot_legend)

    # ax.set_xlabel("epoch")
    # ax.set_ylabel("accuracy")
    # ax.xaxis.get_major_locator().set_params(integer=True)
    # ax.set_ylim(0, 1)

    # fig.savefig(f"output/{base_name}/{experiment.name}.png", bbox_inches='tight', pad_inches=0.1)
    # fig.show()

    # Saving performances onto file
    with open(f"output/{base_name}/{experiment.name}.txt", "w") as f:
        f.write(f"Trained for {experiment.epochs} epochs\n")

        f.write("Final performance:\n")
        for i in range(len(plot_legend)):
            f.write(f"{plot_legend[i]} = {plot_data_mean[-1, i]:.3f} +- {plot_data_dev[-1, i]:.3f}"
                    f" (min={plot_data_min[-1, i]:.3f}, max={plot_data_max[-1, i]:.3f})\n")


def run_experiments(
        base_name: str,
        rounds: int,
        plot_titles: bool, plot_loss: bool,
        experiments: [Experiment]
):
    """
    The experiment runner entry point. Runs each `experiment` `rounds` times. Show and saves a plot of the
    training process and a text file with final accuracies and losses into `output/{base_name}/{experiment.name}/`.
    `plot_titles` and `plot_loss` control whether the plots get titles and loss curves.
    """
    # set_plot_font_size()

    print(f"Running experiments '{base_name}'")
    os.makedirs(f"output/{base_name}", exist_ok=True)

    data_size: int = 1000

    print(f"Running for {rounds} rounds with data_size {data_size}")

    for ei, exp in enumerate(experiments):
        print(f"Running experiment {ei + 1}/{len(experiments)}: {exp.name}")
        run_experiment(base_name, exp, data_size, rounds, plot_loss, plot_titles)
