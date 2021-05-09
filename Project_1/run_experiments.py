import os

import torch
from matplotlib import pyplot
from torch import optim

from core import train_model
from experiments import EXPERIMENTS, Experiment
from util import select_device, load_data

DEVICE = select_device(debug_on_cpu=True)


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
