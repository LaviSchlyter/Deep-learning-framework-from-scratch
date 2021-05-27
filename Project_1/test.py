from torch import nn

from experiment_report import build_conv_model
from models import WeightShareModel
from run_experiments import run_experiments, Experiment

EXPERIMENT = Experiment(
    name="Shared Aux w1",
    epochs=70,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        input_module=build_conv_model(1, 10, True, 0.1, 0.5),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=nn.BCELoss,
    aux_weight=1,
    build_aux_loss=nn.NLLLoss,
)


def main():
    run_experiments("test", 3, plot_titles=True, plot_loss=True, experiments=[EXPERIMENT])


if __name__ == '__main__':
    main()
