from torch import nn

from run_experiments import run_experiments, Experiment


def build_simple_dense_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(2 * 14 * 14, 64),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


EXPERIMENT_MSE = Experiment(
    name="Dense MSE",
    epochs=80,
    build_model=build_simple_dense_model,
    build_loss=nn.MSELoss,
)

EXPERIMENT_BCE = Experiment(
    name="Dense BCE",
    epochs=80,
    build_model=build_simple_dense_model,
    build_loss=nn.BCELoss,
)

EXPERIMENT_MOSTLY_CONV = Experiment(
    name="Mostly conv",
    epochs=100,
    build_model=lambda: nn.Sequential(
        nn.Conv2d(2, 16, (5, 5)),
        nn.ReLU(),
        nn.Conv2d(16, 16, (5, 5)),
        nn.ReLU(),
        nn.Conv2d(16, 16, (5, 5)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2 * 2 * 16, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    ),
    build_loss=nn.BCELoss,
)

REPORT_EXPERIMENTS = [
    # EXPERIMENT_MSE,
    # EXPERIMENT_BCE,

    EXPERIMENT_MOSTLY_CONV,
]

if __name__ == '__main__':
    run_experiments("report", REPORT_EXPERIMENTS)
