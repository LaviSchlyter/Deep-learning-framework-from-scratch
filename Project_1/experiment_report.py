from torch import nn

from models import PreprocessModel, WeightShareModel
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


def build_digit_conv_network():
    return nn.Sequential(
        nn.Conv2d(1, 32, (5, 5)),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, (5, 5)),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        nn.Linear(64, 50),
        nn.ReLU(),
        nn.BatchNorm1d(50),
        nn.Linear(50, 10),
        nn.Softmax(),
    )


EXPERIMENT_CONV_AUX_DUPLICATED = Experiment(
    name="Conv + Aux, duplicated",
    epochs=40,
    batch_size=100,

    build_model=lambda: PreprocessModel(
        a_input_module=build_digit_conv_network(),
        b_input_module=build_digit_conv_network(),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=nn.BCELoss,
    aux_weight=1.0,
    build_aux_loss=nn.NLLLoss,

)

EXPERIMENT_CONV_AUX_SHARED = Experiment(
    name="Conv + Aux, shared",
    epochs=40,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        input_module=build_digit_conv_network(),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=nn.BCELoss,
    aux_weight=1.0,
    build_aux_loss=nn.NLLLoss,
)

REPORT_EXPERIMENTS = [
    # EXPERIMENT_MSE,
    # EXPERIMENT_BCE,

    # EXPERIMENT_MOSTLY_CONV,

    EXPERIMENT_CONV_AUX_DUPLICATED,
    EXPERIMENT_CONV_AUX_SHARED,
]

if __name__ == '__main__':
    run_experiments("report", REPORT_EXPERIMENTS)
