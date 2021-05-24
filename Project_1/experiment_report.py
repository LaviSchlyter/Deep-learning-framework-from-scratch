from torch import nn

from models import PreprocessModel, WeightShareModel
from run_experiments import run_experiments, Experiment


def build_simple_dense_model(dropout=0.0):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(2 * 14 * 14, 64),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


def build_simple_dense_model_smaller(dropout=0.0):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(2 * 14 * 14, 32),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


EXPERIMENT_MSE = Experiment(
    name="Dense MSE",
    epochs=20,
    batch_size=100,
    build_model=build_simple_dense_model,
    build_loss=nn.MSELoss,
)

EXPERIMENT_BCE = Experiment(
    name="Dense BCE",
    epochs=20,
    batch_size=100,
    build_model=build_simple_dense_model,
    build_loss=nn.BCELoss,
)

EXPERIMENT_BCE_SMALLER = Experiment(
    name="Dense BCE",
    epochs=20,
    batch_size=100,
    build_model=build_simple_dense_model_smaller,
    build_loss=nn.BCELoss,
)

EXPERIMENT_BCE_REG = Experiment(
    name="Dense BCE Regularized",
    epochs=20,
    batch_size=100,
    build_model=lambda: build_simple_dense_model(dropout=0.5),
    build_loss=nn.BCELoss,
    weight_decay=0.8,
)


def build_conv_model(batch_norm: bool, conv_dropout: float, linear_dropout: float):
    return nn.Sequential(
        nn.Conv2d(2, 16, (3, 3)),
        nn.Dropout(conv_dropout),
        *[nn.BatchNorm2d(16)] * batch_norm,
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(16, 16, (3, 3)),
        nn.Dropout(conv_dropout),
        *[nn.BatchNorm2d(16)] * batch_norm,
        nn.MaxPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(2 * 2 * 16, 32),
        nn.Dropout(linear_dropout),
        *[nn.BatchNorm1d(32)] * batch_norm,
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


EXPERIMENTS_CONV = [
    Experiment(
        name="Conv",
        epochs=80,
        build_model=lambda: build_conv_model(False, 0.0, 0.0),
        build_loss=nn.BCELoss,
    ),
    Experiment(
        name="Conv + BatchNorm",
        epochs=80,
        build_model=lambda: build_conv_model(True, 0.0, 0.0),
        build_loss=nn.BCELoss,
    ),
    Experiment(
        name="Conv + Dropout",
        epochs=400,
        build_model=lambda: build_conv_model(False, 0.0, 0.5),
        build_loss=nn.BCELoss,
    ),
    Experiment(
        name="Conv + BatchNorm + Dropout",
        epochs=160,
        build_model=lambda: build_conv_model(True, 0.0, 0.5),
        build_loss=nn.BCELoss,
    )
]


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
    # EXPERIMENT_BCE_REG,
    # EXPERIMENT_BCE_SMALLER,

    *EXPERIMENTS_CONV,

    # EXPERIMENT_CONV_AUX_DUPLICATED,
    # EXPERIMENT_CONV_AUX_SHARED,

]


def main():
    run_experiments("report", rounds=3, plot_titles=False, experiments=REPORT_EXPERIMENTS)


if __name__ == '__main__':
    main()
