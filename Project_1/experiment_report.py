""" Experiments reported in the project

The following framework is used in order to produce the networks described in the report
"""

from models import PreprocessModel, WeightShareModel, shared_resnet, ProbOutputLayer
from run_experiments import run_experiments, Experiment
from torch import nn


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


# Same model as above with twice less nodes
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
    # TODO Probably add smaller in the name ?
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


def build_conv_model(
        input_channels: int, output_size: int,
        batch_norm: bool,
        conv_dropout: float, linear_dropout: float
):
    """ Building a convolutional model

    :param batch_norm: Whether to include bath normalization or not
    :param conv_dropout: Ratio of feature detectors  to drop in the convolutional layers
    :param linear_dropout: Ratio of feature detectors  to drop in the linear layers
    :return: Sequential containing the layers passed
    """

    if output_size == 1:
        final_activation = nn.Sigmoid()
    else:
        final_activation = nn.Softmax()

    return nn.Sequential(
        nn.Conv2d(input_channels, 16, (3, 3)),
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
        nn.Linear(32, output_size),
        final_activation
    )


EXPERIMENT_CONV = Experiment(
    name="Conv",
    epochs=80,
    build_model=lambda: build_conv_model(2, 1, False, 0.0, 0.0),
    build_loss=nn.BCELoss,
)
EXPERIMENT_CONV_BN = Experiment(
    name="Conv + BatchNorm",
    epochs=80,
    build_model=lambda: build_conv_model(2, 1, True, 0.0, 0.0),
    build_loss=nn.BCELoss,
)
EXPERIMENT_CONV_DROP = Experiment(
    name="Conv + Dropout",
    epochs=400,
    build_model=lambda: build_conv_model(2, 1, False, 0.1, 0.5),
    build_loss=nn.BCELoss,
)
EXPERIMENT_CONV_DROP_BN = Experiment(
    name="Conv + BatchNorm + Dropout",
    epochs=160,
    build_model=lambda: build_conv_model(2, 1, True, 0.1, 0.5),
    build_loss=nn.BCELoss,
)

EXPERIMENT_CONV_FLIP = Experiment(
    name="Conv + Flipped",
    epochs=80,
    build_model=lambda: build_conv_model(2, 1, False, 0.0, 0.0),
    build_loss=nn.BCELoss,
    expand_flip=True,
)

# TODO: what do you mean by duplicated
EXPERIMENT_CONV_DUPLICATED = Experiment(
    name="Duplicated",
    epochs=20,
    batch_size=100,

    build_model=lambda: PreprocessModel(
        a_input_module=build_conv_model(1, 10, True, 0.1, 0.5),
        b_input_module=build_conv_model(1, 10, True, 0.1, 0.5),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=nn.BCELoss,
)
EXPERIMENT_CONV_SHARED = Experiment(
    name="Shared",
    epochs=20,
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
)

AUX_EPOCHS = 70

EXPERIMENT_CONV_DUPLICATED_AUX = Experiment(
    name="Duplicated Aux",
    epochs=AUX_EPOCHS,
    batch_size=100,

    build_model=lambda: PreprocessModel(
        a_input_module=build_conv_model(1, 10, True, 0.1, 0.5),
        b_input_module=build_conv_model(1, 10, True, 0.1, 0.5),
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
EXPERIMENT_CONV_SHARED_AUX = Experiment(
    name="Shared Aux",
    epochs=AUX_EPOCHS,
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
EXPERIMENT_CONV_SHARED_AUX_LESS = Experiment(
    name="Shared Aux w0.1",
    epochs=AUX_EPOCHS,
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
    aux_weight=0.1,
    build_aux_loss=nn.NLLLoss,
)
EXPERIMENT_CONV_SHARED_AUX_MORE = Experiment(
    name="Shared Aux w10",
    epochs=AUX_EPOCHS,
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
    aux_weight=10,
    build_aux_loss=nn.NLLLoss,
)
EXPERIMENT_CONV_SHARED_AUX_MORE_MORE = Experiment(
    name="Shared Aux w100",
    epochs=AUX_EPOCHS,
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
    aux_weight=100,
    build_aux_loss=nn.NLLLoss,
)

EXPERIMENT_RESNET = Experiment(
    name="Resnet",
    epochs=120,
        batch_size=100,

        build_model=lambda: WeightShareModel(
            input_module=shared_resnet(10, True),
            output_head=nn.Sequential(
                nn.Flatten(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 1),
                nn.Sigmoid(),
            )
        ),

        build_loss=nn.BCELoss,
        aux_weight=10,
        build_aux_loss=nn.NLLLoss,
    )

EXPERIMENT_RESNET_RESLESS = Experiment(
    name="Resnet resless",
    epochs=40,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        input_module=shared_resnet(10, False),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=nn.BCELoss,
    aux_weight=10,
    build_aux_loss=nn.NLLLoss,
)

EXPERIMENT_RESNET_RESLESS_PROB = Experiment(
    name="Resnet resless, Probability output",
    epochs=40,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        input_module=shared_resnet(10, False),
        output_head=ProbOutputLayer()
    ),

    build_loss=nn.BCELoss,
    aux_weight=10,
    build_aux_loss=nn.NLLLoss,
)

REPORT_EXPERIMENTS = [
    EXPERIMENT_BCE,
    EXPERIMENT_MSE,
    # EXPERIMENT_BCE_REG,
    # EXPERIMENT_BCE_SMALLER,
    EXPERIMENT_CONV,
    EXPERIMENT_CONV_FLIP,
    EXPERIMENT_CONV_BN,
    # EXPERIMENT_CONV_DROP,
    # EXPERIMENT_CONV_DROP_BN,

    EXPERIMENT_CONV_DUPLICATED,
    EXPERIMENT_CONV_SHARED,

    # EXPERIMENT_CONV_DUPLICATED_AUX,
    EXPERIMENT_CONV_SHARED_AUX_LESS,
    # EXPERIMENT_CONV_SHARED_AUX,
    EXPERIMENT_CONV_SHARED_AUX_MORE,
    # EXPERIMENT_CONV_SHARED_AUX_MORE_MORE,

    EXPERIMENT_RESNET,
    EXPERIMENT_RESNET_RESLESS,
    # EXPERIMENT_RESNET_RESLESS_PROB,
]


def main():
    run_experiments("report", rounds=10, plot_titles=False, experiments=REPORT_EXPERIMENTS)


if __name__ == '__main__':
    main()
