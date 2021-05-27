from torch import nn

from models import dense_network, WeightShareModel, basic_conv_network, shared_conv_network, ProbOutputLayer, \
    build_resnet, PreprocessModel
from run_experiments import run_experiments, Experiment
from util import InputNormalization

EXPERIMENT_DENSE_MSE = Experiment(
    name="Dense MSE",
    epochs=50,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),

    build_loss=lambda: nn.MSELoss(),
)

EXPERIMENT_DENSE_BCE = Experiment(
    name="Dense BCE",
    epochs=50,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),
    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_DENSE_BCE_BATCHED = Experiment(
    name="Dense BCE, batched",
    epochs=50,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),
    build_loss=lambda: nn.BCELoss(),

    batch_size=100,
)

EXPERIMENT_DENSE_INPUT_NORM_ELE = Experiment(
    name="Dense, norm elementwise",
    epochs=50,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),
    build_loss=lambda: nn.BCELoss(),

    input_normalization=InputNormalization.ElementWise,
)

EXPERIMENT_DENSE_INPUT_NORM_TOTAL = Experiment(
    name="Dense, norm total",
    epochs=50,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),
    build_loss=lambda: nn.BCELoss(),

    input_normalization=InputNormalization.Total,
)

EXPERIMENT_DENSE_EXPAND = Experiment(
    name="Dense, Expanded",
    epochs=50,
    expand_factor=2,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_DENSE_EXPAND_FLIP = Experiment(
    name="Dense, Expanded Flipped",
    epochs=50,
    expand_flip=True,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_DENSE_SHARE = Experiment(
    name="Shared Dense + Dense",
    epochs=100,

    build_model=lambda: WeightShareModel(
        dense_network([14 * 14, 255, 50, 10], nn.ReLU(), nn.Sigmoid()),
        dense_network([20, 1], nn.ReLU(), nn.Sigmoid()),
    ),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_DENSE_SHARE_PROB = Experiment(
    name="Shared Dense + Dense, Prob",
    epochs=100,

    build_model=lambda: WeightShareModel(
        dense_network([14 * 14, 255, 50, 10], nn.ReLU(), nn.Sigmoid()),
        ProbOutputLayer(),
    ),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_DENSE_SHARE_AUX = Experiment(
    name="Shared Dense + Dense, Aux",
    epochs=1000,

    build_model=lambda: WeightShareModel(
        dense_network([14 * 14, 255, 50, 10], nn.ReLU(), nn.Softmax()),
        dense_network([20, 1], None, nn.Sigmoid()),
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_DENSE_SHARE_AUX_PROB = Experiment(
    name="Shared Dense + Dense, Aux, Prob",
    epochs=150,

    build_model=lambda: WeightShareModel(
        dense_network([14 * 14, 255, 50, 10], nn.ReLU(), nn.Softmax()),
        ProbOutputLayer(),
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_CONV = Experiment(
    name="Conv",
    epochs=1000,

    build_model=lambda: basic_conv_network(),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_CONV_SHARED = Experiment(
    name="Shared Conv + Dense",
    epochs=1000,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        dense_network([20, 1], None, nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_CONV_SHARED_AUX = Experiment(
    name="Shared Conv + Dense, Aux 1.0",
    epochs=20,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

WEIGHT = .8

EXPERIMENT_CONV_SHARED_AUX_INV_10 = Experiment(
    name=f"Shared Conv + Dense, Aux {WEIGHT}",
    epochs=40,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=WEIGHT,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_CONV_SHARED_AUX_10 = Experiment(
    name="Shared Conv + Dense, Aux 10.0",
    epochs=20,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        dense_network([20, 20, 1], nn.ReLU(), nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=10.0,
    build_aux_loss=lambda: nn.NLLLoss(),

    batch_size=100,
)

EXPERIMENT_CONV_SHARED_AUX_INV_10_BATCH = Experiment(
    name="Shared Conv + Dense, Aux 0.1 batchnorm",
    epochs=100,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=0.1,
    build_aux_loss=lambda: nn.NLLLoss(),

    batch_size=100,
)

# TODO this may not be dropout any more
EXPERIMENT_CONV_SHARED_AUX_INV_10_DROP = Experiment(
    name="Shared Conv + Dense, Aux 0.1, Drop",
    epochs=100,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    weight_decay=0.0,

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),

    batch_size=100,
)

EXPERIMENT_CONV_SHARED_AUX_EXPAND = Experiment(
    name="Shared Conv + Dense, Aux, Expand",
    epochs=50,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        dense_network([20, 20, 1], nn.ReLU(), nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),

    expand_factor=4,
    expand_flip=False,
)

EXPERIMENT_CONV_SHARED_AUX_HEAD = Experiment(
    name="Shared Conv + Dense, Aux, Head",
    epochs=200,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        output_head=dense_network([20, 20, 1], nn.ReLU(), nn.Sigmoid()),
        digit_head=dense_network([10, 10], nn.ReLU(), nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_CONV_SHARED_AUX_HEAD_BIGGER = Experiment(
    name="Shared Conv + Dense, Aux, Head bigger",
    epochs=200,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=20),
        output_head=dense_network([40, 1], nn.ReLU(), nn.Sigmoid()),
        digit_head=dense_network([20, 10], nn.ReLU(), nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_RESNET = Experiment(
    name="Resnet, Shared, Aux",
    epochs=50,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        build_resnet(output_size=10, res=True),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_RESNET_RESLESS = Experiment(
    name="Resnet resless, Shared, Aux",
    epochs=50,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        build_resnet(output_size=10, res=False),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_RESNET_RESLESS_PROB = Experiment(
    name="Resnet resless, Shared, Aux, Prob",
    epochs=50,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        build_resnet(output_size=10, res=False),
        output_head=ProbOutputLayer(),
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_RESNET_RESLESS_MSE = Experiment(
    name="Resnet resless, Shared, Aux, MSE",
    epochs=50,
    batch_size=100,

    build_model=lambda: WeightShareModel(
        build_resnet(output_size=10, res=False),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=lambda: nn.MSELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_RESNET_RESLESS_DUP = Experiment(
    name="Resnet resless, Duplicated, Aux",
    epochs=50,
    batch_size=100,

    build_model=lambda: PreprocessModel(
        build_resnet(output_size=10, res=False),
        build_resnet(output_size=10, res=False),
        output_head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENTS_EXPLORE = [
    # EXPERIMENT_DENSE_MSE,
    # EXPERIMENT_DENSE_BCE,
    # EXPERIMENT_DENSE_BCE_BATCHED,
    # EXPERIMENT_DENSE_INPUT_NORM_ELE,
    # EXPERIMENT_DENSE_INPUT_NORM_TOTAL,
    # EXPERIMENT_DENSE_EXPAND,
    # EXPERIMENT_DENSE_EXPAND_FLIP,
    # EXPERIMENT_DENSE_SHARE,
    # EXPERIMENT_DENSE_SHARE_PROB,
    # EXPERIMENT_DENSE_SHARE_AUX,
    # EXPERIMENT_DENSE_SHARE_AUX_PROB,
    # EXPERIMENT_CONV,
    # EXPERIMENT_CONV_SHARED,
    # EXPERIMENT_CONV_SHARED_AUX,
    # EXPERIMENT_CONV_SHARED_AUX_10,
    # EXPERIMENT_CONV_SHARED_AUX_INV_10,
    # EXPERIMENT_CONV_SHARED_AUX_INV_10_DROP,
    # EXPERIMENT_CONV_SHARED_AUX_INV_10_DECAY,
    # EXPERIMENT_CONV_SHARED_AUX_EXPAND,
    # EXPERIMENT_CONV_SHARED_AUX_HEAD,

    EXPERIMENT_RESNET,
    # EXPERIMENT_RESNET_RESLESS,
    # EXPERIMENT_RESNET_RESLESS_DUP,
    # EXPERIMENT_RESNET_RESLESS_MSE,
    # EXPERIMENT_RESNET_RESLESS_PROB,
]

if __name__ == '__main__':
    run_experiments("explore", rounds=1, plot_titles=True, plot_loss=True, experiments=EXPERIMENTS_EXPLORE)
