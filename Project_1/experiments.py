from dataclasses import dataclass
from typing import Callable, Optional

from torch import nn

from models import dense_network, WeightShareModel, full_conv_network, shared_conv_network, ProbOutputLayer


@dataclass
class Experiment:
    name: str
    epochs: int

    build_model: Callable[[], nn.Module]
    build_loss: Callable[[], nn.Module]

    aux_weight: float = 0.0
    build_aux_loss: Optional[Callable[[], nn.Module]] = None

    expand_factor: int = 1
    expand_flip: bool = False

    def build(self):
        return self.build_model(), self.build_loss(), None if self.build_aux_loss is None else self.build_aux_loss()


EXPERIMENT_DENSE_MSE = Experiment(
    name="Dense MSE",
    epochs=50,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_DENSE_BCE = Experiment(
    name="Dense BCE",
    epochs=50,

    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1], nn.ReLU(), nn.Sigmoid()),

    build_loss=lambda: nn.BCELoss(),
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
    epochs=10,

    build_model=lambda: full_conv_network(),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_CONV_SHARED = Experiment(
    name="Shared Conv + Dense",
    epochs=1000,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax()),
        dense_network([20, 1], None, nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
)

EXPERIMENT_CONV_SHARED_AUX = Experiment(
    name="Shared Conv + Dense, Aux",
    epochs=200,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        dense_network([20, 20, 1], nn.ReLU(), nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_CONV_SHARED_AUX_HEAD = Experiment(
    name="Shared Conv + Dense, Aux, Head",
    epochs=1000,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=10),
        output_head=dense_network([20, 1], nn.ReLU(), nn.Sigmoid()),
        digit_head=dense_network([10, 10], nn.ReLU(), nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENT_CONV_SHARED_AUX_HEAD_BIGGER = Experiment(
    name="Shared Conv + Dense, Aux, Head bigger",
    epochs=1000,

    build_model=lambda: WeightShareModel(
        shared_conv_network(nn.Softmax(), output_size=20),
        output_head=dense_network([40, 1], nn.ReLU(), nn.Sigmoid()),
        digit_head=dense_network([20, 10], nn.ReLU(), nn.Sigmoid())
    ),

    build_loss=lambda: nn.BCELoss(),
    aux_weight=1.0,
    build_aux_loss=lambda: nn.NLLLoss(),
)

EXPERIMENTS = [
    EXPERIMENT_DENSE_MSE,
    EXPERIMENT_DENSE_BCE,
    EXPERIMENT_DENSE_EXPAND,
    EXPERIMENT_DENSE_EXPAND_FLIP,
    EXPERIMENT_DENSE_SHARE,
    EXPERIMENT_DENSE_SHARE_PROB,
    EXPERIMENT_DENSE_SHARE_AUX,
    EXPERIMENT_DENSE_SHARE_AUX_PROB,
    EXPERIMENT_CONV,
    EXPERIMENT_CONV_SHARED,
    EXPERIMENT_CONV_SHARED_AUX,
    EXPERIMENT_CONV_SHARED_AUX_HEAD,
]
