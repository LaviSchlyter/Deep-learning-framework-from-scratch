from dataclasses import dataclass
from typing import Callable

from torch import nn

from models import dense_network, WeightShareModel, full_conv_network, shared_conv_network


@dataclass
class Experiment:
    name: str
    build_model: Callable[[], nn.Module]
    build_loss: Callable[[], nn.Module] = lambda: nn.MSELoss()
    epochs: int = 20
    expand_factor: int = 1

    def build(self):
        return self.build_model(), self.build_loss()


EXPERIMENT_DENSE = Experiment(
    name="Dense",
    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1]),
)

EXPERIMENT_DENSE_EXPAND = Experiment(
    name="Dense",
    build_model=lambda: dense_network([2 * 14 * 14, 255, 50, 1]),
    expand_factor=10
)

EXPERIMENT_DENSE_SHARE = Experiment(
    name="Weightshare Dense + Dense",
    build_model=lambda: WeightShareModel(
        dense_network([14 * 14, 255, 50, 10]),
        dense_network([20, 1]),
    ),
)

EXPERIMENT_CONV = Experiment(
    name="Conv",
    build_model=lambda: full_conv_network(),
)

EXPERIMENT_CONV_SHARED = Experiment(
    name="Conv",
    build_model=lambda: WeightShareModel(
        shared_conv_network(),
        dense_network([20, 1])
    ),
)

EXPERIMENTS = [
    EXPERIMENT_DENSE,
    EXPERIMENT_DENSE_EXPAND,
    EXPERIMENT_DENSE_SHARE,
    EXPERIMENT_CONV,
    EXPERIMENT_CONV_SHARED,
]
