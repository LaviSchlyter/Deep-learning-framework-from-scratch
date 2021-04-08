import sys
from dataclasses import dataclass, fields

import torch
from torchvision.transforms import RandomAffine

from dlc_practical_prologue import generate_pair_sets


def select_device(debug_on_cpu: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if debug_on_cpu:
        # switch to cpu device if debugging
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is not None and gettrace() is not None:
            device = "cpu"

    return device


@dataclass
class Data:
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_y_float: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_y_float: torch.Tensor

    def to(self, device):
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name).to(device))

    def expand_train_data(self, factor: int):
        assert factor >= 1
        transform = RandomAffine(degrees=10)

        train_x_new = [transform(self.train_x) for _ in range(factor - 1)]
        self.train_x = torch.cat([self.train_x] + train_x_new, dim=0)

        self.train_y = self.train_y.repeat(factor)
        self.train_y_float = self.train_y_float.repeat(factor)

        # shuffle
        perm = torch.randperm(len(self.train_x))
        self.train_x = self.train_x[perm]
        self.train_y = self.train_y[perm]
        self.train_y_float = self.train_y_float[perm]


def load_data(data_size) -> Data:
    train_x, train_y, _, test_x, test_y, _ = generate_pair_sets(data_size)
    data = Data(
        train_x=train_x, train_y=train_y, train_y_float=train_y.float(),
        test_x=test_x, test_y=test_y, test_y_float=test_y.float(),
    )
    return data
