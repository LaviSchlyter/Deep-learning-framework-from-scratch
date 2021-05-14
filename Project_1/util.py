import sys
from dataclasses import dataclass, fields
from enum import Enum, auto

import torch
from torchvision.transforms import RandomAffine, InterpolationMode

from dlc_practical_prologue import generate_pair_sets


def select_device(debug_on_cpu: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if debug_on_cpu:
        # switch to cpu device if debugging
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is not None and gettrace() is not None:
            device = "cpu"

    print(f"Using device {device}")
    return device


DEVICE = select_device(debug_on_cpu=True)


@dataclass
class Data:
    train_size: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_y_float: torch.Tensor
    train_digit: torch.Tensor

    test_size: int
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_y_float: torch.Tensor
    test_digit: torch.Tensor

    def to(self, device):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))

    def expand_train_flip(self):
        same_digit = self.train_digit[:, 0] == self.train_digit[:, 1]

        self.train_x = torch.cat([self.train_x, self.train_x.flip(1)])
        self.train_y = torch.cat([self.train_y, 1 - self.train_y + same_digit])
        self.train_y_float = torch.cat([self.train_y_float, 1 - self.train_y_float + same_digit])
        self.train_digit = torch.cat([self.train_digit, self.train_digit.flip(1)])

    def expand_train_transform(self, factor: int):
        assert factor >= 1
        transform = RandomAffine(degrees=5, shear=10, interpolation=InterpolationMode.BILINEAR)

        self.train_size *= factor

        train_x_new = [transform(self.train_x) for _ in range(factor - 1)]
        self.train_x = torch.cat([self.train_x] + train_x_new, dim=0)

        self.train_y = self.train_y.repeat(factor)
        self.train_y_float = self.train_y_float.repeat(factor)
        self.train_digit = self.train_digit.repeat(factor, 1)

    def shuffle_train(self):
        perm = torch.randperm(len(self.train_x))
        self.train_x = self.train_x[perm]
        self.train_y = self.train_y[perm]
        self.train_y_float = self.train_y_float[perm]
        self.train_digit = self.train_digit[perm]


class InputNormalization(Enum):
    # don't do any normalization
    No = auto()
    # normalize each element of the input data separately
    ElementWise = auto()
    # normalize all elements of the input data together
    Total = auto()


def normalize(train_x, test_x, input_normalization: InputNormalization):
    if input_normalization == InputNormalization.No:
        std = torch.tensor(1)
        mean = torch.tensor(0)
    elif input_normalization == InputNormalization.ElementWise:
        std, mean = torch.std_mean(train_x, dim=(0, 1))
    elif input_normalization == InputNormalization.Total:
        std, mean = torch.std_mean(train_x)
    else:
        assert False, input_normalization

    std[std == 0] = 1

    return (train_x - mean) / std, (test_x - mean) / std


def load_data(data_size: int, input_normalization: InputNormalization) -> Data:
    train_x, train_y, train_digit, test_x, test_y, test_digit = generate_pair_sets(data_size)

    train_x, test_x = normalize(train_x, test_x, input_normalization)

    data = Data(
        train_x=train_x, train_y=train_y, train_y_float=train_y.float(), train_digit=train_digit, train_size=data_size,
        test_x=test_x, test_y=test_y, test_y_float=test_y.float(), test_digit=test_digit, test_size=data_size,
    )
    return data
