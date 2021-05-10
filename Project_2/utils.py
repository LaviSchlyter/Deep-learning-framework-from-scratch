import math

import torch

from Project_2.modules import Tensor


def generate_disc_set(nb):
    input_ = torch.empty(nb, 2).uniform_()
    target = ((input_ - 0.5).pow(2).sum(1) < 1 / (2 * math.pi)).float()
    return Tensor(input_), Tensor(target[:, None])


def evaluate(pred, target):
    return 1 - ((pred.value > 0.5) == (target.value > 0.5)).sum() / len(pred.value)
