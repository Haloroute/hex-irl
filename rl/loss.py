import torch

import torch.nn as nn

from tensordict import TensorDictBase
from torch import Tensor


class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("SimpleLoss is a placeholder and needs to be implemented.")

    def forward(self, tensordict: TensorDictBase) -> Tensor:
        raise NotImplementedError("SimpleLoss is a placeholder and needs to be implemented.")