import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class SkipLayerBias(nn.Module):
    def __init__(self, n_channels: int, padding: int = 1, scale: float = 1):
        super(SkipLayerBias, self).__init__()
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=padding*2+1, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(n_channels)
        self.scale = scale

    def forward(self, x):
        return self.activation(x + self.scale * self.norm(self.conv(x)))