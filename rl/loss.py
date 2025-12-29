import torch

import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDictBase
from torch import Tensor
from typing import Literal


class SimpleLoss(nn.Module):
    def __init__(self, ratio: float = 0.8, reduction: Literal['none', 'mean', 'sum'] = 'mean'):
        super().__init__()
        self.ratio = 0
        self.reduction = reduction

    def forward(self, logits: Tensor, action: Tensor, reward: Tensor) -> Tensor:
        """
        Tính toán loss đơn giản: Cross-Entropy giữa logits và action, nhân với reward.

        Args:
            logits (Tensor): Logits từ policy network, shape (N, num_actions)
            action (Tensor): Hành động đã chọn, shape (N,)
            reward (Tensor): Phần thưởng nhận được, shape (N, 1)

        Returns:
            Tensor: Giá trị loss trung bình trên batch.
        """
        # probabilities: Tensor = F.softmax(logits, dim=-1 ) # (N, num_actions)
        selected_logits: Tensor = torch.gather(logits, 1, action.unsqueeze(1).long()).sigmoid() # (N, 1)
        loss: Tensor = self.ratio * F.l1_loss(selected_logits, reward, reduction=self.reduction) + \
                (1 - self.ratio) * F.binary_cross_entropy(selected_logits, reward, reduction=self.reduction)
        return loss