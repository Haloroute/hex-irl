import torch

import torch.nn as nn

from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data import Composite, TensorSpec
from torch import Tensor


class ModelWrapper(nn.Module):
    """
    Bọc HexModel với logic Canonicalization.
    Input: (N, H, W, 5) -> Output: Logits (N, H*W)
    """
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model # Tham chiếu đến model chung
        self.temperature = temperature

    def forward(self, observation: Tensor, action_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        observation: Tensor = observation.to(self.model.device)
        if action_mask is not None:
            action_mask = action_mask.to(self.model.device)

        # Xử lý batch dimension nếu thiếu
        had_batch_dim = observation.dim() == 4
        if not had_batch_dim:
            observation = observation.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)

        N, H, W, _ = observation.shape
        
        # 1. Xác định Player 1 (Blue)
        player_mask = (observation[..., 0, 0, 2] > 0.5) # (N,)

        # 2. Chuẩn bị Input (Bỏ kênh 2 - Current Player)
        observation_input = observation[..., [0, 1, 3, 4]].clone() # (N, H, W, 4)

        # 3. Canonicalize cho Player 1
        if player_mask.any():
            observation_input[player_mask] = observation_input[player_mask].transpose(1, 2)
            r = observation_input[player_mask, ..., 0].clone()
            b = observation_input[player_mask, ..., 1].clone()
            observation_input[player_mask, ..., 0] = b
            observation_input[player_mask, ..., 1] = r

        # 4. Chạy Model
        logits: Tensor = self.model(observation_input) # (N, H*W)

        # 5. Reshape và xoay ngược
        # Sau transpose, Player 1 có shape (W, H), Player 0 vẫn là (H, W)
        # Model output flatten theo row-major, nên cần reshape đúng
        logits = logits.reshape(N, H, W)  # Reshape về grid

        if player_mask.any():
            # Player 1: output đang là (H, W) nhưng thực tế là (W, H) của board gốc
            # Transpose lại để về đúng tọa độ board gốc
            logits[player_mask] = logits[player_mask].transpose(1, 2) # (N, W, H) -> (N, H, W)

        # 6. Flatten
        if self.temperature > 1e-6:
            logits = logits.reshape(N, -1) / (self.temperature + 1e-8) # (N, H*W)
        else:
            logits = logits.reshape(N, -1)  # (N, H*W)
            max_indices = logits.argmax(dim=-1, keepdim=True)  # (N, 1)
            max_mask = torch.zeros_like(logits, dtype=torch.bool)
            max_mask.scatter_(-1, max_indices, True)
            logits = torch.where(max_mask, logits, torch.zeros_like(logits))

        # 7. Xử lý action_mask
        if action_mask is not None:
            action_mask = action_mask.view(N, -1) # (N, H*W)

        if not had_batch_dim:
            logits = logits.squeeze(0)
            if action_mask is not None:
                action_mask = action_mask.squeeze(0)

        return logits, action_mask