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
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model # Tham chiếu đến model chung

    def forward(self, observation: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        # Xử lý batch dimension nếu thiếu
        had_batch_dim = observation.dim() == 4
        if not had_batch_dim:
            observation = observation.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        N, H, W, _ = observation.shape
        
        # 1. Xác định Player 1 (Blue)
        player_mask = (observation[..., 0, 0, 2] > 0.5) 

        # 2. Chuẩn bị Input (Bỏ kênh 2 - Current Player)
        obs_input = observation[..., [0, 1, 3, 4]].clone()

        # 3. Canonicalize cho Player 1
        if player_mask.any():
            obs_input[player_mask] = obs_input[player_mask].transpose(1, 2)
            r = obs_input[player_mask, ..., 0].clone()
            b = obs_input[player_mask, ..., 1].clone()
            obs_input[player_mask, ..., 0] = b
            obs_input[player_mask, ..., 1] = r

        # 4. Chạy Model
        logits = self.model(obs_input)

        # 5. Reshape và xoay ngược
        # Sau transpose, Player 1 có shape (W, H), Player 0 vẫn là (H, W)
        # Model output flatten theo row-major, nên cần reshape đúng
        logits = logits.view(N, H, W)  # Reshape về grid

        if player_mask.any():
            # Player 1: output đang là (H, W) nhưng thực tế là (W, H) của board gốc
            # Transpose lại để về đúng tọa độ board gốc
            logits[player_mask] = logits[player_mask].transpose(1, 2)

        # 6. Flatten
        logits = logits.reshape(N, -1)
        action_mask = action_mask.view(N, -1)

        if not had_batch_dim:
            logits = logits.squeeze(0)
            action_mask = action_mask.squeeze(0)

        return logits, action_mask