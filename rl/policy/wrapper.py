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
        logits = logits.reshape(N, -1)  # (N, H*W)
        
        # Apply temperature: với xác suất = temperature, cộng 1000 vào 1 logit ngẫu nhiên của 1 sample
        if self.temperature > 1e-6 and torch.rand(1, device=logits.device).item() < self.temperature:
            num_actions = logits.shape[1]
            
            # Chọn random 1 sample trong batch
            sample_idx = torch.randint(0, N, (1,), device=logits.device).item()
            
            # Chọn random action cho sample đó
            if action_mask is not None:
                valid_mask = action_mask.view(N, -1)[sample_idx]  # (H*W,)
                random_probs = torch.rand(num_actions, device=logits.device) * valid_mask.float()
                action_idx = random_probs.argmax().item()
            else:
                action_idx = torch.randint(0, num_actions, (1,), device=logits.device).item()
            
            # Cộng 1000 vào logit được chọn
            logits[sample_idx, action_idx] += 1000.0

        # 7. Xử lý action_mask
        if action_mask is not None:
            action_mask = action_mask.view(N, -1) # (N, H*W)

        if not had_batch_dim:
            logits = logits.squeeze(0)
            if action_mask is not None:
                action_mask = action_mask.squeeze(0)

        return logits, action_mask