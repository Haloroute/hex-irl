import torch

import torch.nn as nn

from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data import Composite, TensorSpec
from torch import Tensor

from rl.config import (
    DEVICE, STORAGE_DEVICE,
    BOARD_SIZE, MAX_BOARD_SIZE, SWAP_RULE, N_CHANNEL,
    MODEL_PARAMS, BUFFER_SIZE, MAX_N_STEPS, N_SAMPLES_PER_EPOCH, N_EPISODES_PER_EPOCH, N_TRAINING_ROUNDS_PER_EPOCH, N_MEMMAP_CHUNKS,
    INITIAL_TEMPERATURE, FINAL_TEMPERATURE, DECAY_RATE,
    N_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    TOTAL_FRAMES, WARMUP_FRAMES, OPTIMIZATION_STEPS, GAMMA, TAU, GRAD_CLIP_NORM,
    LOG_INTERVAL, RANDOM_EVAL_INTERVAL, PAST_EVAL_INTERVAL, MCTS_EVAL_INTERVAL, EVAL_GAMES, MCTS_ITERMAX,
    CHECKPOINT_DIR, RESULTS_DIR
)


class ModelWrapper(nn.Module):
    """
    Bọc HexModel với logic Canonicalization.
    Input: (N, H, W, 5) -> Output: Logits (N, H*W)
    """
    def __init__(self, model: nn.Module, board_size: int, temperature: float = 1.0):
        super().__init__()
        self.model = model  # Tham chiếu đến model chung
        self.board_size = board_size
        self.temperature = temperature
        
        # Tạo board_border với kích thước (H+2, W+2, 3)
        # Tất cả các ô đều là 0, ngoại trừ:
        # - Hàng đầu tiên và cuối cùng (kênh 0)
        # - Cột đầu tiên và cuối cùng (kênh 1)
        # - 4 góc luôn bằng 0
        H, W = board_size, board_size
        board_border = torch.zeros(H + 2, W + 2, N_CHANNEL)
        
        # Kênh 0: hàng đầu tiên và cuối cùng (trừ góc)
        board_border[0, 1:-1, 0] = 1.0      # Hàng đầu tiên (trừ góc)
        board_border[-1, 1:-1, 0] = 1.0     # Hàng cuối cùng (trừ góc)
        
        # Kênh 1: cột đầu tiên và cuối cùng (trừ góc)
        board_border[1:-1, 0, 1] = 1.0      # Cột đầu tiên (trừ góc)
        board_border[1:-1, -1, 1] = 1.0     # Cột cuối cùng (trừ góc)
        
        # self.register_buffer('board_border', board_border)
        self.board_border = board_border  # Không dùng register_buffer để tránh lỗi khi save/load model

    def forward(self, observation: Tensor, action_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        device = next(self.model.parameters()).device
        observation = observation.to(device)
        if action_mask is not None:
            action_mask = action_mask.to(device)

        # Xử lý batch dimension nếu thiếu
        had_batch_dim = observation.dim() == 4
        if not had_batch_dim:
            observation = observation.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)

        N, H, W, _ = observation.shape
        
        # 1. Xác định Player 1 (Blue)
        player_mask = (observation[:, 0, 0, 2] > 0.5)  # (N,)

        # 2. Chỉ giữ lại 2 kênh đầu tiênvà kênh thứ 4 (kênh swap rule)
        observation_input = observation[..., [0, 1, 4]].clone()  # (N, H, W, 3)

        # 3. Canonicalize cho Player 1
        if player_mask.any():
            observation_input[player_mask] = observation_input[player_mask].transpose(1, 2)
            r = observation_input[player_mask, ..., 0].clone()
            b = observation_input[player_mask, ..., 1].clone()
            observation_input[player_mask, ..., 0] = b
            observation_input[player_mask, ..., 1] = r

        # 4. Padding với board_border: (N, H, W, 3) -> (N, H+2, W+2, 3)
        padded = self.board_border.unsqueeze(0).expand(N, -1, -1, -1).to(device)  # (N, H+2, W+2, 3)
        padded[:, 1:-1, 1:-1, :] = observation_input  # Chèn observation vào giữa
        
        # # 5. Chuyển sang format (N, C, H, W) cho Conv2d
        # padded = padded.permute(0, 3, 1, 2).contiguous()  # (N, 3, H+2, W+2)

        # 6. Chạy Model
        logits: Tensor = self.model(padded)  # (N, H*W)

        # 7. Reshape và xoay ngược
        logits = logits.reshape(N, H, W)  # Reshape về grid

        if player_mask.any():
            # Player 1: output đang là (H, W) nhưng thực tế là (W, H) của board gốc
            # Transpose lại để về đúng tọa độ board gốc
            logits[player_mask] = logits[player_mask].transpose(1, 2)  # (N, W, H) -> (N, H, W)

        # 8. Flatten
        logits = logits.reshape(N, -1)  # (N, H*W)
        
        # Apply temperature: với xác suất = temperature, cộng 1000 vào 1 logit ngẫu nhiên của 1 sample
        if not torch.rand(1, device=logits.device).item() >= self.temperature:
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

        # 9. Xử lý action_mask
        if action_mask is not None:
            action_mask = action_mask.view(N, -1)  # (N, H*W)

        if not had_batch_dim:
            logits = logits.squeeze(0)
            if action_mask is not None:
                action_mask = action_mask.squeeze(0)

        return logits, action_mask