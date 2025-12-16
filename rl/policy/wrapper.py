import torch.nn as nn

from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data import Composite, TensorSpec
from torch import Tensor

from rl.model.network import HexModel


class ActorWrapper(nn.Module):
    """
    Bọc HexModel (Actor) với logic Canonicalization.
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

        N, H, W, C = observation.shape
        
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


class CriticWrapper(nn.Module):
    """
    Bọc HexModel (Critic) với logic Canonicalization.
    Input: (N, H, W, 5) -> Output: Q-Values (N, H*W)
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model # Tham chiếu đến CÙNG model chung

    def forward(self, observation: Tensor) -> Tensor:
        had_batch_dim = observation.dim() == 4
        if not had_batch_dim:
            observation = observation.unsqueeze(0)

        N, H, W, _ = observation.shape
        player_mask = (observation[..., 0, 0, 2] > 0.5)
        obs_input = observation[..., [0, 1, 3, 4]].clone()

        if player_mask.any():
            obs_input[player_mask] = obs_input[player_mask].transpose(1, 2)
            r = obs_input[player_mask, ..., 0].clone()
            b = obs_input[player_mask, ..., 1].clone()
            obs_input[player_mask, ..., 0] = b
            obs_input[player_mask, ..., 1] = r

        q_values = self.model(obs_input) 
        q_values = q_values.view(N, H, W)

        if player_mask.any():
            q_values[player_mask] = q_values[player_mask].transpose(1, 2)

        q_values = q_values.reshape(N, -1)        
        if not had_batch_dim:
            q_values = q_values.squeeze(0)

        return q_values


class MaskedRandomPolicy:
    """A masked random policy for data collectors.

    This policy selects random actions from the set of valid (masked) actions only.
    It respects the action_mask in the TensorDict to ensure only legal moves are chosen.

    This is useful for:
    - Warmup phase in RL training (collecting initial random experiences)
    - Baseline evaluation (comparing against random play)
    - Opponent behavior in self-play scenarios

    Args:
        action_spec: TensorSpec object describing the action space.
            Must be a Categorical spec that supports action masking.
        action_key: Key name for the action in TensorDict (default: "action")

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Categorical
        >>> import torch
        >>> 
        >>> # Create action spec for a 5x5 board (25 possible actions)
        >>> action_spec = Categorical(n=25, device='cpu')
        >>> policy = MaskedRandomPolicy(action_spec=action_spec)
        >>> 
        >>> # Create a tensordict with action mask (only positions 0, 5, 10 are valid)
        >>> action_mask = torch.zeros(25, dtype=torch.bool)
        >>> action_mask[[0, 5, 10]] = True
        >>> td = TensorDict({"action_mask": action_mask}, batch_size=[])
        >>> 
        >>> # Sample random action from valid positions only
        >>> td = policy(td)
        >>> print(td["action"])  # Will be 0, 5, or 10

    Note:
        - The action_mask must be present in the input TensorDict
        - Invalid (masked) actions will never be selected
        - This ensures compliance with environment constraints (e.g., empty cells in Hex)
    """

    def __init__(self, action_spec: TensorSpec, action_key: NestedKey = "action"):
        super().__init__()
        self.action_spec = action_spec
        self.action_key = action_key

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Select a random valid action based on the action mask.

        Args:
            td: TensorDict containing at minimum:
                - "action_mask": Boolean tensor indicating valid actions

        Returns:
            TensorDict with added "action" key containing the selected action
        """
        action_mask: Tensor = td.get("action_mask")
        self.action_spec.update_mask(action_mask)
        if isinstance(self.action_spec, Composite):
            return td.update(self.action_spec.rand())
        else:
            return td.set(self.action_key, self.action_spec.rand())