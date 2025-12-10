import torch.nn as nn

from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data import Composite, TensorSpec
from torch import Tensor

from rl.model.network import HexModel


class ActorWrapper(nn.Module):
    """Bọc TransformerQL_AC, chỉ trả về 'logits'."""
    def __init__(self, model: HexModel):
        super().__init__()
        self.model = model # Tham chiếu đến model chung

    def forward(self, observation: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        # Đảm bảo action_mask có shape (N, H*W)
        if observation.dim() > 3:
            action_mask = action_mask.view(observation.shape[0], -1) # (N, H*W)
        elif observation.dim() == 3:
            action_mask = action_mask.view(-1) # (H*W)
        else:
            raise ValueError("Observation tensor must have at least 3 dimensions (H, W, C) and at most 4 dimensions (N, H, W, C).")

        # Chạy model chung, chỉ lấy đầu ra đầu tiên
        logits = self.model(observation) # logits shape (N, H*W) (hoặc (H*W) nếu không có batch)

        # logits[~action_mask] = -torch.inf # Áp dụng mask
        return logits, action_mask


class CriticWrapper(nn.Module):
    """Bọc HexModel, chỉ trả về 'action_value'."""
    def __init__(self, model: HexModel):
        super().__init__()
        self.model = model # Tham chiếu đến CÙNG model chung

    def forward(self, observation: Tensor) -> Tensor:
        # action_mask = action_mask.view(observation.shape[0], -1)

        # Chạy model chung, chỉ lấy đầu ra thứ hai
        q_values = self.model(observation) # q_values shape (N, H*W)

        # q_values[~action_mask] = -torch.inf # Áp dụng mask
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