from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.data import Composite, TensorSpec
from torch import Tensor


class MaskedRandomPolicy(TensorDictModuleBase):
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

    def __init__(self, action_spec: TensorSpec, action_mask_key: NestedKey = "action_mask", action_key: NestedKey = "action"):
        super().__init__()
        self.in_keys, self.out_keys = [action_mask_key], [action_key]
        self.action_spec = action_spec
        self.action_mask_key = action_mask_key
        self.action_key = action_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Select a random valid action based on the action mask.
        
        Args:
            tensordict: TensorDict containing at minimum:
                - "action_mask": Boolean tensor indicating valid actions
        Returns:
            TensorDict with added "action" key containing the selected action
        """
        action_mask: Tensor = tensordict.get(self.action_mask_key)
        self.action_spec.update_mask(action_mask)
        if isinstance(self.action_spec, Composite):
            return tensordict.update(self.action_spec.rand())
        else:
            return tensordict.set(self.action_key, self.action_spec.rand())