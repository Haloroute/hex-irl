import torch

from torch.utils.data import Dataset, DataLoader


class HexDataset(Dataset):
    """A simple Dataset for Hex game states and actions.

    This dataset holds pairs of (state, action) where:
    - state: A tensor representing the Hex board state.
    - action: An integer representing the action taken in that state.

    Args:
        data: A list of tuples (state, action).
    """

    def __init__(self, data: list[tuple[torch.Tensor, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx]