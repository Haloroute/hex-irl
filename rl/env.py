import torch, torchrl

import numpy as np

from tensordict import TensorDict
from torch import Tensor
from torchrl.envs import EnvBase
from torchrl.envs.transforms import ActionMask, TransformedEnv
from torchrl.data import (
    Binary,
    Bounded,
    Categorical,
    Composite,
    TensorSpec,
    UnboundedContinuous
)


# Constants
N_CHANNEL = 4 # Number of channels for the observation (Red, Blue, Current Player, Valid Board)
MAX_BOARD_SIZE = 32 # Maximum board size for the Hex game
SWAP_RULE = False # Whether to use the swap rule in the Hex game

# class CategoricalNd(Categorical):
#     """
#     A custom TensorSpec that behaves like Categorical but works in N-D and includes a mask to prevent unwanted choices.
#     The mask is a boolean tensor with the same shape as the output tensor, where True indicates valid values.
#     """
#     def __init__(self, 
#                  n: int, # Number of categories
#                  d_n: int, # Number of categorical dimensions
#                  shape: torch.Size = torch.Size(), # Shape of batch dimensions
#                  mask: Tensor | None = None, # Mask tensor with the same shape as the full output shape (batch + categorical)
#                  device: torch.device = 'cpu', # Device for the tensors
#                  dtype: torch.dtype = torch.long):
#         # Store parameters
#         assert n >= 1, "n must be at least 1 for N-D categorical."
#         assert d_n >= 2, "d_n must be at least 2 for N-D categorical. For 1D categorical, use the standard Categorical spec."
#         # self.n: int = n
#         self.d_n: int = d_n
#         self.batch_shape: torch.Size = shape
#         # self.device: torch.device = device
#         # self.dtype: torch.dtype = dtype
#         self.categorical_shape: torch.Size = torch.Size((d_n,)) # (d_n,)
#         self.mask_shape: torch.Size = torch.Size(d_n * [n]) # (n, n, ..., n) repeated d_n times
#         self.batch_categorical_shape: torch.Size = self.batch_shape + self.categorical_shape # Full shape including batches and categories
#         self.batch_mask_shape: torch.Size = self.batch_shape + self.mask_shape # Full shape including batches and mask

#         # Initialize the mask
#         if mask is not None and not isinstance(mask, (Tensor, type(None))): # Convert to tensor if not already
#             try:
#                 mask = torch.tensor(mask, device=device, dtype=torch.bool)
#             except Exception as e:
#                 raise ValueError(f"Failed to convert mask to tensor: {e}")

#         if mask is not None: # Validate and set the mask
#             if mask.shape == self.batch_mask_shape: # No need to expand batch dimensions
#                 self.mask_nd = mask.to(device=device, dtype=torch.bool).clone() # (batch_shape, d_n, d_n, ..., d_n) with n times d_n
#             elif mask.shape == self.mask_shape: # Only categorical dimensions, expand batch dimensions
#                 self.mask_nd = mask.to(device=device, dtype=torch.bool).expand(self.batch_mask_shape).clone() # (batch_shape, d_n, d_n, ..., d_n) with n times d_n
#             elif mask.shape == self.batch_shape: # Only batch dimensions, expand categorical dimensions
#                 self.mask_nd = mask.to(device=device, dtype=torch.bool).expand(self.batch_mask_shape).clone() # (batch_shape, d_n, d_n, ..., d_n) with n times d_n
#             else:
#                 raise ValueError(
#                     f"Mask shape {mask.shape} is incompatible with full shape {self.batch_mask_shape}, categorical shape {self.categorical_shape}, or batch shape {self.batch_shape}."
#                 )
#         else: # Default for None mask: all True
#             self.mask_nd = torch.ones(self.batch_mask_shape, device=device, dtype=torch.bool) # (batch_shape, d_n, d_n, ..., d_n) with n times d_n

#         # Call the parent constructor with adjusted parameters
#         super().__init__(n, self.batch_categorical_shape, device, dtype) # Don't use mask in parent class

#     def __repr__(self):
#         return f"CategoricalNd(n={self.n}, d_n={self.d_n}, batch shape={self.batch_shape}, categorical shape={self.categorical_shape}, mask shape={self.mask_shape})"

#     def cardinality(self): # Count only valid positions
#         return self.mask.sum().item()

#     def is_in(self, val: Tensor) -> bool: # Check both bounds and mask
#         # First check if val is in the bounded space using parent method
#         in_bounds: bool = super().is_in(val)
#         if not in_bounds:
#             return False

#         # Check the shape first
#         if val.shape != self.batch_categorical_shape:
#             return False

#         # After the bounds check, val tensor has the same shape as self.batch_categorical_shape (i.e., batch_shape + (d_n,)). Now check the mask. 
#         # For each N-D categorical (the last dimensions, contains (d_n)), we need to check if the corresponding position in the mask is True.
#         # Which means we need to convert the N-D categorical indices to a flat index in the mask.
#         # Example: if n=3, d_n=2, shape=(), and val[..., 0]=2, val[..., 1]=1, then the corresponding position in the mask is (2, 1) in a 3x3 grid.
#         # We can achieve this by using advanced indexing.
#         # Create a grid of indices for the categorical dimensions
#         grid = torch.meshgrid(*[torch.arange(self.n, device=val.device) for _ in range(self.d_n)], indexing='ij')
#         flat_indices = sum((grid[i] == val[..., i].unsqueeze(-1)).long() * (self.n ** i) for i in range(self.d_n))
#         # Now flat_indices has the same shape as val, and each position corresponds to a position in the mask
#         # We can use this to index into the mask
#         return (self.mask_nd[flat_indices]).all().item() != 0 # If all position is True in the mask, return True. Else return False.

#     # def rand(self, shape=None) -> Tensor: # Override to respect the mask
#     #     # Sample from the bounded space, but set masked positions to some default (e.g., low)
#     #     sample = super().rand(shape)
#     #     if shape is None:
#     #         shape = self.shape
#     #     mask_expanded = self.mask.expand(shape + self.mask.shape[len(shape):]) if len(shape) > len(self.mask.shape) else self.mask
#     #     sample = torch.where(mask_expanded, sample, self.low)
#     #     return sample

#     def update_mask(self, new_mask: Tensor | None = None):
#         # Initialize the mask
#         if new_mask is not None and not isinstance(new_mask, (Tensor, type(None))): # Convert to tensor if not already
#             try:
#                 new_mask = torch.tensor(new_mask, device=self.device, dtype=torch.bool)
#             except Exception as e:
#                 raise ValueError(f"Failed to convert mask to tensor: {e}")

#         if new_mask is not None: # Validate and set the mask
#             if new_mask.shape == self.batch_mask_shape: # No need to expand batch dimensions
#                 self.mask_nd = new_mask.to(device=self.device, dtype=torch.bool).clone() # (batch_shape, d_n, d_n, ..., d_n) with n times d_n
#             elif new_mask.shape == self.mask_shape: # Only categorical dimensions, expand batch dimensions
#                 self.mask_nd = new_mask.to(device=self.device, dtype=torch.bool).expand(self.batch_mask_shape).clone() # (batch_shape, d_n, d_n, ..., d_n) with n times d_n
#             elif new_mask.shape == self.batch_shape: # Only batch dimensions, expand categorical dimensions
#                 self.mask_nd = new_mask.to(device=self.device, dtype=torch.bool).expand(self.batch_mask_shape).clone() # (batch_shape, d_n, d_n, ..., d_n) with n times d_n
#             else:
#                 raise ValueError(
#                     f"Mask shape {new_mask.shape} is incompatible with full shape {self.batch_mask_shape}, categorical shape {self.categorical_shape}, or batch shape {self.batch_shape}."
#                 )
#         else: # Default for None mask: all True
#             self.mask_nd = torch.ones(self.batch_mask_shape, device=self.device, dtype=torch.bool) # (batch_shape, d_n, d_n, ..., d_n) with n times d_n


class HexEnv(EnvBase):
    def __init__(self, 
                 board_size: int,
                 max_board_size: int = MAX_BOARD_SIZE,
                #  swap_rule: bool = SWAP_RULE,
                 device: torch.device = 'cpu',
                #  batch_size: torch.Size = torch.Size()
                ):

        # Assertions
        assert board_size >= 1, "Board size must be greater than or equal to 1."
        assert board_size <= max_board_size, "Board size must be less than or equal to max Board size."

        super().__init__(device=device, spec_locked=False)

        # Parameters
        self.board_size: int = board_size
        self.max_board_size: int = max_board_size
        self.n_channel: int = N_CHANNEL
        # self.swap_rule: bool = swap_rule # Not implemented yet
        # self.device: torch.device = device
        # self.batch_size: torch.Size = batch_size # No batching at all

        # Create shape variables
        self.board_shape: torch.Size = torch.Size(
            (self.max_board_size, self.max_board_size)
        ) # (max_board_size, max_board_size)

        # Valid board mask
        self.valid_board: Tensor = torch.zeros(
            self.board_shape, 
            dtype=torch.bool, 
            device=self.device
        ) # (max_board_size, max_board_size)
        self.valid_board[:self.board_size, :self.board_size] = 1

        # Create private spec variables
        self.observation_spec = Composite({
            "observation": Binary(
                shape=self.board_shape + (self.n_channel,),
                # (max_board_size, max_board_size, n_channel)
                device=self.device,
                dtype=torch.float32
            ),
            "mask": Binary(
                shape=self.board_shape,
                # (max_board_size, max_board_size)
                device=self.device,
                dtype=torch.bool
            )
        })
        self.action_spec = Categorical(
            n=self.max_board_size ** 2, 
            # Number of discrete actions for each side of the board. Scalar
            device=self.device,
            dtype=torch.long,
            mask=(self.valid_board.flatten() == 1) # (max_board_size ** 2)
        )
        self.reward_spec = UnboundedContinuous(
            shape=(2,),
            device=device,
            dtype=torch.float32
        ) # Reward for both players

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # Initialize a fresh board
        board: Tensor = torch.full((self.max_board_size, self.max_board_size), -1, dtype=torch.long) # -1: empty, 0: player 0 (red), 1: player 1 (blue)
        current_player: int = 0 # 0: player 0 (red), 1: player 1 (blue)
        # valid_move: Tensor = self.valid_board.float() # All valid moves at the start
        done: Tensor = torch.tensor(False, dtype=torch.bool) # Game not done
        reward: Tensor = torch.tensor([0.0, 0.0], dtype=torch.float32) # No reward at the start for both players

        # Create fresh observation, mask, done, reward
        fresh_action: Tensor = torch.tensor(0, dtype=torch.long) # Placeholder action
        fresh_observation: Tensor = torch.zeros((self.max_board_size, self.max_board_size, self.n_channel), dtype=torch.float32) # (max_board_size, max_board_size, n_channel)
        fresh_observation[..., 0] = (board == 0).float() # Red pieces channel
        fresh_observation[..., 1] = (board == 1).float() # Blue pieces channel
        fresh_observation[..., 2] = current_player # Current player channel
        fresh_observation[..., -1] = self.valid_board.clone().float() # (max_board_size, max_board_size) Playable board mask
        fresh_mask: Tensor = self.valid_board.clone().bool() # (max_board_size ** 2) Valid move mask
        fresh_done: Tensor = done # Not done
        fresh_reward: Tensor = reward # No reward at the start

        # Update action spec for the environment
        self.action_spec.update_mask(fresh_mask.flatten())

        # Update tensordict
        fresh_tensordict = TensorDict({
            "action": fresh_action,
            "observation": fresh_observation,
            "mask": fresh_mask,
            "done": fresh_done,
            "reward": fresh_reward
        })

        return fresh_tensordict

    def _step(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        # Extract action
        action: Tensor = tensordict.get("action") # Scalar tensor representing the action
        observation: Tensor = tensordict.get("observation") # (max_board_size, max_board_size, n_channel)
        mask: Tensor = tensordict.get("mask") # (max_board_size, max_board_size)
        done: Tensor = tensordict.get("done") # Scalar tensor representing if the game is done
        reward: Tensor = tensordict.get("reward") # (2,)

        # Extract indexes of action from observation
        index: int = int(action.item())
        row, col = divmod(index, self.max_board_size) # Convert flat index to 2D coordinates

        # Extract current state from observation
        current_player: int = int(observation[..., 0, 0, 2].item()) # 0: player 0 (red), 1: player 1 (blue)
        reward: Tensor = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device) # Initialize reward for both players

        # Validate action
        is_valid = (
            0 <= row < self.max_board_size and # Must be within board's max bounds
            0 <= col < self.max_board_size and # Must be within board's max bounds
            self.valid_board[row, col] and # Must be in valid board area
            mask[row, col] == 1  # Must be empty to place a piece
        )

        # If action is not valid (only when action_spec mask is not working properly)
        if not is_valid:
            raise ValueError(f"Invalid action {action.item()} at row={row}; col={col}; valid={self.valid_board[row, col]}, mask={mask[row, col]}.")
            # reward[self.current_player - 1] = -1.0 # Penalty for invalid move
            # self.done = False # Continue the game even if the move is invalid
            # new_observation, new_mask = tensordict.get("observation"), tensordict.get("mask") # Keep previous observation and mask
        else:
            # Place the piece
            observation[row, col, current_player] = 1.0 # Update observation for the current player
            mask[row, col] = 0 # Update mask to prevent placing another piece here

            # Check for win condition (placeholder logic)
            if self._check_done(observation, current_player):
                reward[current_player] = 1.0 # Reward for winning
                reward[1 - current_player] = -1.0 # Penalty for losing
                done = torch.tensor(True, dtype=torch.bool) # Game done
            else:
                done = torch.tensor(False, dtype=torch.bool) # Game not done

                # Switch player
                current_player = 1 - current_player # Switch between 0 and 1

            # Update observation, mask
            new_observation: Tensor = torch.zeros((self.max_board_size, self.max_board_size, self.n_channel), dtype=torch.float) # (max_board_size, max_board_size, n_channel)
            new_observation[..., 0] = observation[..., 0] # Red pieces channel
            new_observation[..., 1] = observation[..., 1] # Blue pieces channel
            new_observation[..., 2] = float(current_player) # Current player channel
            new_observation[..., -1] = observation[..., -1] # (max_board_size, max_board_size) Playable board mask (doesn't change)
            new_mask: Tensor = mask.bool().clone() # Valid move mask

        # Create done, reward tensors
        new_action: Tensor = action
        new_done: Tensor = done
        new_reward: Tensor = reward

        # Update action spec for the environment
        self.action_spec.update_mask(new_mask.flatten())

        # Update tensordict
        new_tensordict = TensorDict({
            "action": new_action,
            "observation": new_observation,
            "mask": new_mask,
            "done": new_done,
            "reward": new_reward
        })

        return new_tensordict

    def _check_done(self, observation: Tensor, current_player: int) -> bool:
        board_1 = (observation[..., 0] == 1).float() # Player 0 pieces
        board_2 = (observation[..., 1] == 1).float() # Player 1 pieces

        def dfs(board, start_positions, target_condition, directions):
            visited = torch.zeros((self.board_size, self.board_size), dtype=torch.bool)
            for start in start_positions:
                if board[start] != 0 and not visited[start]:
                    stack = [start]
                    visited[start] = True
                    while stack:
                        r, c = stack.pop()
                        if target_condition(r, c):
                            return True
                        for dr, dc in directions:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
            return False

        # Use DFS to check if player 1 (red) has connected top to bottom
        if current_player == 1:
            directions = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
            start_positions = [(0, col) for col in range(self.board_size)]
            target_condition = lambda r, c: r == self.board_size - 1
            if dfs(board_1, start_positions, target_condition, directions):
                return True

        # Use DFS to check if player 2 (blue) has connected left to right
        else:
            directions = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
            start_positions = [(row, 0) for row in range(self.board_size)]
            target_condition = lambda r, c: c == self.board_size - 1
            if dfs(board_2, start_positions, target_condition, directions):
                return True

        return False # No winner yet

    def _set_seed(self, seed: int) -> None:
        super()._set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)