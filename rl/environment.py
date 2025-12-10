import os, torch

import numpy as np

from tensordict import TensorDict
from torch import Tensor
from torch.utils.cpp_extension import load
from torchrl.envs import EnvBase
from torchrl.data import (
    Binary,
    Categorical,
    Composite,
    UnboundedContinuous
)


# # 1. Lấy đường dẫn thư mục chứa file environment.py hiện tại (<root>/rl/model)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 2. Tạo đường dẫn tuyệt đối tới file cpp
# cpp_source_path = os.path.join(current_dir, "hex_utils.cpp")

# # 3. Load extension
# # Lưu ý: 'build_directory' là tùy chọn, nhưng nên dùng để tránh rác file sinh ra lộn xộn
# hex_utils = load(
#     name="hex_utils",
#     sources=[cpp_source_path],
#     verbose=False,
#     build_directory=os.path.join(current_dir, "build") # Gom file build vào thư mục con
# )


class HexEnv(EnvBase):
    """Hex game environment for reinforcement learning."""
    N_ENV_CHANNELS = 5  # Red (P0), Blue (P1), Current Player, Playable Mask, Swapable Indicator
    """
    Args:
        board_size: Size of the game board (board_size x board_size)
        max_board_size: Maximum board size for padding
        swap_rule: Whether to enable the swap rule (default: True)
        device: Device for tensor computation (default: 'cpu')
    """
    def __init__(
        self, 
        board_size: int,
        max_board_size: int,
        swap_rule: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        # Assertions
        assert board_size >= 1, "Board size must be greater than or equal to 1."
        assert board_size <= max_board_size, "Board size must be less than or equal to max Board size."

        super().__init__(device=device, spec_locked=False)

        # Parameters
        self.board_size: int = board_size
        self.max_board_size: int = max_board_size
        self.n_channel: int = self.N_ENV_CHANNELS
        self.swap_rule: bool = swap_rule

        # Create shape variables
        self.board_shape: torch.Size = torch.Size(
            (self.max_board_size, self.max_board_size)
        ) # (max_board_size, max_board_size)

        # Valid board mask
        valid_board: Tensor = torch.zeros(
            self.board_shape, 
            dtype=torch.bool, 
            device=self.device
        ) # (max_board_size, max_board_size)
        valid_board[:self.board_size, :self.board_size] = 1
        self.register_buffer("valid_board", valid_board)

        # Create private spec variables
        self.observation_spec = Composite({
            "observation": Binary(
                shape=self.board_shape + (self.n_channel,),
                # (max_board_size, max_board_size, n_channel)
                device=self.device,
                dtype=torch.float32
            ),
            "action_mask": Binary(
                shape=(self.max_board_size ** 2,),
                # (max_board_size ** 2,)
                device=self.device,
                dtype=torch.bool
            )
        })
        self.action_spec = Categorical(
            n=self.max_board_size ** 2,
            # Number of discrete actions for each side of the board
            device=self.device,
            dtype=torch.long
        )
        self.reward_spec = UnboundedContinuous(
            shape=(1,),
            device=device,
            dtype=torch.float32
        ) # Reward for both players

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # Initialize a fresh board
        board: Tensor = torch.full((self.max_board_size, self.max_board_size), -1, dtype=torch.long, device=self.device) # -1: empty, 0: player 0 (red), 1: player 1 (blue)
        current_player: int = 0 # 0: player 0 (red), 1: player 1 (blue)
        done: Tensor = torch.tensor(False, dtype=torch.bool, device=self.device) # Game not done

        # Create fresh observation, mask, done, reward
        fresh_action: Tensor = torch.tensor([0], dtype=torch.long, device=self.device) # Placeholder action
        fresh_observation: Tensor = torch.zeros((self.max_board_size, self.max_board_size, self.n_channel), dtype=torch.float32, device=self.device) # (max_board_size, max_board_size, n_channel)
        fresh_observation[..., 0] = (board == 0).float() # Red pieces channel
        fresh_observation[..., 1] = (board == 1).float() # Blue pieces channel
        fresh_observation[..., 2] = current_player # 0: player 0 (red), 1: player 1 (blue)
        fresh_observation[..., 3] = self.valid_board.clone().float() # (max_board_size, max_board_size) Playable board mask
        fresh_observation[..., 4] = self.swap_rule * 1.0 # Swap rule indicator channel
        fresh_action_mask: Tensor = self.valid_board.clone().bool().flatten() # (max_board_size ** 2,) Valid move mask
        fresh_done: Tensor = done # Not done

        fresh_tensordict: TensorDict = TensorDict({
            "action": fresh_action,
            "observation": fresh_observation,
            "action_mask": fresh_action_mask,
            "done": fresh_done
        }, device=self.device)
        return fresh_tensordict

    def _step(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        action: Tensor = tensordict.get("action").to(self.device) # Scalar tensor representing the action
        observation: Tensor = tensordict.get("observation").to(self.device) # (max_board_size, max_board_size, n_channel)

        # Extract action details
        index = int(action.item())
        row, col = divmod(index, self.max_board_size)

        # Extract observation details
        current_player = int(observation[0, 0, 2].item())
        swap_available = bool(observation[0, 0, 4].item())

        board_red, board_blue = observation[..., 0], observation[..., 1]
        total_stones = int((board_red + board_blue).sum().item())

        if not (0 <= row < self.max_board_size and 0 <= col < self.max_board_size):
            raise ValueError(f"Invalid action {index}: out of bounds (row={row}, col={col}).")
        if not self.valid_board[row, col]:
            raise ValueError(f"Invalid action {index}: outside playable area (row={row}, col={col}).")
    
        cell_empty = (board_red[row, col] == 0) and (board_blue[row, col] == 0)
        swap_action = (
            self.swap_rule
            and swap_available
            and current_player == 1
            and total_stones == 1
            and board_red[row, col] == 1
            and board_blue[row, col] == 0
        ) # Player 1 (blue) can swap only on their first move

        if not (cell_empty or swap_action):
            raise ValueError(f"Invalid action {index}: cell occupied and swap not permitted.")

        # Create next observation
        next_observation: Tensor = observation.clone()
        if swap_action:
            next_observation[..., 0], next_observation[..., 1] = next_observation[..., 1].clone(), next_observation[..., 0].clone()
        else:
            next_observation[row, col, current_player] = 1.0

        if self._check_done(next_observation, current_player):
            next_reward = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            next_done = torch.tensor(True, dtype=torch.bool, device=self.device)
        else:
            next_reward = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            next_done = torch.tensor(False, dtype=torch.bool, device=self.device)
            current_player = 1 - current_player

        swap_available_next = bool(self.swap_rule and (total_stones == 0) and not next_done) # Update swap availability for next state
        next_observation[..., 2] = float(current_player)
        next_observation[..., 4] = float(swap_available_next)

        # empty_mask = (next_observation[..., 0] == 0) & (next_observation[..., 1] == 0) & self.valid_board
        # next_action_mask = empty_mask.flatten()
        # if swap_available_next:
        #     next_action_mask[index] = True
        next_action_mask = self.get_action_mask(next_observation, flatten=True)

        return TensorDict({
            "observation": next_observation,
            "action_mask": next_action_mask,
            "done": next_done,
            "reward": next_reward
        }, device=self.device)

    def get_action_mask(self, observation: Tensor, flatten: bool = True) -> Tensor:
        """
        Get the action_mask from the input observation

        :param observation: Input observation tensor
        :type observation: Tensor
        :param flatten: Whether to flatten the action mask
        :type flatten: bool 
        :return: The action mask tensor, where True indicates a valid action
        :rtype: Tensor
        """
        swap_available = bool(observation[0, 0, 4].item()) # Swap rule indicator
        if not swap_available:
            board_red, board_blue = observation[..., 0], observation[..., 1] # (max_board_size, max_board_size)
            empty_mask = (board_red == 0) & (board_blue == 0) & self.valid_board # Valid empty positions
        else:
            empty_mask = self.valid_board.clone() # All valid positions are playable when swap is available
        if flatten:
            return empty_mask.flatten() # (max_board_size ** 2,)
        else:
            return empty_mask # (max_board_size, max_board_size)

    def _check_done(self, observation: Tensor, current_player: int) -> bool:
        def dfs(board, start_positions, target_condition, directions):
            visited = torch.zeros((self.board_size, self.board_size), dtype=torch.bool)
            for start in start_positions:
                if board[start] == 1 and not visited[start]:
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

        directions = [(-1,0), (1,0), (0,-1), (0,1), (1,-1), (-1,1)] # 6 possible directions in a hex grid
        board_state = observation[:self.board_size, :self.board_size, :]
        # Use DFS to check if player 0 (red) has connected top to bottom
        if current_player == 0:
            board = board_state[..., 0] # Shape (board_size, board_size) # Player 0 pieces
            start_positions = [(0, col) for col in range(self.board_size)]
            target_condition = lambda r, c: r == self.board_size - 1
            if dfs(board, start_positions, target_condition, directions):
                return True

        # Use DFS to check if player 1 (blue) has connected left to right
        else:
            board = board_state[..., 1] # Shape (board_size, board_size) # Player 1 pieces
            start_positions = [(row, 0) for row in range(self.board_size)]
            target_condition = lambda r, c: c == self.board_size - 1
            if dfs(board, start_positions, target_condition, directions):
                return True

        return False # No winner yet

    # def _check_done(self, observation: Tensor, current_player: int) -> bool:
    #     # Cắt lấy phần bàn cờ thực tế (loại bỏ padding nếu có)
    #     # observation shape: (max_size, max_size, channels)
        
    #     # 1. Trích xuất board của người chơi hiện tại
    #     if current_player == 0:
    #         # Player 0 (Red): Channel 0
    #         board_tensor = observation[:self.board_size, :self.board_size, 0]
    #     else:
    #         # Player 1 (Blue): Channel 1
    #         board_tensor = observation[:self.board_size, :self.board_size, 1]

    #     # 2. Đảm bảo tensor ở trên CPU và contiguous (bắt buộc cho C++ accessor)
    #     # Nếu thiết bị là CUDA, .cpu() sẽ copy dữ liệu. 
    #     # Nếu đã là CPU, nó gần như cost-free.
    #     board_cpu = board_tensor.detach().cpu().contiguous()

    #     # 3. Gọi hàm C++
    #     # Trả về True/False
    #     return hex_utils.check_win(board_cpu, current_player)

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)