import torch, torchrl

import numpy as np

from tensordict import TensorDict
from torch import Tensor
from torchrl.envs import EnvBase, SerialEnv
from torchrl.data import (
    Binary,
    Categorical,
    Composite,
    UnboundedContinuous
)


# Constants
N_CHANNEL = 4 # Number of channels for the observation (Red, Blue, Current Player, Valid Board)
MAX_BOARD_SIZE = 32 # Maximum board size for the Hex game
SWAP_RULE = True # Whether to use the swap rule in the Hex game

class HexEnv(EnvBase):
    def __init__(self, 
                 board_size: int,
                 max_board_size: int = MAX_BOARD_SIZE,
                 swap_rule: bool = SWAP_RULE,
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
        self.swap_rule: bool = swap_rule
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
            # Number of discrete actions for each side of the board
            shape=(1,),
            # (1,) because action is a scalar representing the flat index of the board
            device=self.device,
            dtype=torch.long,
            mask=(self.valid_board.flatten() == 1) # (max_board_size ** 2)
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
        # valid_move: Tensor = self.valid_board.float() # All valid moves at the start
        done: Tensor = torch.tensor(False, dtype=torch.bool, device=self.device) # Game not done
        # reward: Tensor = torch.tensor([0.0], dtype=torch.float32, device=self.device) # No reward at the start

        # Create fresh observation, mask, done, reward
        fresh_action: Tensor = torch.tensor([0], dtype=torch.long, device=self.device) # Placeholder action
        fresh_observation: Tensor = torch.zeros((self.max_board_size, self.max_board_size, self.n_channel), dtype=torch.float32, device=self.device) # (max_board_size, max_board_size, n_channel)
        fresh_observation[..., 0] = (board == 0).float() # Red pieces channel
        fresh_observation[..., 1] = (board == 1).float() # Blue pieces channel
        fresh_observation[..., -2] = current_player # 0: player 0 (red), 1: player 1 (blue)
        fresh_observation[..., -1] = self.valid_board.clone().float() # (max_board_size, max_board_size) Playable board mask
        fresh_mask: Tensor = self.valid_board.clone().bool() # (max_board_size ** 2) Valid move mask
        fresh_done: Tensor = done # Not done
        # fresh_reward: Tensor = reward # No reward at the start

        # Update action spec for the environment
        self.action_spec.update_mask(fresh_mask.flatten())

        # Create fresh tensordict
        fresh_tensordict: TensorDict = TensorDict({
            "action": fresh_action,
            "observation": fresh_observation,
            "mask": fresh_mask,
            "done": fresh_done,
            # "reward": fresh_reward
        }, device=self.device)

        return fresh_tensordict

    def _step(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        # Extract action
        action: Tensor = tensordict.get("action").clone() # Scalar tensor representing the action
        observation: Tensor = tensordict.get("observation").clone() # (max_board_size, max_board_size, n_channel)
        mask: Tensor = tensordict.get("mask").clone() # (max_board_size, max_board_size)
        done: Tensor = tensordict.get("done").clone() # Scalar tensor representing if the game is done
        reward: Tensor = self.reward_spec.zero() # Initialize reward tensor # (1,)
        # reward: Tensor = tensordict.get("reward").clone() # (2,)

        # Extract indexes of action from observation
        index: int = int(action.item())
        row, col = divmod(index, self.max_board_size) # Convert flat index to 2D coordinates

        # Extract current state from observation
        current_player: int = int(observation[0, 0, -2].item()) # 0: player 0 (red), 1: player 1 (blue)

        # Check if this is a swap situation
        is_first_move = (torch.sum(observation[..., 0:2]).item() == 0 and
                         current_player == 0)  # Player 0's turn and no pieces placed yet
        is_second_move = (torch.sum(observation[..., 0:2]).item() == 1 and
                          current_player == 1)  # Player 1's turn and only one piece placed
        is_swap_action = (self.swap_rule and
                        is_second_move and
                        observation[row, col, 0] == 1) # Player 1 selecting player 0's piece

        # Validate action
        is_valid = (
            0 <= row < self.max_board_size and # Must be within board's max bounds
            0 <= col < self.max_board_size and # Must be within board's max bounds
            self.valid_board[row, col] and # Must be in valid board area
            (mask[row, col] == 1 or is_swap_action)  # Must be empty to place a piece, or a valid swap action
        )

        # If action is not valid (only when action_spec mask is not working properly)
        if not is_valid:
            raise ValueError(f"Invalid action {action.item()} at row={row}; col={col}; valid={self.valid_board[row, col]}, mask={mask[row, col]}.")
            # reward[self.current_player - 1] = -1.0 # Penalty for invalid move
            # self.done = False # Continue the game even if the move is invalid
            # new_observation, new_mask = tensordict.get("observation"), tensordict.get("mask") # Keep previous observation and mask
        else:
            # Update mask to prevent placing another piece here
            mask[row, col] = 0 # Update mask to prevent placing another piece here

            # Place piece or swap
            if is_swap_action: # Swap the pieces
                observation[..., 0], observation[..., 1] = observation[..., 1].clone(), observation[..., 0].clone()
            else: # Place the piece on the board
                observation[row, col, current_player] = 1.0 # Update observation for the current player

            # Check for win condition (placeholder logic)
            if self._check_done(observation, current_player):
                reward: Tensor = torch.tensor([1.0 * (1 - current_player) - 1.0 * current_player], dtype=torch.float32, device=self.device) # Single reward for the current player (+1 if player 0 wins, -1 if player 1 wins)
                done = torch.tensor(True, dtype=torch.bool) # Game done
            else:
                reward: Tensor = torch.tensor([0.0], dtype=torch.float32, device=self.device) # Initialize reward
                done = torch.tensor(False, dtype=torch.bool) # Game not done

                # Switch player
                current_player = 1 - current_player # Switch between 0 and 1

            # Update observation, mask
            new_observation: Tensor = torch.zeros((self.max_board_size, self.max_board_size, self.n_channel), dtype=torch.float, device=self.device) # (max_board_size, max_board_size, n_channel)
            new_observation[..., 0] = observation[..., 0] # Red pieces channel
            new_observation[..., 1] = observation[..., 1] # Blue pieces channel
            new_observation[..., -2] = float(current_player) # Current player channel
            new_observation[..., -1] = observation[..., -1] # (max_board_size, max_board_size) Playable board mask (doesn't change)
            new_mask: Tensor = mask.bool() # Valid move mask

        # Create done, reward tensors
        new_action: Tensor = action
        new_done: Tensor = done
        new_reward: Tensor = reward

        # Update action spec for the environment
        if is_first_move and self.swap_rule:
            # Allow swap action if it's the first move and swap rule is enabled
            swap_mask = new_mask.clone()
            swap_mask[row, col] = 1 # Allow the swap action
            self.action_spec.update_mask(swap_mask.flatten())
        else:
            # Update action spec for the environment
            self.action_spec.update_mask(new_mask.flatten())

        # Update tensordict
        new_tensordict = TensorDict({
            "action": new_action,
            "observation": new_observation,
            "mask": new_mask,
            "done": new_done,
            "reward": new_reward
        }, device=self.device)

        return new_tensordict

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
        # Use DFS to check if player 0 (red) has connected top to bottom
        if current_player == 0:
            board = (observation[..., 0] == 1)[..., :self.board_size, :self.board_size] # Player 0 pieces
            start_positions = [(0, col) for col in range(self.board_size)]
            target_condition = lambda r, c: r == self.board_size - 1
            if dfs(board, start_positions, target_condition, directions):
                return True

        # Use DFS to check if player 1 (blue) has connected left to right
        else:
            board = (observation[..., 1] == 1)[..., :self.board_size, :self.board_size] # Player 1 pieces
            start_positions = [(row, 0) for row in range(self.board_size)]
            target_condition = lambda r, c: c == self.board_size - 1
            if dfs(board, start_positions, target_condition, directions):
                return True

        return False # No winner yet

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class MaskedSerialEnv(SerialEnv):
    """
    Một lớp SerialEnv tùy chỉnh tự động đồng bộ action masks
    từ các môi trường con (child envs) lên môi trường cha (parent env).

    Lớp này ghi đè `step()` và `reset()` để đảm bảo
    `self.action_spec.mask` của SerialEnv luôn
    phản ánh trạng thái mask tổng hợp (stacked) của các môi trường con.
    
    Điều này làm cho các phương thức như `rand_action()` và `rand_step()`
    hoạt động chính xác với các action mask động.
    """

    def __init__(self, *args, **kwargs):
        """
        Khởi tạo SerialEnv gốc.
        Tất cả các đối số được chuyển thẳng đến lớp cha (SerialEnv).
        """
        super().__init__(*args, **kwargs)

    def _sync_action_mask(self):
        """
        Hàm helper (nội bộ) để thu thập mask từ tất cả các env con
        và gán vào action_spec của env cha.
        """
        try:
            # Lấy danh sách các mask từ mỗi môi trường con
            # `self._envs` là danh sách các env con được SerialEnv quản lý
            all_masks = [env.action_spec.mask for env in self._envs]

            # Nếu tất cả các env con đều chưa có mask (ví dụ: chưa reset lần nào)
            if all(m is None for m in all_masks):
                self.action_spec.mask = None
                return

            # Stack tất cả các mask lại với nhau dọc theo chiều batch (dim=0)
            # Điều này tạo ra một tensor mask có shape [num_workers, *mask_shape]
            stacked_masks = torch.stack(all_masks, dim=0)

            # Gán tensor mask đã stack vào spec của env cha
            self.action_spec.mask = stacked_masks

        except AttributeError:
            # Xử lý trường hợp `env.action_spec.mask` không tồn tại
            print(
                "Cảnh báo: Không thể đồng bộ mask. Một hoặc nhiều env con "
                "thiếu thuộc tính 'mask' trong action_spec."
            )
            self.action_spec.mask = None
        except Exception as e:
            # Bắt các lỗi khác (ví dụ: shape, device, dtype không khớp khi stack)
            print(f"Cảnh báo: Không thể stack các action mask: {e}")
            self.action_spec.mask = None

    def reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        """
        Ghi đè phương thức reset().

        Nó gọi hàm reset() gốc của SerialEnv, sau đó
        đồng bộ action mask mới từ các môi trường con.
        
        Sử dụng 'TensorDict | None' (PEP 604) thay vì Optional[TensorDict].
        Bỏ ': Any' khỏi **kwargs.
        """
        # 1. Gọi hàm reset() gốc của SerialEnv
        data = super().reset(tensordict=tensordict, **kwargs)

        # 2. Đồng bộ mask TỪ trạng thái MỚI
        self._sync_action_mask()

        # 3. Trả về dữ liệu state ban đầu
        return data

    def step(self, tensordict: TensorDict) -> TensorDict:
        """
        Ghi đè phương thức step().

        Nó gọi hàm step() gốc của SerialEnv, sau đó
        đồng bộ action mask mới (của state tiếp theo) từ các môi trường con.
        """
        # 1. Gọi hàm step() gốc của SerialEnv
        next_data = super().step(tensordict)

        # 2. Đồng bộ mask TỪ trạng thái "next"
        self._sync_action_mask()

        # 3. Trả về dữ liệu của bước tiếp theo
        return next_data