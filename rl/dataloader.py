import os, shutil, torch

import torch.nn as nn

from pathlib import Path
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm.auto import tqdm


class HexDataCollector:
    """
    Data Collector for Hex game rollouts with backward reward propagation.
    
    This collector:
    1. Rolls out episodes using the provided policy
    2. Propagates rewards backward with: reward[k] = (-1)^k * gamma^(2 * floor(k/2))
       where k is counted from the terminal state (k=0 at the state before termination)
    3. Keeps only observation, action_mask, and reward
    4. Saves to disk as memmap for efficient loading
    
    Args:
        env: HexEnv environment instance
        policy: Policy module (e.g., ProbabilisticActor) that takes TensorDict and returns action
        gamma: Discount factor for reward propagation
        device: Device for computation
        storage_device: Device for storing collected data
    """
    
    def __init__(
        self,
        env: EnvBase,
        policy: TensorDictModule,
        gamma: float = 0.99,
        device: torch.device = torch.device('cpu'),
        storage_device: torch.device = torch.device('cpu')
    ):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.device = device
        self.device, self.storage_device = device, storage_device
        
    def _compute_backward_rewards(
        self, 
        trajectory_length: int,
        terminal_reward: float = 1.0
    ) -> Tensor:
        """
        Compute backward-propagated rewards for a trajectory.
        
        Formula: reward[k] = (-1)^k * gamma^(2 * floor(k/2))
        where k is the step index counting backward from terminal state.
        
        Args:
            trajectory_length: Number of steps in the trajectory
            terminal_reward: The reward at terminal state (typically 1.0 for win)
            
        Returns:
            Tensor of shape (trajectory_length,) with computed rewards
        """
        rewards = torch.zeros(trajectory_length, dtype=torch.float32, device=self.device)  # (T,)
        
        for t in range(trajectory_length):
            # k is the index counting backward from terminal (k=0 at last step)
            k = trajectory_length - 1 - t
            sign = (-1) ** k
            discount = self.gamma ** (2 * (k // 2))
            rewards[t] = 0.5 + 0.5 * sign * discount * terminal_reward
            
        return rewards  # (T,)
    
    def _process_rollout(self, rollout_tensordict: TensorDict) -> TensorDict | None:
        """
        Process a rollout TensorDict to extract and compute rewards.
        
        Args:
            rollout_tensordict: TensorDict from env.rollout() with shape [T]
            
        Returns:
            Processed TensorDict with observation, action_mask, reward
            or None if episode didn't terminate properly.
        """
        # Check if episode terminated (not just truncated)
        terminated: Tensor = rollout_tensordict.get(('next', 'terminated'))  # [T, 1]
        
        # Find the first termination point
        terminated_indices = terminated.squeeze(-1).nonzero(as_tuple=True)[0] # Length equals number of terminations
        if len(terminated_indices) == 0:
            # Episode didn't terminate, skip it
            return None

        # Process all segments between terminations
        all_segments: list[TensorDict] = []
        prev_idx = 0
        
        for terminated_idx in terminated_indices:
            terminated_idx = terminated_idx.item() + 1  # +1 to include the terminal step
            
            # Extract relevant tensors for this segment
            observation_tensor: Tensor = rollout_tensordict.get('observation')[prev_idx:terminated_idx].to(self.storage_device)
            action_tensor: Tensor = rollout_tensordict.get('action')[prev_idx:terminated_idx].to(self.storage_device)
            action_mask_tensor = rollout_tensordict.get('action_mask')[prev_idx:terminated_idx].to(self.storage_device)
            
            segment_length = terminated_idx - prev_idx
            
            # Compute backward rewards for this segment
            rewards = self._compute_backward_rewards(segment_length)
            rewards = rewards.unsqueeze(-1).to(self.storage_device)  # [T, 1]
            
            segment_tensordict = TensorDict({
                'observation': observation_tensor,
                'action': action_tensor,
                'action_mask': action_mask_tensor,
                'reward': rewards
            }, batch_size=[segment_length], device=self.storage_device)
            
            all_segments.append(segment_tensordict)
            prev_idx = terminated_idx
        
        # Concatenate all segments
        return torch.cat(all_segments, dim=0)
    
    def rollout_single(self, max_steps: int = 1024) -> TensorDict | None:
        """
        Rollout a single episode until termination using env.rollout().
        
        Args:
            max_steps: Maximum number of steps before truncation
            
        Returns:
            TensorDict with shape [T] containing:
            - observation: Tensor of shape [T, H, W, C]
            - action: Tensor of shape [T, 1]
            - action_mask: Tensor of shape [T, H, W]
            - reward: Tensor of shape [T, 1]
            Returns None if episode is empty or truncated without termination.
        """
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            # Use TorchRL's built-in rollout
            rollout_tensordict = self.env.rollout(
                max_steps=max_steps,
                policy=self.policy,
                auto_reset=True,
                break_when_any_done=True
            ).to(self.device)
        
        return self._process_rollout(rollout_tensordict)
    
    def collect(
        self,
        n_episodes: int,
        max_steps_per_episode: int = 1024
    ) -> TensorDict:
        """
        Collect multiple episodes of data.
        
        Args:
            n_episodes: Number of episodes to collect
            max_steps_per_episode: Maximum steps per episode
            show_progress: Whether to show progress bar
            
        Returns:
            TensorDict with shape [N] where N is total transitions:
            - observation: Tensor of shape [N, H, W, C]
            - action_mask: Tensor of shape [N, H*W]
            - reward: Tensor of shape [N, 1]
        """
        all_data: list[TensorDict] = []
        for _ in tqdm(range(n_episodes), desc="Collecting episodes", leave=False):
            episode_data = self.rollout_single(max_steps_per_episode)
            if episode_data is not None:
                all_data.append(episode_data)
        
        if len(all_data) == 0:
            raise RuntimeError("No complete episodes collected!")
        
        # Concatenate all episodes along batch dimension
        combined: TensorDict = torch.cat(all_data, dim=0)
        return combined
    
    def collect_and_save(
        self,
        n_episodes: int,
        save_dir: str | Path,
        filename: str = "rollout_data",
        max_steps_per_episode: int = 1024
    ) -> Path:
        """
        Collect episodes and save to disk as memmap.
        
        Args:
            n_episodes: Number of episodes to collect
            save_dir: Directory to save the data
            filename: Base filename for the saved data
            max_steps_per_episode: Maximum steps per episode
            show_progress: Whether to show progress bar
            
        Returns:
            Path to the saved memmap directory
        """
        # Collect data
        data: TensorDict = self.collect(
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps_per_episode
        )
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as memmap
        save_path = save_dir / filename
        data.memmap_(save_path)
        
        # print(f"Saved {len(data)} transitions to {save_path}")
        # print(f"  - Observations: {data['observation'].shape}")
        # print(f"  - Action masks: {data['action_mask'].shape}")
        # print(f"  - Rewards: {data['reward'].shape}")
        
        return save_path


class HexRolloutDataset(Dataset):
    """
    PyTorch Dataset for loading Hex rollout data from memmap.
    
    Args:
        data_path: Path to the memmap directory
        device: Device to load tensors to
    """
    def __init__(
        self,
        data_path: str | Path,
        train: bool,
        n_memmap_chunks: int = 5,
        device: torch.device = torch.device('cpu')
    ):
        self.data_path = Path(data_path)
        self.train = train
        self.n_memmap_chunks = n_memmap_chunks
        self.device = device
        
        # Load n newest memmap chunks if available
        all_entries = os.listdir(self.data_path)
        folders = [entry for entry in all_entries if os.path.isdir(os.path.join(self.data_path, entry))]
        if len(folders) > n_memmap_chunks:
            folders.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_path, x)), reverse=True)
            for folder in folders[n_memmap_chunks:]:
                full_path = os.path.join(self.data_path, folder)
                print(f"Removing old memmap chunk: {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)

        # Load memmap chunks
        data_list: list[TensorDict] = []
        for folder in folders[:n_memmap_chunks]:
            full_path = os.path.join(self.data_path, folder)
            data_list.append(TensorDict.load_memmap(full_path))
        self.data: TensorDict = torch.cat(data_list, dim=0)

        # For training, use only 80% of data
        if self.train:
            self.data = self.data[-int(len(self.data) * 0.8):]
        else:
            self.data = self.data[:int(len(self.data) * 0.2)]
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> TensorDict:
        item = self.data[idx]
        return item.to(self.device)


# def create_dataloader(
#     data_path: str | Path,
#     batch_size: int,
#     shuffle: bool = True,
#     num_workers: int = 0,
#     device: torch.device = torch.device('cpu')
# ) -> DataLoader:
#     """
#     Create a DataLoader for Hex rollout data.
    
#     Args:
#         data_path: Path to the memmap directory
#         batch_size: Batch size for training
#         shuffle: Whether to shuffle data
#         num_workers: Number of worker processes
#         device: Device to load tensors to
        
#     Returns:
#         DataLoader instance
#     """
#     dataset = HexRolloutDataset(data_path, device)
    
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers
#     )