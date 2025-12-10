import math, torch

import torch.nn as nn

from torch import Tensor
from torchrl.envs import EnvBase

from rl.config import DEVICE, STORAGE_DEVICE
from rl.environment import HexEnv


def init_params(model: nn.Module):
    """Initialize model parameters optimally for RL with GELU activation.
    
    Uses orthogonal initialization with appropriate gain values:
    - Hidden layers: gain = sqrt(2) for GELU/ReLU
    - Final projection: gain = 0.01 for max entropy initialization
    - Normalization layers: weight=1, bias=0
    
    Args:
        model: Neural network module to initialize
    """
    for m in model.modules():
        # Linear and Conv2d layers
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # Small init for final projection (max entropy policy)
            if hasattr(model, 'projection') and m is model.projection:
                nn.init.orthogonal_(m.weight, gain=0.01)
            # Standard init for hidden layers
            else:
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
        # Normalization layers
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def get_optimizer_params(model: nn.Module, weight_decay: float = 1e-5) -> list[dict]:
    """Create parameter groups for AdamW optimizer.
    
    Separates parameters into groups with/without weight decay:
    - Conv/Linear weights: WITH decay
    - All biases: NO decay
    - Normalization weights: NO decay
    
    Args:
        model: Neural network module
        weight_decay: Weight decay coefficient
    
    Returns:
        List of parameter group dicts for optimizer
    """
    decay_params = []
    no_decay_params = []
    
    # Whitelist: layers that NEED weight decay
    whitelist_weight_modules = (nn.Linear, nn.Conv2d)
    # Blacklist: layers that DON'T need weight decay
    blacklist_weight_modules = (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            
            # All biases -> NO decay
            if pn.endswith('bias'):
                no_decay_params.append(p)
            
            # Conv/Linear weights -> WITH decay
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay_params.append(p)
            
            # Normalization weights -> NO decay
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay_params.append(p)
            
            # Default: NO decay for safety
            else:
                no_decay_params.append(p)

    # Sanity check
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    total_params = len(decay_params) + len(no_decay_params)
    assert len(param_dict) == total_params, \
        f"Parameter count mismatch: {total_params} filtered vs {len(param_dict)} in model"

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def merge_optimizer_params(
    loss_fn_params, 
    actor_groups: list[dict], 
    qvalue_groups: list[dict]
) -> list[dict]:
    """Merge parameter groups from different sources, removing duplicates.
    
    Priority order:
    1. Actor groups (custom weight decay logic)
    2. QValue groups (custom weight decay logic)
    3. Loss module leftovers (e.g., log_alpha in SAC)
    
    Args:
        loss_fn_params: Iterable from loss_fn.parameters()
        actor_groups: Parameter groups from actor model
        qvalue_groups: Parameter groups from qvalue model
    
    Returns:
        List of parameter group dicts for optimizer
    """
    final_groups = []
    seen_param_ids = set()

    # Priority 1: Actor Model
    for group in actor_groups:
        params = group['params']
        new_params = [p for p in params if id(p) not in seen_param_ids]
        seen_param_ids.update(id(p) for p in new_params)
        
        if new_params:
            new_group = group.copy()
            new_group['params'] = new_params
            final_groups.append(new_group)

    # Priority 2: QValue Model
    for group in qvalue_groups:
        params = group['params']
        new_params = [p for p in params if id(p) not in seen_param_ids]
        seen_param_ids.update(id(p) for p in new_params)
        
        if new_params:
            new_group = group.copy()
            new_group['params'] = new_params
            final_groups.append(new_group)

    # Priority 3: Loss Module Leftovers
    leftover_params = [p for p in loss_fn_params if id(p) not in seen_param_ids]
    
    if leftover_params:
        # Leftover params (like log_alpha) should not have weight decay
        final_groups.append({
            'params': leftover_params, 
            'weight_decay': 0.0 
        })

    return final_groups


def check_params_changed(model: nn.Module, model_name: str, old_params: dict) -> bool:
    """Check if model parameters changed after optimization step.
    
    Useful for debugging gradient flow issues.
    
    Args:
        model: Neural network module
        model_name: Name for logging
        old_params: Dictionary of old parameter values
    
    Returns:
        True if any parameters changed, False otherwise
    """
    changed = False
    print(f"Checking updates for {model_name}...")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            old_p = old_params[name]
            diff = (param - old_p).abs().sum().item()
            
            if diff > 0:
                changed = True
                if "weight" in name and diff > 1e-6:
                    print(f"  ✓ {name} changed (diff: {diff:.6f})")
            else:
                print(f"  ⚠️ {name} did NOT change (diff: 0.0)")
    
    if changed:
        print(f"✅ {model_name} weights updated successfully.")
    else:
        print(f"❌ {model_name} weights did NOT update. Check gradients or learning rate.")
    
    return changed


def evaluate_agent(
    actor_0,
    actor_1,
    env: HexEnv, 
    n_games: int = 50
) -> dict:
    """Evaluate actor against random policy.
    Plays n_games total (half as each player) against a random opponent.

    Args:
        actor_0: Actor network (ProbabilisticActor)
        actor_1: Actor network (ProbabilisticActor/MaskedRandomPolicy)
        env: Game environment (HexEnv), must be single-instance
        n_games: Number of games to play

    Returns:
        Dictionary with win statistics (for actor 0):
        - win_rate: Overall win rate
        - wins_as_p0: Wins as Player 0
        - games_as_p0: Games played as Player 0
        - wins_as_p1: Wins as Player 1
        - games_as_p1: Games played as Player 1
        - total_wins: Total wins
        - total_games: Total games
    """    
    wins_as_p0 = wins_as_p1 = 0
    games_as_p0 = games_as_p1 = n_games

    with torch.no_grad():
        # Play as Player 0 (Red)
        for _ in range(games_as_p0):
            td = env.reset().to(DEVICE)
            done = False
            step_count = 0

            while not done and step_count < 100:
                # Get current player
                current_player = int(td['observation'][0, 0, 2].item())

                # Actor's turn
                if current_player == 0:  # Actor 0's turn
                    td = td.to(DEVICE)
                    td = actor_0(td)
                else:  # Actor 1's turn
                    td = td.to(STORAGE_DEVICE)
                    td = actor_1(td)

                # Step the environment
                td = env.step(td)['next']
                done = td['done'].item()
                step_count += 1

                # Check whether P0 won
                if done and current_player == 0:
                    wins_as_p0 += 1
                    break

        # Play as Player 1 (Blue)
        for _ in range(games_as_p1):
            td = env.reset().to(DEVICE)
            done = False
            step_count = 0

            while not done and step_count < 100:
                # Get current player
                current_player = int(td['observation'][0, 0, 2].item())

                # Actor's turn
                if current_player == 1:  # Actor 0's turn
                    td = td.to(DEVICE)
                    td = actor_0(td)
                else:  # Actor 1's turn
                    td = td.to(STORAGE_DEVICE)
                    td = actor_1(td)

                # Step the environment
                td = env.step(td)['next']
                done = td['done'].item()
                step_count += 1

                # Check whether P0 won
                if done and current_player == 1:
                    wins_as_p1 += 1
                    break

    total_wins = wins_as_p0 + wins_as_p1
    win_rate = total_wins / (2 * n_games)

    return {
        'win_rate': win_rate,
        'wins_as_p0': wins_as_p0,
        'games_as_p0': games_as_p0,
        'wins_as_p1': wins_as_p1,
        'games_as_p1': games_as_p1,
        'total_wins': total_wins,
        'total_games': n_games * 2
    }