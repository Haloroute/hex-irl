"""
Hex Game RL Evaluation Script

This script evaluates a pre-trained agent on the Hex game using:
- Transformer-based policy and Q-value networks
"""

import torch
from pathlib import Path
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.modules import ProbabilisticActor, MaskedCategorical

from rl.environment import HexEnv
from rl.model.v1.network import HexModel
from rl.model.v2.network import Conv, RotationWrapperModel as HexModelV2
from rl.policy.mcts import MCTSPolicy
from rl.policy.random import MaskedRandomPolicy
from rl.policy.v1.wrapper import ModelWrapper
from rl.policy.v2.wrapper import ModelWrapper as ModelWrapperV2
from rl.utility import evaluate_agent, init_params
from rl.config import (
    DEVICE, STORAGE_DEVICE,
    BOARD_SIZE, MAX_BOARD_SIZE, SWAP_RULE, N_CHANNEL,
    MODEL_PARAMS, BUFFER_SIZE, MAX_N_STEPS, N_EPISODES_PER_EPOCH, N_MEMMAP_CHUNKS,
    INITIAL_TEMPERATURE, FINAL_TEMPERATURE, DECAY_RATE,
    N_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    TOTAL_FRAMES, WARMUP_FRAMES, OPTIMIZATION_STEPS, GAMMA, TAU, GRAD_CLIP_NORM,
    LOG_INTERVAL, RANDOM_EVAL_INTERVAL, PAST_EVAL_INTERVAL, MCTS_EVAL_INTERVAL, EVAL_GAMES, MCTS_ITERMAX,
    CHECKPOINT_DIR, RESULTS_DIR
)


def load_latest_checkpoint(board_size: int, network: TensorDictModule):
    """Load the latest checkpoint for the given board size if available."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None, None
    pattern = f"hex_{board_size}x{board_size}_e*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None, None

    def _epoch_from_path(path: Path) -> int:
        try:
            return int(path.stem.split("_e")[-1])
        except Exception:
            return -1

    latest_path = max(checkpoints, key=_epoch_from_path)
    checkpoint = torch.load(latest_path, map_location=DEVICE)
    network.load_state_dict(checkpoint["state_dict"])
    print(f"✓ Loaded checkpoint: {latest_path.name}")
    return checkpoint, latest_path


def final_evaluation(env: EnvBase, actor_1, actor_2, device_0: torch.device, device_1: torch.device):
    """Perform final evaluation with pre-defined number of games."""
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION - {EVAL_GAMES} Games")
    print("=" * 60)
    final_eval = evaluate_agent(
        actor_1, actor_2,
        device_0=device_0, device_1=device_1,
        env=env, n_games=EVAL_GAMES
    )
    print(f"\nFinal Results:")
    print(f"  Total Win Rate: {final_eval['win_rate']:.1%}")
    print(f"  Wins as Player 0 (Red): {final_eval['wins_as_p0']}/{final_eval['games_as_p0']} "
          f"({final_eval['wins_as_p0']/final_eval['games_as_p0']:.1%})")
    print(f"  Wins as Player 1 (Blue): {final_eval['wins_as_p1']}/{final_eval['games_as_p1']} "
          f"({final_eval['wins_as_p1']/final_eval['games_as_p1']:.1%})")
    print("=" * 60)


def main():
    print("=" * 60)
    print("HEX GAME RL EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}")
    print("=" * 60)

    # Environment
    create_hex_env = lambda: HexEnv(
        board_size=BOARD_SIZE,
        max_board_size=MAX_BOARD_SIZE,
        swap_rule=SWAP_RULE,
        device=STORAGE_DEVICE
    )
    environment = TransformedEnv(
        create_hex_env(),
        ActionMask()
    )

    # Model and actor
    base_model = Conv(**MODEL_PARAMS)
    model = HexModelV2(base_model)
    model_wrapper = ModelWrapper(model, board_size=BOARD_SIZE, temperature=0).train().to(DEVICE)
    init_params(model)
    network = TensorDictModule(
        model_wrapper,
        in_keys=["observation", "action_mask"],
        out_keys=["logits", "mask"]
    )
    actor = ProbabilisticActor(
        network,
        in_keys=["logits", "mask"],
        spec=environment.action_spec,
        distribution_class=MaskedCategorical
    )

    # Load latest checkpoint
    checkpoint, checkpoint_path = load_latest_checkpoint(BOARD_SIZE, network)
    if checkpoint_path is None:
        print("⚠️ No checkpoint found. Exiting.")
        return

    # MCTS rollout input
    default_rollouts = 4 * (BOARD_SIZE ** 2)
    try:
        user_input = input(f"Nhập số rollout MCTS (mặc định {default_rollouts}): ").strip()
        mcts_rollouts = int(user_input) if user_input else default_rollouts
    except Exception:
        print(f"⚠️ Giá trị không hợp lệ, dùng mặc định {default_rollouts}")
        mcts_rollouts = default_rollouts
    # mcts_policy = MCTSPolicy(environment, itermax=mcts_rollouts)
    mcts_policy = MaskedRandomPolicy(environment.action_spec)
    print(f"✓ Sử dụng MCTS rollout: {mcts_rollouts}")

    # Final evaluation
    final_evaluation(environment, actor_1=actor, actor_2=mcts_policy, device_0=DEVICE, device_1=STORAGE_DEVICE)


if __name__ == "__main__":
    main()