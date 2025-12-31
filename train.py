"""
Hex Game RL Training Script using Discrete SAC with Negamax

This script trains an agent to play Hex using:
- Transformer-based policy and Q-value networks
- Discrete SAC with Negamax adjustment for zero-sum games
"""

import copy, math, os, shutil, torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from pyinstrument import Profiler
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import EnvBase, SerialEnv, TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from tqdm.auto import tqdm

# Import custom modules
from rl.dataloader import HexDataCollector, HexRolloutDataset
from rl.environment import HexEnv
from rl.loss import SimpleLoss
from rl.model.network import HexModel
from rl.model.v2.network import Conv, RotationWrapperModel as HexModelV2
from rl.policy.mcts import MCTSPolicy
from rl.policy.random import MaskedRandomPolicy
from rl.policy.wrapper import ModelWrapper
from rl.policy.v2.wrapper import ModelWrapper as ModelWrapperV2
from rl.utility import (
    init_params,
    get_optimizer_params,
    merge_optimizer_params,
    check_params_changed,
    evaluate_agent
)
from rl.config import (
    DEVICE, STORAGE_DEVICE,
    BOARD_SIZE, MAX_BOARD_SIZE, SWAP_RULE, N_CHANNEL,
    MODEL_PARAMS, BUFFER_SIZE, MAX_N_STEPS, N_SAMPLES_PER_EPOCH, N_EPISODES_PER_EPOCH, N_TRAINING_ROUNDS_PER_EPOCH, N_MEMMAP_CHUNKS,
    INITIAL_TEMPERATURE, FINAL_TEMPERATURE, DECAY_RATE,
    N_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    TOTAL_FRAMES, WARMUP_FRAMES, OPTIMIZATION_STEPS, GAMMA, TAU, GRAD_CLIP_NORM,
    LOG_INTERVAL, RANDOM_EVAL_INTERVAL, PAST_EVAL_INTERVAL, MCTS_EVAL_INTERVAL, EVAL_GAMES, MCTS_ITERMAX,
    CHECKPOINT_DIR, RESULTS_DIR
)


# def test_components():
#     """Test all components before training."""
#     print("=" * 60)
#     print("COMPONENT TESTING")
#     print("=" * 60)
    
#     # Test environment
#     print("\n1. Testing Environment...")
#     test_env = HexEnv(
#         board_size=BOARD_SIZE,
#         max_board_size=MAX_BOARD_SIZE,
#         device=DEVICE
#     )
#     test_td = test_env.reset()
#     print(f"   ✓ Reset output keys: {test_td.keys()}")
#     print(f"   ✓ Observation shape: {test_td['observation'].shape}")
#     print(f"   ✓ Action mask shape: {test_td['action_mask'].shape}")
    
#     test_td = test_env.rand_step(test_td)
#     print(f"   ✓ Step completed successfully")
    
#     # Test model
#     print("\n2. Testing Model...")
#     test_model = HexModel(**MODEL_PARAMS).to(DEVICE)
#     test_input = test_td['observation'].unsqueeze(0)
#     test_output = test_model(test_input)
#     print(f"   ✓ Model output shape: {test_output.shape}")
    
#     print("\n✅ All components working!")
#     print("=" * 60)


def load_latest_checkpoint(board_size: int, network: TensorDictModule, optimizer: optim.Optimizer):
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
        except (IndexError, ValueError):
            return -1

    latest_path = max(checkpoints, key=_epoch_from_path)
    checkpoint = torch.load(latest_path, map_location=DEVICE)
    try:
        network.load_state_dict(checkpoint["state_dict"])
        print(f"✓ Loaded model state: {latest_path.name}")
    except Exception as e:
        print(f"⚠️  Failed to load model state: {latest_path.name} | Error: {e}")

    try:
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✓ Loaded optimizer state: {latest_path.name}")
    except Exception as e:
        print(f"⚠️  Failed to load optimizer state: {latest_path.name} | Error: {e}")
    return checkpoint, latest_path


def training_loop(
    environment: EnvBase,
    actor: ProbabilisticActor,
    network: TensorDictModule,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    collector: HexDataCollector,
    random_policy: MaskedRandomPolicy,
    start_epoch: int = 0,
    training_history: dict = None
):
    """Main training loop."""
    print("=" * 60)
    print("MAIN TRAINING LOOP")
    print("=" * 60)
    
    # Training metrics
    if training_history is None:
        training_history = {
            'epoch': [],
            'loss': {
                'train': [],
                'val': []
            },
            'win_rate': {
                'random': [],
                'past': []
            },
            'frames': []
        }
    n_total_frames = training_history['frames'][-1] if training_history['frames'] else 0
    
    # Create checkpoint directory
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # START TRAINING LOOP
    for epoch in range(start_epoch, N_EPOCHS):
        print(f"{'='*20} Epoch {epoch+1}/{N_EPOCHS} {'='*20}")

        # Create lists to track losses
        train_loss_list = []

        # Create past actor for compare the performance with current actor
        past_actor = copy.deepcopy(actor)
        past_actor.eval()
        for param in past_actor.parameters():
            param.requires_grad = False

        # Set temperature for current epoch
        if network.module.temperature is not None:
            new_temperature = max(
                FINAL_TEMPERATURE,
                INITIAL_TEMPERATURE * (DECAY_RATE ** epoch)
            )
            network.module.temperature = new_temperature
            print(f"✓ Set temperature to {new_temperature:.4f}")

        # Create training data
        # profiler = Profiler()
        # profiler.start()
        collector.collect_and_save(N_EPISODES_PER_EPOCH, save_dir="data", filename=f"epoch_{epoch+1}_data")
        # profiler.stop()
        # profiler.open_in_browser()

        # Create dataset and dataloader
        train_dataset = HexRolloutDataset(
            data_path=Path("data"), 
            train=True, 
            n_memmap_chunks=N_MEMMAP_CHUNKS, 
            device=STORAGE_DEVICE
        )
        val_dataset = HexRolloutDataset(
            data_path=Path("data"),
            train=False,
            n_memmap_chunks=N_MEMMAP_CHUNKS,
            device=STORAGE_DEVICE
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: torch.stack(x))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: torch.stack(x))

        # Set actor to training mode
        actor.train()

        # Training over the train dataset
        for round in range(N_TRAINING_ROUNDS_PER_EPOCH):
            for batch_data in (iteration := tqdm(train_loader, desc=f"Training Round {round+1}/{N_TRAINING_ROUNDS_PER_EPOCH}", leave=False)):
                # Data preparation
                batch_data: TensorDict = batch_data.to(DEVICE)
                observation, action, action_mask, reward = (
                    batch_data.get('observation').to(DEVICE),
                    batch_data.get('action').to(DEVICE),
                    batch_data.get('action_mask').to(DEVICE),
                    batch_data.get('reward').to(DEVICE)
                )
                
                # Compute loss
                input_data: TensorDict = TensorDict({
                    'observation': observation,
                    'action_mask': action_mask
                })
                logits: Tensor = network(input_data).get('logits')
                loss: Tensor = loss_fn(logits, action, reward)

                # Gradient descent
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()),
                    max_norm=GRAD_CLIP_NORM
                )            
                optimizer.step()
                
                # Progress bar update
                train_loss_list.append(loss.item())
                iteration.set_postfix({
                    'Train Loss': loss.item()
                })

        # Evaluation over the validation dataset
        actor.eval()
        with torch.no_grad():
            val_loss_list = []
            for batch_data in (val_iteration := tqdm(val_loader, desc="Validation Batches", leave=False)):
                batch_data: TensorDict = batch_data.to(DEVICE)
                observation, action, action_mask, reward = (
                    batch_data.get('observation').to(DEVICE),
                    batch_data.get('action').to(DEVICE),
                    batch_data.get('action_mask').to(DEVICE),
                    batch_data.get('reward').to(DEVICE)
                )

                # Compute loss
                input_data: TensorDict = TensorDict({
                    'observation': observation,
                    'action_mask': action_mask
                })
                logits: Tensor = network(input_data).get('logits')
                loss: Tensor = loss_fn(logits, action, reward)

                # Progress bar update
                val_loss_list.append(loss.item())
                val_iteration.set_postfix({
                    'Val Loss': loss.item()
                })

        # Evaluation and logging
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        n_collected_frames = len(train_dataset)
        n_total_frames += n_collected_frames
        
        # Store metrics
        training_history['epoch'].append(epoch)
        training_history['loss']['train'].append(avg_train_loss)
        training_history['loss']['val'].append(avg_val_loss)
        training_history['frames'].append(n_total_frames)

        print(f"{'='*60}")
        print(f"Epoch {epoch + 1} | Frames: {n_total_frames:,}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"{'='*60}")

        # # Evaluation against past policy
        # eval_results = evaluate_agent(
        #     actor, past_actor,
        #     device_0=DEVICE, device_1=STORAGE_DEVICE,
        #     env=environment, n_games=EVAL_GAMES
        # )
        # win_rate = eval_results['win_rate']
        # training_history['win_rate']['past'].append(win_rate)
        
        # print("Evaluation against Past Itself:")
        # print(f"WinRate: {win_rate:.1%} ({eval_results['total_wins']}/{eval_results['total_games']})")
        # print(f"  - As P0: {eval_results['wins_as_p0']}/{eval_results['games_as_p0']}")
        # print(f"  - As P1: {eval_results['wins_as_p1']}/{eval_results['games_as_p1']}")

        # Set temperature to negative for evaluation (to disable random logit boost)
        network.module.temperature *= -1
        # Evaluate against random policy
        eval_results = evaluate_agent(
            actor, random_policy,
            device_0=DEVICE, device_1=STORAGE_DEVICE,
            env=environment, n_games=EVAL_GAMES
        )
        # Restore temperature back (multiply -1 again to get original)
        network.module.temperature *= -1

        win_rate = eval_results['win_rate']
        training_history['win_rate']['random'].append(win_rate)

        print("Evaluation against Random Policy:")
        print(f"WinRate: {win_rate:.1%} ({eval_results['total_wins']}/{eval_results['total_games']})")
        print(f"  - As P0: {eval_results['wins_as_p0']}/{eval_results['games_as_p0']}")
        print(f"  - As P1: {eval_results['wins_as_p1']}/{eval_results['games_as_p1']}")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"hex_{BOARD_SIZE}x{BOARD_SIZE}_e{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'win_rate': win_rate,
            'training_history': training_history
        }, checkpoint_path)

        print(f"✓ Checkpoint Saved! (WinRate: {win_rate:.1%})")
        print(f"{'='*60}\n")

        # NEW: export plots after each epoch
        plot_training_curves(training_history)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    
    return training_history


def plot_training_curves(training_history):
    """Plot and save training curves."""
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Loss
    axes[0].plot(training_history['epoch'], training_history['loss']['train'], label='Train Loss', marker='o')
    axes[0].plot(training_history['epoch'], training_history['loss']['val'], label='Validation Loss', marker='o')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # Win Rate with Random Policy
    axes[1].plot(training_history['epoch'], training_history['win_rate']['random'], marker='o')
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Random Baseline')
    axes[1].set_title('Win Rate vs Random Policy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Win Rate')
    axes[1].legend()
    axes[1].grid(True)

    # # Win Rate with Past Policy
    # axes[1, 0].plot(training_history['epoch'], training_history['win_rate']['past'], marker='o')
    # axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Random Baseline')
    # axes[1, 0].set_title('Win Rate vs Past Policy')
    # axes[1, 0].set_xlabel('Epoch')
    # axes[1, 0].set_ylabel('Win Rate')
    # axes[1, 0].legend()
    # axes[1, 0].grid(True)

    plt.tight_layout()
    plot_path = results_dir / f"training_curves_{BOARD_SIZE}x{BOARD_SIZE}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n✓ Training curves saved to {plot_path}")
    plt.close()


def final_evaluation(env: EnvBase, actor_1, actor_2, device_0: torch.device, device_1: torch.device):
    """Perform final evaluation with 100 games."""
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
    """Main training pipeline."""
    print("=" * 60)
    print("HEX GAME RL TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Total Frames: {TOTAL_FRAMES:,}")
    print(f"Warmup Frames: {WARMUP_FRAMES:,}")
    print("=" * 60)

    # 1. Create environment
    create_hex_env = lambda: HexEnv(
        board_size=BOARD_SIZE,
        max_board_size=MAX_BOARD_SIZE,
        swap_rule=SWAP_RULE,
        device=STORAGE_DEVICE
    )
    # serial_env = TransformedEnv(
    #     SerialEnv(num_workers=1, create_env_fn=create_hex_env),
    #     ActionMask()
    # )
    # evaluate_env = TransformedEnv(
    #     create_hex_env(),
    #     ActionMask()
    # )
    environment = TransformedEnv(
        create_hex_env(),
        ActionMask()
    )

    # 2. Create models, wrapper, and policy
    base_model = Conv(**MODEL_PARAMS)
    model = HexModelV2(base_model)
    model_wrapper = ModelWrapperV2(model, board_size=BOARD_SIZE, temperature=INITIAL_TEMPERATURE).train().to(DEVICE)
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
    ).train().to(DEVICE)

    # 3. Create loss function, optimizer
    loss_fn = SimpleLoss(ratio=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Load latest checkpoint if available
    checkpoint, checkpoint_path = load_latest_checkpoint(BOARD_SIZE, network, optimizer)
    start_epoch = 0
    training_history = None
    if checkpoint:
        start_epoch = checkpoint.get('epoch', -1) + 1
        training_history = checkpoint.get('training_history')
        print(f"Resuming from checkpoint {checkpoint_path.name} at epoch {start_epoch}")

    # 4. Create data collector
    shutil.rmtree("data", ignore_errors=True)  # Clear previous data
    os.makedirs("data", exist_ok=True)
    collector = HexDataCollector(
        environment,
        actor,
        gamma=GAMMA,
        device=DEVICE,
        storage_device=STORAGE_DEVICE
    )

    # 5. Create evaluated policy
    random_policy = MaskedRandomPolicy(environment.action_spec)
    mcts_policy = MCTSPolicy(environment, itermax=MCTS_ITERMAX)

    # 6. Start training
    training_history = training_loop(
        environment,
        actor,
        network,
        loss_fn,
        optimizer,
        collector,
        random_policy,
        start_epoch=start_epoch,
        training_history=training_history
    )

    # 9. Plot results
    plot_training_curves(training_history)

    # 10. Final evaluation
    final_evaluation(environment, actor_1=actor, actor_2=mcts_policy, device_0=DEVICE, device_1=STORAGE_DEVICE)


if __name__ == "__main__":
    main()