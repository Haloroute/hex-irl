"""
Hex Game RL Training Script using Discrete SAC with Negamax

This script trains an agent to play Hex using:
- Transformer-based policy and Q-value networks
- Discrete SAC with Negamax adjustment for zero-sum games
- Soft Actor-Critic with automatic entropy tuning
- Experience replay with warmup phase
"""

import math, torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import SerialEnv, TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from torchrl.objectives import SoftUpdate

# Import custom modules
from rl.environment import HexEnv
from rl.model.network import HexModel
from rl.policy.mcts import MCTSPolicy
from rl.policy.random import MaskedRandomPolicy
from rl.policy.wrapper import ActorWrapper, CriticWrapper
from rl.loss import NegamaxDiscreteSACLoss
from rl.utility import (
    init_params, 
    get_optimizer_params, 
    merge_optimizer_params,
    check_params_changed,
    evaluate_agent
)
from rl.config import (
    DEVICE, BOARD_SIZE, MAX_BOARD_SIZE,
    MODEL_PARAMS, BUFFER_SIZE, N_FRAMES_PER_BATCH, STORAGE_DEVICE,
    BATCH_SIZE, LR, WEIGHT_DECAY,
    TOTAL_FRAMES, WARMUP_FRAMES, OPTIMIZATION_STEPS, TAU,
    LOG_INTERVAL, EVAL_GAMES, MCTS_ITERMAX, GAMMA, GRAD_CLIP_NORM,
    CHECKPOINT_DIR, RESULTS_DIR
)


def test_components():
    """Test all components before training."""
    print("=" * 60)
    print("COMPONENT TESTING")
    print("=" * 60)
    
    # Test environment
    print("\n1. Testing Environment...")
    test_env = HexEnv(
        board_size=BOARD_SIZE,
        max_board_size=MAX_BOARD_SIZE,
        device=DEVICE
    )
    test_td = test_env.reset()
    print(f"   ✓ Reset output keys: {test_td.keys()}")
    print(f"   ✓ Observation shape: {test_td['observation'].shape}")
    print(f"   ✓ Action mask shape: {test_td['action_mask'].shape}")
    
    test_td = test_env.rand_step(test_td)
    print(f"   ✓ Step completed successfully")
    
    # Test model
    print("\n2. Testing Model...")
    test_model = HexModel(**MODEL_PARAMS).to(DEVICE)
    test_input = test_td['observation'].unsqueeze(0)
    test_output = test_model(test_input)
    print(f"   ✓ Model output shape: {test_output.shape}")
    
    print("\n✅ All components working!")
    print("=" * 60)


def warmup_phase(replay_buffer, warmup_collector):
    """Execute warmup phase with random exploration."""
    print("=" * 60)
    print("WARMUP PHASE - Random Exploration")
    print("=" * 60)
    
    current_frames = 0
    warmup_iterations = 0
    
    for warmup_batch in warmup_collector:
        warmup_batch = warmup_batch.reshape(-1)
        replay_buffer.extend(warmup_batch)
        current_frames += len(warmup_batch)
        warmup_iterations += 1
        
        if warmup_iterations % 5 == 0:
            print(f"Warmup Progress: {current_frames}/{WARMUP_FRAMES} frames "
                  f"({current_frames/WARMUP_FRAMES*100:.1f}%) | "
                  f"Buffer Size: {len(replay_buffer)}")
    
    print(f"\n✓ Warmup completed! Buffer size: {len(replay_buffer)}")
    print("=" * 60)
    
    return current_frames


def training_loop(
    collector, replay_buffer, loss_fn, optimizer, updater,
    actor, qvalue_network, serial_env, evaluate_env, total_frames_collected
):
    """Main training loop."""
    print("\n" + "=" * 60)
    print("MAIN TRAINING LOOP")
    print("=" * 60)
    
    # Training metrics
    iteration = 0
    best_win_rate = 0.0
    training_history = {
        'iteration': [],
        'actor_loss': [],
        'qvalue_loss': [],
        'alpha_loss': [],
        'win_rate': [],
        'frames': []
    }
    
    # Create checkpoint directory
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set actor to training mode
    actor.train()
    
    for batch_data in collector:
        # A. Add new data to replay buffer
        batch_data = batch_data.reshape(-1)
        replay_buffer.extend(batch_data)
        total_frames_collected += len(batch_data)
        
        # B. Optimization loop (UTD Ratio)
        actor_losses = []
        qvalue_losses = []
        alpha_losses = []
        
        for opt_step in range(OPTIMIZATION_STEPS):
            # B1. Sample batch
            sample = replay_buffer.sample(BATCH_SIZE)
            sample = sample.to(DEVICE)
            
            # B2. Compute losses
            loss_dict = loss_fn(sample)
            
            total_loss = (
                loss_dict['loss_actor'] + 
                loss_dict['loss_qvalue'] + 
                loss_dict['loss_alpha']
            )
            
            # B3. Gradient descent
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(qvalue_network.parameters()), 
                max_norm=GRAD_CLIP_NORM
            )            
            optimizer.step()
            
            # B4. Soft update target network
            updater.step()
            
            # Collect losses
            actor_losses.append(loss_dict['loss_actor'].item())
            qvalue_losses.append(loss_dict['loss_qvalue'].item())
            alpha_losses.append(loss_dict['loss_alpha'].item())
        
        # C. Logging and evaluation
        iteration += 1
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        avg_qvalue_loss = sum(qvalue_losses) / len(qvalue_losses)
        avg_alpha_loss = sum(alpha_losses) / len(alpha_losses)
        
        # Store metrics
        training_history['iteration'].append(iteration)
        training_history['actor_loss'].append(avg_actor_loss)
        training_history['qvalue_loss'].append(avg_qvalue_loss)
        training_history['alpha_loss'].append(avg_alpha_loss)
        training_history['frames'].append(total_frames_collected)
        
        # Periodic evaluation
        if iteration % LOG_INTERVAL == 0:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} | Frames: {total_frames_collected:,}/{TOTAL_FRAMES:,}")
            print(f"{'='*60}")
            
            # Evaluate agent
            actor.eval()
            eval_results = evaluate_agent(actor, MaskedRandomPolicy(serial_env.action_spec), evaluate_env, n_games=EVAL_GAMES)
            actor.train()
            win_rate = eval_results['win_rate']
            training_history['win_rate'].append(win_rate)
            
            print(f"Loss - Actor: {avg_actor_loss:.4f} | QValue: {avg_qvalue_loss:.4f} | Alpha: {avg_alpha_loss:.4f}")
            print(f"WinRate: {win_rate:.1%} ({eval_results['total_wins']}/{eval_results['total_games']})")
            print(f"  - As P0: {eval_results['wins_as_p0']}/{eval_results['games_as_p0']}")
            print(f"  - As P1: {eval_results['wins_as_p1']}/{eval_results['games_as_p1']}")
            print(f"Buffer Size: {len(replay_buffer)}")
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                checkpoint_path = checkpoint_dir / f"hex_{BOARD_SIZE}x{BOARD_SIZE}_best.pth"
                torch.save({
                    'iteration': iteration,
                    'actor_state_dict': actor.module[0].model.state_dict(),
                    'qvalue_state_dict': qvalue_network.module.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'win_rate': win_rate,
                    'training_history': training_history
                }, checkpoint_path)
                print(f"✓ New Best Model Saved! (WinRate: {best_win_rate:.1%})")
            
            print(f"{'='*60}\n")
            
            # Early stopping
            if best_win_rate > 0.98:
                print("🎉 Target win rate achieved! Stopping training.")
                break
        else:
            # Brief progress update
            if iteration % math.sqrt(LOG_INTERVAL) == 0:
                print(f"Iter {iteration} | Frames: {total_frames_collected:,} | "
                      f"Loss: A={avg_actor_loss:.3f} Q={avg_qvalue_loss:.3f} α={avg_alpha_loss:.3f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print(f"Best Win Rate: {best_win_rate:.1%}")
    print("=" * 60)
    
    return training_history, best_win_rate


def plot_training_curves(training_history, best_win_rate):
    """Plot and save training curves."""
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Actor Loss
    axes[0, 0].plot(training_history['iteration'], training_history['actor_loss'])
    axes[0, 0].set_title('Actor Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # QValue Loss
    axes[0, 1].plot(training_history['iteration'], training_history['qvalue_loss'])
    axes[0, 1].set_title('Q-Value Loss')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Alpha Loss
    axes[1, 0].plot(training_history['iteration'], training_history['alpha_loss'])
    axes[1, 0].set_title('Alpha Loss (Temperature)')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Win Rate
    eval_iterations = [
        training_history['iteration'][i] 
        for i in range(0, len(training_history['iteration']), LOG_INTERVAL)
    ]
    axes[1, 1].plot(eval_iterations, training_history['win_rate'], marker='o')
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Random Baseline')
    axes[1, 1].set_title('Win Rate vs Random Policy')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = results_dir / f"training_curves_{BOARD_SIZE}x{BOARD_SIZE}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n✓ Training curves saved to {plot_path}")
    plt.close()


def final_evaluation(actor_1, actor_2, env, best_win_rate):
    """Perform final evaluation with 100 games."""
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION - {EVAL_GAMES} Games")
    print("=" * 60)
    
    final_eval = evaluate_agent(actor_1, actor_2, env, n_games=EVAL_GAMES)

    print(f"\nFinal Results:")
    print(f"  Total Win Rate: {final_eval['win_rate']:.1%}")
    print(f"  Wins as Player 0 (Red): {final_eval['wins_as_p0']}/{final_eval['games_as_p0']} "
          f"({final_eval['wins_as_p0']/final_eval['games_as_p0']:.1%})")
    print(f"  Wins as Player 1 (Blue): {final_eval['wins_as_p1']}/{final_eval['games_as_p1']} "
          f"({final_eval['wins_as_p1']/final_eval['games_as_p1']:.1%})")
    print(f"\nBest Win Rate During Training: {best_win_rate:.1%}")
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
        device=STORAGE_DEVICE
    )
    serial_env = TransformedEnv(
        SerialEnv(num_workers=1, create_env_fn=create_hex_env),
        ActionMask()
    )
    evaluate_env = TransformedEnv(
        create_hex_env(),
        ActionMask()
    )

    # 2. Create models
    actor_model = HexModel(**MODEL_PARAMS).train().to(DEVICE)
    qvalue_model = HexModel(**MODEL_PARAMS).train().to(DEVICE)
    init_params(actor_model)
    init_params(qvalue_model)

    # 3. Create wrappers and policy
    actor_network = TensorDictModule(
        ActorWrapper(actor_model),
        in_keys=["observation", "action_mask"],
        out_keys=["logits", "mask"]
    )
    qvalue_network = TensorDictModule(
        CriticWrapper(qvalue_model),
        in_keys=["observation"],
        out_keys=["action_value"]
    )
    actor = ProbabilisticActor(
        actor_network,
        in_keys=["logits", "mask"],
        spec=serial_env.action_spec,
        distribution_class=MaskedCategorical
    )

    # 4. Create loss function
    loss_fn = NegamaxDiscreteSACLoss(
        actor_network=actor,
        qvalue_network=qvalue_network,
        action_space=serial_env.action_spec,
        num_actions=serial_env.action_spec.n,
        skip_done_states=True,
        deactivate_vmap=True
    ).to(DEVICE)

    # 5. Create optimizer
    actor_params_groups = get_optimizer_params(actor_model, WEIGHT_DECAY)
    qvalue_params_groups = get_optimizer_params(qvalue_model, WEIGHT_DECAY)
    combined_params = merge_optimizer_params(
        loss_fn_params=loss_fn.parameters(),
        actor_groups=actor_params_groups,
        qvalue_groups=qvalue_params_groups
    )
    optimizer = optim.AdamW(params=combined_params, lr=LR, weight_decay=WEIGHT_DECAY)
    updater = SoftUpdate(loss_module=loss_fn, tau=TAU)

    # 6. Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(BUFFER_SIZE, device=STORAGE_DEVICE),
        sampler=SamplerWithoutReplacement(),
        batch_size=BATCH_SIZE
    )

    # 7. Warmup phase
    warmup_collector = SyncDataCollector(
        create_env_fn=serial_env,
        policy=MaskedRandomPolicy(serial_env.action_spec),
        frames_per_batch=N_FRAMES_PER_BATCH,
        total_frames=WARMUP_FRAMES,
        device=DEVICE,
        storing_device=STORAGE_DEVICE,
        env_device=STORAGE_DEVICE
    )
    total_frames_collected = warmup_phase(replay_buffer, warmup_collector)

    # 8. Main training loop
    collector = SyncDataCollector(
        create_env_fn=serial_env,
        policy=actor,
        frames_per_batch=N_FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES - WARMUP_FRAMES,
        device=DEVICE,
        storing_device=STORAGE_DEVICE,
        env_device=STORAGE_DEVICE
    )

    training_history, best_win_rate = training_loop(
        collector, replay_buffer, loss_fn, optimizer, updater,
        actor, qvalue_network, serial_env, evaluate_env, total_frames_collected
    )

    # 9. Plot results
    plot_training_curves(training_history, best_win_rate)

    # 10. Final evaluation
    random_actor = MCTSPolicy(evaluate_env, itermax=MCTS_ITERMAX)
    final_evaluation(actor, random_actor, evaluate_env, best_win_rate)


if __name__ == "__main__":
    main()