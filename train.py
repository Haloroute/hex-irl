import os, torch, torchrl

import matplotlib.pyplot as plt
import torch.optim as optim

from collections import defaultdict
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
# from torchrl.envs import SerialEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss
from tqdm import tqdm

import rl.config as cfg
from rl.env import HexEnv, MaskedSerialEnv
from rl.policy import TransformerQL, MaskWrapper


os.environ["RL_WARNINGS"] = "False"

def main():
    # ------------------------------------
    # SETUP ENVIRONMENT, POLICY, ACTOR, LOSS, BUFFER, COLLECTOR, OPTIMIZER
    # ------------------------------------
    # --- Environment Setup ---
    env = MaskedSerialEnv(
        num_workers=len(cfg.BOARD_SIZES),
        create_env_fn=[
            lambda: HexEnv(board_size=board_size, max_board_size=cfg.MAX_BOARD_SIZE, swap_rule=cfg.SWAP_RULE)
            for board_size in cfg.BOARD_SIZES
        ],
        device=cfg.DEVICE
    )
    # env = HexEnv(board_size=cfg.MAX_BOARD_SIZE, max_board_size=cfg.MAX_BOARD_SIZE, swap_rule=cfg.SWAP_RULE).to(cfg.DEVICE)

    # --- Policy Setup ---
    model = TransformerQL(
        conv_layers=cfg.CONV_LAYERS,
        n_encoder_layers=cfg.N_ENCODER_LAYERS,
        d_input=cfg.N_INPUT_CHANNELS,
        n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT,
        output_flatten=cfg.OUTPUT_FLATTEN,
    ).to(cfg.DEVICE)
    policy = MaskWrapper(model).to(cfg.DEVICE)

    # --- Actor and Exploration Setup ---
    q_actor = QValueActor(
        module=model,
        in_keys=["observation"],
        spec=env.action_spec
    )
    turn_actor = QValueActor(
        module=policy,
        in_keys=["observation"],
        spec=env.action_spec
    )
    # actor = TensorDictSequential(
    #     q_actor,
    #     EGreedyModule(
    #         spec=env.action_spec,
    #         annealing_num_steps=1_000_000
    #     )
    # )

    # --- Loss function ---
    loss_fn = DQNLoss(
        value_network=q_actor,
    )

    # --- Buffer and Collector
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.BUFFER_MAX_SIZE, device=cfg.DEVICE),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.BATCH_SIZE
    )
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=turn_actor,
        frames_per_batch=cfg.FRAMES_PER_BATCH,
        total_frames=cfg.TOTAL_FRAMES,
        device=cfg.DEVICE
    )

    # --- Optimizer ---
    optimizer = optim.Adam(q_actor.parameters(), lr=cfg.LEARNING_RATE)
    q_actor.train()

    # ------------------------------------
    # TRAINING LOOP
    # ------------------------------------
    for i, tensordict_data in enumerate(t := tqdm(collector, leave=False)):
        # Add data to replay buffer
        tensordict_data = tensordict_data.reshape(-1)  # Flatten the batch size from all envs
        buffer.extend(tensordict_data)
    
        # Sample from replay buffer
        if len(buffer) >= cfg.BATCH_SIZE:
            batch = buffer.sample()
            loss = loss_fn(batch).get("loss")

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Logging
        t.set_postfix(loss=loss.item())

if __name__ == "__main__":
    main()