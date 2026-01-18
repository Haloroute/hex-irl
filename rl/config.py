"""Configuration file for Hex RL training.

This module contains all hyperparameters and settings for:
- Environment configuration
- Model architecture
- Training parameters
- Data collection settings
"""

import torch

# ---------------------------------
# DEVICE CONFIGURATION
# ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"  # Force CPU for debugging/testing
# STORAGE_DEVICE = DEVICE  # Device for replay buffer storage
STORAGE_DEVICE = "cpu"  # Device for replay buffer storage

# ---------------------------------
# ENVIRONMENT CONFIGURATION
# ---------------------------------
BOARD_SIZE = 11  # Size of the Hex board (board_size x board_size)
MAX_BOARD_SIZE = BOARD_SIZE  # Maximum board size for padding
SWAP_RULE = True  # Whether to enable swap rule in Hex game
N_CHANNEL = 3  # Number of input channels (Red, Blue, Swap Rule)

# ---------------------------------
# MODEL ARCHITECTURE
# ---------------------------------
MODEL_PARAMS = {
    "board_size": BOARD_SIZE,
    "conv_layers": [(64, 3)] * 16,  # List of (out_channels, kernel_size) tuples
    "n_encoder_layers": 0,  # Number of transformer encoder layers
    "d_input": N_CHANNEL,  # Input feature dimension
    "n_heads": 4,  # Number of attention heads
    "d_ff": 1024,  # Feedforward network dimension
    "dropout": 0.001,  # Dropout rate
    "output_flatten": True,  # Must be True for RL agents
}

# ---------------------------------
# DATA COLLECTION
# ---------------------------------
BUFFER_SIZE = 100_000  # Maximum size of replay buffer
MAX_N_STEPS = BOARD_SIZE ** 2  # Number of max steps per episode
N_SAMPLES_PER_EPOCH = 10_000  # Number of samples collected per rollout
N_EPISODES_PER_EPOCH = int(N_SAMPLES_PER_EPOCH / MAX_N_STEPS) + 1  # Episodes collected per epoch
N_TRAINING_ROUNDS_PER_EPOCH = 1  # Number of training rounds per epoch
N_MEMMAP_CHUNKS = int(BUFFER_SIZE / N_SAMPLES_PER_EPOCH) + 1  # Number of memmap chunks to load for dataset

# --------------------------------
# TEMPERATURE SETTINGS
# ---------------------------------
INITIAL_TEMPERATURE = 1.0  # Initial temperature for action selection
FINAL_TEMPERATURE = 0.1  # Final temperature after decay
DECAY_RATE = 0.995  # Decay rate per epoch

# ---------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------
# Optimization Settings
N_EPOCHS = 10_000  # Number of epochs per training iteration
BATCH_SIZE = 512  # Batch size for training
LR = 1e-4  # Learning rate (Adam/AdamW)
WEIGHT_DECAY = 1e-5  # Weight decay for optimizer

# Training Loop Configuration
TOTAL_FRAMES = 1_000_000  # Total training frames
WARMUP_FRAMES = 10_000  # Random exploration frames before training starts
OPTIMIZATION_STEPS = 10  # UTD Ratio: gradient updates per data collection
GAMMA = 0.95  # Discount factor for future rewards
TAU = 0.005  # Soft update coefficient for target network (Polyak averaging)
GRAD_CLIP_NORM = 1.0  # Maximum norm for gradient clipping

# Logging and Evaluation
LOG_INTERVAL = 10  # Log and evaluate every N iterations
RANDOM_EVAL_INTERVAL = 100  # Evaluate using random policy every N iterations
PAST_EVAL_INTERVAL = 200  # Evaluate against past actor every N iterations
MCTS_EVAL_INTERVAL = 1000  # Evaluate using MCTS policy every N iterations
EVAL_GAMES = 100  # Number of games for evaluation against random policy
EVAL_EVERY_EPOCHS = 10  # Evaluate every N epochs
MCTS_ITERMAX = 1000  # MCTS iterations for evaluation

# ---------------------------------
# CHECKPOINT SETTINGS
# ---------------------------------
CHECKPOINT_DIR = "checkpoints"  # Directory for saving models
RESULTS_DIR = "results"  # Directory for saving results/plots