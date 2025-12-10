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
BOARD_SIZE = 5  # Size of the Hex board (board_size x board_size)
MAX_BOARD_SIZE = 5  # Maximum board size for padding
SWAP_RULE = True  # Whether to enable swap rule in Hex game
N_CHANNEL = 5  # Number of input channels (Red, Blue, Current Player, Valid Board)

# ---------------------------------
# MODEL ARCHITECTURE
# ---------------------------------
MODEL_PARAMS = {
    "conv_layers": [(32, 3)],  # List of (out_channels, kernel_size) tuples
    "n_encoder_layers": 1,  # Number of transformer encoder layers
    "d_input": N_CHANNEL,  # Input feature dimension
    "n_heads": 2,  # Number of attention heads
    "d_ff": 128,  # Feedforward network dimension
    "dropout": 0.01,  # Dropout rate
    "output_flatten": True,  # Must be True for RL agents
}

# ---------------------------------
# DATA COLLECTION
# ---------------------------------
BUFFER_SIZE = 100_000  # Maximum size of replay buffer
N_FRAMES_PER_BATCH = 128  # Number of frames to collect per batch

# ---------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------
BATCH_SIZE = 128  # Batch size for training
LR = 1e-3  # Learning rate (Adam/AdamW)
WEIGHT_DECAY = 1e-4  # Weight decay for optimizer

# Training Loop Configuration
TOTAL_FRAMES = 100_000  # Total training frames
WARMUP_FRAMES = 5_000  # Random exploration frames before training starts
OPTIMIZATION_STEPS = 10  # UTD Ratio: gradient updates per data collection
TAU = 0.005  # Soft update coefficient for target network (Polyak averaging)
GRAD_CLIP_NORM = 1.0  # Maximum norm for gradient clipping

# Logging and Evaluation
LOG_INTERVAL = 16  # Log and evaluate every N iterations
EVAL_GAMES = 10  # Number of games for evaluation against random policy
MCTS_ITERMAX = 100  # MCTS iterations for evaluation

# ---------------------------------
# ALGORITHM-SPECIFIC SETTINGS
# ---------------------------------
# SAC (Soft Actor-Critic) Settings
GAMMA = 0.99  # Discount factor for future rewards

# ---------------------------------
# CHECKPOINT SETTINGS
# ---------------------------------
CHECKPOINT_DIR = "d:/Code/Python/hex-irl/checkpoints"  # Directory for saving models
RESULTS_DIR = "d:/Code/Python/hex-irl/results"  # Directory for saving results/plots