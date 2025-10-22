# Define constants, configurations, or settings for the project here.
# This file can include paths, hyperparameters, or any other settings
# that need to be accessed globally across different modules.

import torch

# ---------------------------------
# DEVICE CONFIGURATION
# ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# ENVIRONMENT
# --------------------------------
BOARD_SIZES = [2, 2, 2]  # Different sizes of the Hex board to be used
MAX_BOARD_SIZE = 4  # Maximum size of the Hex board
SWAP_RULE = True  # Whether to use the swap rule in the Hex game

# --------------------------------
# POLICY
# --------------------------------
N_INPUT_CHANNELS = 4  # Number of input channels for the neural network. 4 means (red stones, blue stones, current player, valid move mask)
CONV_LAYERS = [(32, 3), (64, 3), (128, 3)]  # List of tuples (n_out_channels, kernel_size) for convolutional layers
N_ENCODER_LAYERS = 4  # Number of transformer encoder layers
N_HEADS = 8  # Number of attention heads in the transformer
D_FF = 2048  # Dimension of feedforward network in the transformer
DROPOUT = 0.1  # Dropout rate for the neural network
OUTPUT_FLATTEN = True  # Whether to flatten the output of the policy network. MUST BE TRUE FOR RL AGENTS.

# --------------------------------
# DATA COLLECTOR
# --------------------------------
BUFFER_MAX_SIZE = 2 ** 10  # Maximum size of the replay buffer
FRAMES_PER_BATCH = 64  # Number of frames to collect per batch
TOTAL_FRAMES = 2 ** 10  # Total number of frames to collect

# --------------------------------
# TRAINING HYPERPARAMETERS
# --------------------------------
BATCH_SIZE = 64  # Batch size for training
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
GAMMA = 0.99  # Discount factor for future rewards