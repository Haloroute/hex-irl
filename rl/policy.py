import torch

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor


class HexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(HexConv2d, self).__init__()
        assert kernel_size % 2 == 1 and kernel_size > 0, "kernel_size must be odd and positive."
        stride, padding = 1, kernel_size // 2  # To maintain spatial dimensions

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask = nn.Parameter(self._create_hex_mask(kernel_size), requires_grad=False) # (k, k), requires_grad=False to keep it fixed
        with torch.no_grad():
            self.conv.weight.mul_(self.mask)  # Apply mask to weights

    @staticmethod
    def _create_hex_mask(kernel_size: int) -> Tensor:
        assert kernel_size % 2 == 1 and kernel_size > 0, "kernel_size must be odd and positive."

        mask = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        center = kernel_size // 2

        for r in range(kernel_size): # Row index
            for c in range(kernel_size): # Column index
                # Using axial distance for a vertically oriented hex grid
                # mapped to an offset coordinate system in the kernel.
                # (r, c) are kernel indices, (dr, dc) are relative to center.
                dr, dc = r - center, c - center
                chebyshev_distance = max(abs(dr), abs(dc), abs(dr + dc))

                if chebyshev_distance <= center: # Inside or on the hexagon
                    mask[r, c] = 1.0

        return mask # (kernel_size, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        # Assuming x is of shape (batch_size, channels, height, width)
        # Apply convolution
        x = self.conv(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, adjust_input: nn.Module, original_input: nn.Module = nn.Identity()):
        super(SkipConnection, self).__init__()
        self.adjust_input = adjust_input
        self.original_input = original_input

    def forward(self, x: Tensor) -> Tensor:
        return self.adjust_input(x) + self.original_input(x)


class TransformerQL(nn.Module):
    def __init__(self, 
                 conv_layers: list[tuple[int, int]],
                 n_encoder_layers: int,
                 d_input: int,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 output_flatten: bool = True):
        """Args:
            conv_layers: List of tuples (out_channels, kernel_size) for each conv layer.
                Note that, in_channels is inferred from the previous layer's out_channels (d_input for the first layer).
            n_encoder_layers: Number of transformer encoder layers.
            d_input: Dimension of input features to the transformer.
            n_heads: Number of attention heads in the transformer.
            d_ff: Dimension of the feedforward network in the transformer.
            dropout: Dropout rate.
        """
        super(TransformerQL, self).__init__()
        self.output_flatten = output_flatten
        self.d_encoder: int = conv_layers[-1][0] # Last conv layer's out_channels as d_model
        self.conv = nn.Sequential(*[
            SkipConnection(
                nn.Sequential(
                    HexConv2d(conv_layers[i-1][0] if i > 0 else d_input, conv_layers[i][0], conv_layers[i][1]),
                    nn.BatchNorm2d(conv_layers[i][0]),
                    nn.GELU(),
                    HexConv2d(conv_layers[i][0], conv_layers[i][0], conv_layers[i][1]),
                    nn.BatchNorm2d(conv_layers[i][0]),
                ),
                nn.Identity() if conv_layers[i][0] == conv_layers[i-1][0] # Skip connection (identity)
                else HexConv2d(conv_layers[i-1][0] if i > 0 else d_input, conv_layers[i][0], 1) # Combine with Conv2d (kernel_size = 1) for channel adjustment
            )
            for i in range(len(conv_layers))
        ])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_encoder,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=n_encoder_layers
        )
        self.projection = nn.Linear(self.d_encoder, 1)

    def forward(self, x: Tensor, tensordict: TensorDict | None = None) -> Tensor:
        """Args:
            x: Input tensor of shape (N, H, W, C) where
               N = batch size, H = height, W = width, C = channels (d_input).
            Returns: Tensor of shape (N, H, W) with Q-values for each position.
        """
        # Reshape input to (N, C, H, W) for Conv2d
        batch_size, height, width = x.size(0), x.size(1), x.size(2)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, C, H, W)
        x = self.conv(x) # (N, d_encoder, H, W)
        # Reshape to (N, H*W, d_encoder)
        x = x.view(batch_size, -1, self.d_encoder) # (N, H*W, d_encoder)
        x = self.encoder(x) # (N, H*W, d_encoder)
        x = self.projection(x) # (N, H*W, 1)
        if self.output_flatten:
            return x.view(batch_size, -1) # (N, H*W)
        else:
            return x.view(batch_size, height, width) # (N, H, W)


class MaskWrapper(nn.Module):
    """
    A custom policy for Hex that wraps a DQN.
    - If it's Player 0's turn, it maximizes the Q-value.
    - If it's Player 1's turn, it minimizes the Q-value.
    It always respects the action mask.
    """
    def __init__(self, dqn_network: nn.Module, turn_wrapper: bool = True):
        super().__init__()
        self.dqn_network = dqn_network
        self.turn_wrapper = turn_wrapper

    def forward(self, x: Tensor, tensordict: TensorDict | None = None) -> Tensor:
        # x: (N, H, W, C)
        if x.dim() == 3:
            x = x.unsqueeze(0) # Add batch dimension if missing
        batch_size = x.size(0)
        # Step 1: Get the raw Q-values from your network
        mask = ~(x[..., 0].bool() | x[..., 1].bool() | ~x[..., -1].bool()).view(batch_size, -1) # Assuming mask is the sum of red and blue channels
        q_values: Tensor = self.dqn_network(x) # (N, num_actions)

        # If turn_wrapper is disabled, just apply the mask and return
        if not self.turn_wrapper:
            # If turn_wrapper is disabled, just apply the mask and return
            q_values[~mask] = -torch.inf
            return q_values
        else:
            # Step 2: Determine the current player and adjust Q-values accordingly.
            # We assume the 3rd channel (index 2) of the observation indicates the player.
            # Player 0: channel is all 0s. Player 1: channel is all 1s.
            current_player: float = x[..., 2].any().item() # Will be 0.0 or 1.0

            if current_player == 1.0:
                # Player 1 (blue) wants to MINIMIZE the Q-value.
                # This is equivalent to MAXIMIZING the negative Q-value.
                effective_q_values = -q_values
            else:
                # Player 0 (red) wants to MAXIMIZE the Q-value.
                effective_q_values = q_values

            # Step 3: Apply the action mask. This is crucial.
            # Set the Q-value of all illegal moves to negative infinity.
            effective_q_values[~mask] = -torch.inf

            return effective_q_values