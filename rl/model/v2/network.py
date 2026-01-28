import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from rl.model.v2.submodules import SkipLayerBias


class Conv(nn.Module):
    '''
    model consists of a convolutional layer to change the number of channels from two input channels to intermediate channels
    then a specified amount of residual or skip-layers https://en.wikipedia.org/wiki/Residual_neural_network
    then policyconv reduce the intermediate channels to one
    value range is (-inf, inf) 
    for training the sigmoid is taken, interpretable as probability to win the game when making this move
    for data generation and evaluation the softmax is taken to select a move
    '''
    def __init__(
            self, 
            board_size: int, 
            conv_layers: list[tuple[int, int]], 
            d_input: int = 2, 
            dropout: float = 0.1,
            output_flatten: bool = True,
            **kwargs):
        """
        Args:
            board_size: Size of the board (e.g., 11 for 11x11).
            conv_layers: List of tuples (out_channels, kernel_size) for each conv layer.
                Note that, in_channels is inferred from the previous layer's out_channels (d_input for the first layer).
            d_input: Number of input channels (default: 2 for player positions).
        """
        super(Conv, self).__init__()
        self.board_size = board_size
        self.d_input = d_input
        self.output_flatten = output_flatten
        
        # First conv layer with appropriate padding
        first_channels, first_kernel = conv_layers[0]
        first_padding = max((first_kernel - 1) // 2 - 1, 0)
        self.conv = nn.Conv2d(d_input, first_channels, kernel_size=first_kernel, padding=first_padding)
        
        # Skip layers - similar structure to HexModel but using SkipLayerBias
        self.skip_layers = nn.Sequential(*[
            SkipLayerBias(
                n_channels=conv_layers[i-1][0] if i > 0 else first_channels,
                padding=(conv_layers[i][1] - 1) // 2  # Convert kernel_size to padding
            )
            for i in range(len(conv_layers))
        ])
        
        # Policy conv layer
        last_channels, last_kernel = conv_layers[-1]
        last_padding = (last_kernel - 1) // 2
        self.activation = nn.SiLU()
        self.projection = nn.Conv2d(last_channels, 1, kernel_size=last_kernel, padding=last_padding, bias=False)
        # self.bias = nn.Parameter(torch.zeros(board_size ** 2))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (N, d_input, H, W) where
               N = batch size, H = height, W = width.
   x     Returns:
            Tensor of shape (N, H*W) with logits for each position.
        """
        x = self.activation(self.conv(x))
        x = self.skip_layers(x)
        x = self.projection(x).view(-1, self.board_size ** 2)
        if self.output_flatten:
            return x
        else:
            return x.view(-1, self.board_size, self.board_size)


class RotationWrapperModel(nn.Module):
    '''
    evaluates input and its 180° rotation with parent model
    averages both predictions
    '''
    def __init__(self, model):
        super(RotationWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.model = model

    def forward(self, x):
        x_flip = torch.flip(x, [2, 3])
        y_flip = self.model(x_flip)
        y = torch.flip(y_flip, [1])
        return (self.model(x) + y)/2