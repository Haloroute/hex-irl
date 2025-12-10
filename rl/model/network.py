import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from rl.model.submodules import HexConv2d, SkipConnection, TriAxialPositionalEmbedding


class HexModel(nn.Module):
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
        super(HexModel, self).__init__()
        self.output_flatten = output_flatten
        self.d_encoder: int = conv_layers[-1][0] # Last conv layer's out_channels as d_model
        self.conv = nn.Sequential(*[
            SkipConnection(
                nn.Sequential(
                    HexConv2d(conv_layers[i-1][0] if i > 0 else d_input, conv_layers[i][0], conv_layers[i][1], bias=False),
                    nn.GroupNorm(num_groups=4, num_channels=conv_layers[i][0]),
                    nn.GELU(),
                    HexConv2d(conv_layers[i][0], conv_layers[i][0], conv_layers[i][1], bias=False),
                    nn.GroupNorm(num_groups=4, num_channels=conv_layers[i][0]),
                ),
                nn.Identity() if (i > 0 and conv_layers[i][0] == conv_layers[i-1][0]) # Skip connection (identity)
                else nn.Sequential(
                    nn.Conv2d(conv_layers[i-1][0] if i > 0 else d_input, conv_layers[i][0], 1), # Combine with Conv2d (kernel_size = 1) for channel adjustment
                    nn.GroupNorm(num_groups=4, num_channels=conv_layers[i][0]),
                )
            )
            for i in range(len(conv_layers))
        ])
        self.positional_embedding = TriAxialPositionalEmbedding(self.d_encoder)
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
        self.projection = nn.Linear(self.d_encoder, 1) # Đầu ra cho Actor (logits)/Critic (Q-value)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: Input tensor of shape (N, H, W, C) where
               N = batch size, H = height, W = width, C = channels (d_input).
            Returns: Tensor of shape (N, H, W) with Q-values for each position.
        """
        # Reshape input to (N, C, H, W) for Conv2d
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif len(x.shape) != 4:
            raise ValueError(f"Input tensor x must have shape (N, H, W, C) or (H, W, C), but got {x.shape}.")

        # 1. Convolutional layers
        batch_size, height, width = x.size(0), x.size(1), x.size(2)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, C, H, W)
        x = self.conv(x) # (N, d_encoder, H, W)

        # 2. Positional Embedding + Transformer Encoder
        # x = x.permute(0, 2, 3, 1).flatten(1, 2).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        pe: Tensor = self.positional_embedding(x)
        x = (x + pe).flatten(1, 2).contiguous() # (N, H*W, d_encoder)
        x = self.encoder(x) # (N, H*W, d_encoder)

        # Chỉ sử dụng khi sử dụng vmap của DiscreteSACLOss (deactivate_vmap=False)
        # Nếu không dùng vmap thì không cần thiết (do giảm hiệu suất).
        # with sdpa_kernel(SDPBackend.MATH):
        #     x = self.encoder(x) # (N, H*W, d_encoder)

        # 3. Projection to create outputs for Actor/Critic
        x = self.projection(x) # (N, H*W, 1)
        if self.output_flatten:
            return x.view(batch_size, -1) # (N, H*W)
        else:
            return x.view(batch_size, height, width) # (N, H, W)