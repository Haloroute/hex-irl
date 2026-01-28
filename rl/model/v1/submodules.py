import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class HexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, bias: bool = True):
        super(HexConv2d, self).__init__()
        assert kernel_size % 2 == 1 and kernel_size > 0, "kernel_size must be odd and positive."

        stride, padding = 1, kernel_size // 2  # To maintain spatial dimensions
        mask = self._create_hex_mask(kernel_size)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.register_buffer('mask', mask) # (k, k), requires_grad=False to keep it fixed

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
        # 1. Tạo trọng số đã mask (Soft masking)
        # Phép nhân '*' tạo ra tensor mới, KHÔNG sửa in-place trọng số gốc.
        # Gradient vẫn truyền ngược qua đây bình thường, các vị trí mask=0 sẽ có grad=0.
        masked_weight = self.conv.weight * self.mask
        
        # 2. Dùng functional conv2d thay vì self.conv(x)
        # Chúng ta truyền masked_weight vào đây.
        x = F.conv2d(
            input=x,
            weight=masked_weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )
        return x


class SkipConnection(nn.Module):
    def __init__(self, adjust_input: nn.Module, original_input: nn.Module = nn.Identity()):
        super(SkipConnection, self).__init__()
        self.adjust_input = adjust_input
        self.original_input = original_input

    def forward(self, x: Tensor) -> Tensor:
        return self.adjust_input(x) + self.original_input(x)


class TriAxialPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int):
        """
        Args:
            d_model: Kích thước channel của input (C). Phải là số chẵn.
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model (C) phải là số chẵn để tính sin/cos."

        # Pre-compute div_term cho sinusoidal
        # Lưu ý: arange(0, d_model, 2) tạo ra d_model/2 phần tử
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        self.d_model = d_model
        self.register_buffer('div_term', div_term)

    def _get_1d_sinusoidal(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Tạo sinusoidal embedding cho một trục toạ độ.
        Args:
            coords: Tensor chứa giá trị toạ độ (số nguyên hoặc thực), shape (H, W)
        Returns:
            Tensor embedding, shape (H, W, d_model)
        """
        # Mở rộng chiều cuối để tính toán: (H, W, 1)
        coords = coords.unsqueeze(-1).float()
        
        # div_term có shape (d_model/2,)
        # phase = coords * div_term -> Shape (H, W, d_model/2)
        phase = coords * self.div_term
        
        # Tính sin và cos riêng biệt
        sin_part = torch.sin(phase) # (H, W, d_model/2)
        cos_part = torch.cos(phase) # (H, W, d_model/2)
        
        # --- FIX LỖI VMAP Ở ĐÂY ---
        # Thay vì gán in-place (pe[..., 0::2] = ...), ta dùng stack và flatten.
        # 1. Stack lại ở chiều cuối cùng: (H, W, d_model/2, 2)
        #    Tại vị trí cuối: [sin, cos], [sin, cos], ...
        val = torch.stack([sin_part, cos_part], dim=-1)
        
        # 2. Flatten 2 chiều cuối để trộn lại thành (H, W, d_model)
        #    Kết quả sẽ là: sin, cos, sin, cos... đúng thứ tự chẵn lẻ
        pe = val.flatten(-2, -1)
        
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (Giữ nguyên phần logic forward cũ) ...
        N, H, W, C = x.shape
        assert C == self.d_model, f"Input channel {C} không khớp với d_model khởi tạo {self.d_model}"

        device = x.device

        rows = torch.arange(H, device=device, dtype=torch.float)
        cols = torch.arange(W, device=device, dtype=torch.float)
        
        r_grid, q_grid = torch.meshgrid(rows, cols, indexing='ij')

        s_grid = -q_grid - r_grid

        pe_q = self._get_1d_sinusoidal(q_grid)
        pe_r = self._get_1d_sinusoidal(r_grid)
        pe_s = self._get_1d_sinusoidal(s_grid)

        full_pe = (pe_q + pe_r + pe_s) / torch.sqrt(torch.tensor(3.0))

        return full_pe.unsqueeze(0)