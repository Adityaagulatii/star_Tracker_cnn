"""
LSQ (Learned Step Size Quantization) layers.
Reference: Esser et al., "Learned Step Size Quantization," ICLR 2020.

Each Conv2d gets two learnable scalars:
  alpha_w  – weight quantization step size
  alpha_x  – activation quantization step size

Gradient flows through the round via STE (straight-through estimator).
alpha_w and alpha_x receive real gradients scaled by the LSQ grad factor.
"""

import torch
import torch.nn as nn
import math


def lsq_quantize(x: torch.Tensor, alpha: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Quantize x to n_bits using learned step size alpha (STE)."""
    Qn = -(2 ** (n_bits - 1))
    Qp =  (2 ** (n_bits - 1)) - 1
    a  = alpha.abs()
    # scale → clamp → round (STE: detach the discontinuity)
    x_scaled = x / a
    x_clamp  = x_scaled.clamp(Qn, Qp)
    x_round  = x_clamp + (x_clamp.round() - x_clamp).detach()
    return x_round * a


class LSQConv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d with LSQ on both weights and activations.
    Preserves all Conv2d constructor arguments.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

        # Learnable step sizes — initialized after first forward
        self.alpha_w = nn.Parameter(torch.ones(1))
        self.alpha_x = nn.Parameter(torch.ones(1))
        self._initialized = False

    def _init_scales(self, x: torch.Tensor):
        """Initialize alpha from tensor statistics (2σ / Qp)."""
        Qp = 2 ** (self.n_bits - 1) - 1
        with torch.no_grad():
            self.alpha_w.data.fill_(2 * self.conv.weight.std().item() / math.sqrt(Qp))
            self.alpha_x.data.fill_(2 * x.std().item() / math.sqrt(Qp))
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._init_scales(x)

        w_q = lsq_quantize(self.conv.weight, self.alpha_w, self.n_bits)
        x_q = lsq_quantize(x,               self.alpha_x, self.n_bits)

        return nn.functional.conv2d(
            x_q, w_q, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups
        )


def replace_conv2d_with_lsq(module: nn.Module, n_bits: int = 8) -> nn.Module:
    """
    Recursively replace all nn.Conv2d layers in `module` with LSQConv2d.
    Copies pretrained weights and bias in-place.
    Returns the modified module.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            lsq = LSQConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding,
                dilation=child.dilation, groups=child.groups,
                bias=(child.bias is not None), n_bits=n_bits
            )
            lsq.conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lsq.conv.bias.data.copy_(child.bias.data)
            setattr(module, name, lsq)
        else:
            replace_conv2d_with_lsq(child, n_bits)
    return module
