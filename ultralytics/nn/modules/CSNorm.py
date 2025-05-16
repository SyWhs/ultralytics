import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "CSNorm",
)

class Generate_gate(nn.Module):
    """Generate gate module."""
    #门控模块
    def __init__(self, c1, c2):
        super(Generate_gate, self).__init__()
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(c1,c1//2, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(c1//2,c2, 1),
                                  nn.ReLU())

        self.epsilon = 1e-8

    def forward(self, x):


        alpha = self.proj(x)
        gate = (alpha**2) / (alpha**2 + self.epsilon)

        return gate

class CSNorm(nn.Module):
    """CSP Normalization."""

    def __init__(self, c1, c2):
        """
        Initialize CSP Normalization.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.gate = Generate_gate(c1,c2)
        # self.inst_norm = nn.ModuleList([nn.InstanceNorm2d(1) for _ in range(c1)])
        # 使用GroupNorm替代逐通道InstanceNorm
        self.group_norm = nn.GroupNorm(num_groups=c1, num_channels=c1, affine=True)


    def forward(self, x):
        """Forward pass through the CSP normalization layer."""
        gate = self.gate(x)  # 生成通道注意力
        # 逐通道归一化
        # norm_feat = torch.cat([self.inst_norm[i](x[:,i:i+1]) for i in range(x.size(1))], dim=1)
        norm_feat = self.group_norm(x)  # 并行处理所有通道
        return gate * norm_feat + (1 - gate) * x