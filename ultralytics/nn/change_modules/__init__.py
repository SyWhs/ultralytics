# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics modules.

This module provides access to various neural network components used in Ultralytics models, including convolution blocks,
attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron.
    >>> from ultralytics.nn.modules import *
    >>> import torch
    >>> import os
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
"""

from .CSNorm import (
    CSNorm,
)
from .CGLU import (
    C2f_CGLU,
)

__all__ = (
    "CSNorm",
    "C2f_CGLU",
)
