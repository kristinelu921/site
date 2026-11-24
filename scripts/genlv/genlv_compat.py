"""Inference-only import shims for the pinned GenLV source.

GenLV imports two BasicSR helpers.  The released checkpoint only depends on
their module/key layout, so keeping the tiny definitions here avoids pulling
the rest of BasicSR's CUDA-oriented training stack into the TPU environment.
"""
from __future__ import annotations

import sys
import types


def install() -> None:
    import torch.nn as nn

    class ResidualBlockNoBN(nn.Module):
        def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
            super().__init__()
            self.res_scale = res_scale
            self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, value):
            residual = self.conv2(self.relu(self.conv1(value)))
            return value + residual * self.res_scale

    def make_layer(basic_block, num_basic_block, **kwargs):
        return nn.Sequential(*(basic_block(**kwargs) for _ in range(num_basic_block)))

    basicsr = types.ModuleType("basicsr")
    archs = types.ModuleType("basicsr.archs")
    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.ResidualBlockNoBN = ResidualBlockNoBN
    arch_util.make_layer = make_layer
    basicsr.archs = archs
    archs.arch_util = arch_util
    sys.modules.update(
        {
            "basicsr": basicsr,
            "basicsr.archs": archs,
            "basicsr.archs.arch_util": arch_util,
        }
    )

