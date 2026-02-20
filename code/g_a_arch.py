import torch.nn as nn


from code.vpct_util import GSDN

from compressai.models import CompressionModel, get_scale_table, ScaleHyperprior, Cheng2020Anchor
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN
)
from compressai.models.utils import deconv, conv


class ResidualBlockWithStrideGSDN(ResidualBlockWithStride):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__(in_ch, out_ch, stride)
        self.gdn = GSDN(out_ch)

class g_a_reference(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.g_a = nn.Sequential(
            ResidualBlockWithStrideGSDN(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideGSDN(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideGSDN(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )
    def forward(self, x):
        return self.g_a(x)