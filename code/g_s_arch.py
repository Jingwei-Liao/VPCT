import torch.nn as nn

from code.vpct_util import GSDN

from compressai.models import CompressionModel, get_scale_table, ScaleHyperprior
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN
)
from compressai.models.utils import deconv




class ResidualBlockUpsampleGSDN(ResidualBlockUpsample):
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__(in_ch, out_ch, upsample)
        self.igdn = GSDN(out_ch, inverse=True)



class g_s_reference(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.g_s = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsampleGSDN(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsampleGSDN(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsampleGSDN(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
    
    def forward(self, x):
        return self.g_s(x)
