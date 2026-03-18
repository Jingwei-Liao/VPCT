import torch.nn as nn
from compressai.models import Cheng2020Attention
import torch

from ..modules import VPCTModule


class VPCTCheng2020Attention(Cheng2020Attention):
    def __init__(self, N=192, vpct_layers=1, **kwargs):
        super().__init__(N=N, **kwargs)
        M = N
        self.vpct_adapter = VPCTModule(channels=N, num_heads=8, num_layers=vpct_layers)
        self.entropy_parameters_scale = nn.Sequential(
            nn.Conv2d(M * 6 // 3, M * 5 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 5 // 3, M * 4 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 4 // 3, M * 3 // 3, 1),
        )
        self.entropy_parameters_mean = nn.Sequential(
            nn.Conv2d(M * 6 // 3, M * 5 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 5 // 3, M * 4 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 4 // 3, M * 3 // 3, 1),
        )
        
        

    def forward(self, x):
        target = x
        x_hat_view = None
        batch_size = None
        num_viewports = None

        if x.dim() == 5:
            batch_size, num_viewports, channels, height, width = x.shape
            target = x.view(batch_size * num_viewports, channels, height, width)
            x = target

        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        if batch_size is not None:
            y_channels, y_height, y_width = y_hat.shape[1:]
            y_hat = y_hat.view(batch_size, num_viewports, y_channels, y_height, y_width)
            global_info = self.vpct_adapter(y_hat)
            global_info = global_info.view(batch_size * num_viewports, y_channels, y_height, y_width)
            y_hat = y_hat.view(batch_size * num_viewports, y_channels, y_height, y_width)

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat = self.entropy_parameters_scale(torch.cat((scales_hat, global_info), dim=1)) + scales_hat
        means_hat = self.entropy_parameters_mean(torch.cat((means_hat, global_info), dim=1)) + means_hat
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        if target.dim() == 4 and x.dim() == 4 and x_hat.size(0) != target.size(0):
            raise RuntimeError("x_hat and target batch sizes are inconsistent.")

        if batch_size is not None:
            x_hat_view = x_hat.view(batch_size, num_viewports, channels, height, width)

        return {
            "x_hat": x_hat,
            "x_hat_view": x_hat_view,
            "target": target,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

