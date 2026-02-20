
import torch.nn as nn
import torch

from .vpct_util import VPCTModule

from compressai.models import CompressionModel, get_scale_table, ScaleHyperprior
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import deconv, conv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.ops import quantize_ste
from einops import rearrange

from compressai.ops import quantize_ste as ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder

class CheckerboardContext(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out

class reference_w_checkerboard_n_vpct(CompressionModel):
    def __init__(self, N, M):
        super().__init__()

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)
        self.context_prediction = CheckerboardContext(
            in_channels = M, out_channels = M * 2, kernel_size=5, stride=1, padding=2
        )
        # self.vpct_1 = VPCTModule(M, M*2)
        # self.vpct_2 = VPCTModule(M, M*2)
        self.vpct = VPCTModule(M, M*2)

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, M, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(M, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, M * 2),
        )

        self.hy_vpct = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 2, M, 1),
        )
        self.ctx_vpct = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 2, M, 1),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 18 // 3, N * 14 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 14 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, M * 6 // 3, 1),
        )

        self.M = M
        self.N = N

        # self.quantizer = Quantizer()

    def forward(self, y):
        v, b, c, h, w = y.size()
        y = rearrange(y, 'v b c h w -> (v b) c h w')
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        # y_hat_views = self.quantizer.quantize(y, "ste")

        hyper_params = self.h_s(z_hat)
        ctx_params = self.context_prediction(y_hat)

        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        low_dim_hyper = self.hy_vpct(hyper_params)
        low_dim_ctx_params = self.ctx_vpct(ctx_params)

        vpct_i_1 = rearrange(low_dim_hyper, '(v b) c h w -> v b c h w', v = v)
        vpct_j_1 = torch.concat(
            [vpct_i_1[0].unsqueeze(0), rearrange(y_hat, '(v b) c h w -> v b c h w', v = v)], 0
        )[:-1]

        vpct_i_2 = rearrange(low_dim_ctx_params, '(v b) c h w -> v b c h w', v = v)
        vpct_j_2 = torch.concat(
            [vpct_i_2[0].unsqueeze(0), rearrange(y_hat, '(v b) c h w -> v b c h w', v = v)], 0
        )[:-1]

        # vpct_params_anchor = self.vpct_1(
        #     vpct_i_1,
        #     vpct_j_1
        # )
        vpct_params_anchor = self.vpct(
            vpct_i_1,
            vpct_j_1
        )
        vpct_params_anchor = rearrange(vpct_params_anchor, 'v b c h w -> (v b) c h w')
        vpct_params_anchor[:, :, 0::2, 0::2] = 0
        vpct_params_anchor[:, :, 1::2, 1::2] = 0



        # vpct_params_non_anchor = self.vpct_2(
        #     vpct_i_2,
        #     vpct_j_2
        # )
        vpct_params_non_anchor = self.vpct(
            vpct_i_2,
            vpct_j_2
        )
        vpct_params_non_anchor = rearrange(vpct_params_non_anchor, 'v b c h w -> (v b) c h w')
        vpct_params_non_anchor[:, :, 0::2, 1::2] = 0
        vpct_params_non_anchor[:, :, 1::2, 0::2] = 0


        vpct_params = vpct_params_anchor + vpct_params_non_anchor

        
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params, vpct_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        
        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def validate(self, y):
        v, b, c, h, w = y.size()
        y = rearrange(y, 'v b c h w -> (v b) c h w')
        z = self.h_a(y)
        y = rearrange(y, '(v b) c h w -> v b c h w', v = v)
        z = rearrange(z, '(v b) c h w -> v b c h w', v = v)

        vpct_i_1 = torch.zeros_like(y)
        vpct_j_1 = torch.zeros_like(y)
        vpct_i_2 = torch.zeros_like(y)
        vpct_j_2 = torch.zeros_like(y)
        
        data_dict_list = []
        for vi in range(v):
            data_dict = self.val_one_vp(y[vi], z[vi], vi, vpct_i_1, vpct_j_1, vpct_i_2, vpct_j_2)
            data_dict_list.append(data_dict)
        
        return data_dict_list


    def val_one_vp(self, y, z, vp_pos, vpct_i_1, vpct_j_1, vpct_i_2, vpct_j_2):
        # y = self.g_a(x)
        # z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([y.size(0), y.size(1) * 2, y.size(2), y.size(3)], device=y.device)
        low_dim_hy_params = self.hy_vpct(hyper_params)
        if vp_pos == 0:
            vpct_i_1[vp_pos] = low_dim_hy_params.clone()
            vpct_j_1[vp_pos] = low_dim_hy_params.clone()
        else:
            vpct_i_1[vp_pos] = low_dim_hy_params.clone()
            
        # vpct_params_anchor = self.vpct_1(
        #     vpct_i_1,
        #     vpct_j_1
        # )[vp_pos]
        vpct_params_anchor = self.vpct(
            vpct_i_1,
            vpct_j_1
        )[vp_pos]

        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params, vpct_params_anchor], dim=1))
        # mask non-anchor
        gaussian_params_anchor[:, :, 0::2, 0::2] = 0
        gaussian_params_anchor[:, :, 1::2, 1::2] = 0
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = ste_round(y - means_anchor) + means_anchor
        ctx_params = self.context_prediction(anchor_hat)
        # mask anchor
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        ctx_params_ = ctx_params.clone()

        low_dim_ctx_params = self.ctx_vpct(ctx_params)

        if vp_pos == 0:
            vpct_i_2[vp_pos] = low_dim_ctx_params.clone()
            vpct_j_2[vp_pos] = low_dim_ctx_params.clone()
        else:
            vpct_i_2[vp_pos] = low_dim_ctx_params.clone()

        # vpct_params_non_ahchor = self.vpct_2(
        #     vpct_i_2,
        #     vpct_j_2
        # )[vp_pos]
        vpct_params_non_ahchor = self.vpct(
            vpct_i_2,
            vpct_j_2
        )[vp_pos]
        
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params, vpct_params_non_ahchor], dim=1))
        gaussian_params[:, :, 0::2, 1::2] = 0
        gaussian_params[:, :, 1::2, 0::2] = 0
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        means = means_anchor + means_hat
        scales = scales_anchor + scales_hat
        
        y_hat = ste_round(y - means) + means
        _, y_likelihoods = self.gaussian_conditional(y, scales, means=means)

        vpct_j_1[vp_pos] = y_hat.clone()
        vpct_j_2[vp_pos] = y_hat.clone()

        return {
            "y_hat": y_hat,
            "z_hat": z_hat,
            "vpct_params_anchor": vpct_params_anchor,
            "vpct_params_non_ahchor": vpct_params_non_ahchor,
            "gaussian_params": gaussian_params,
            "low_dim_hy_params": low_dim_hy_params,
            "anchor_hat": anchor_hat,
            "ctx_params": ctx_params,
            "ctx_params_": ctx_params_,
            "mean_hat": means_hat,
            "mean_anchor": means_anchor,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, y):

        # v, b, c, h, w = x.size()
        # x = rearrange(x, 'v b c h w -> (v b) c h w')
        # y = self.g_a(x)
        v, b, c, h, w = y.size()
        y = rearrange(y, 'v b c h w -> (v b) c h w')
        z = self.h_a(y)

        y = rearrange(y, '(v b) c h w -> v b c h w', v = v)
        z = rearrange(z, '(v b) c h w -> v b c h w', v = v)
        
        vp_dict_list = []

        vpct_i_1 = torch.zeros_like(y)
        vpct_j_1 = torch.zeros_like(y)
        vpct_i_2 = torch.zeros_like(y)
        vpct_j_2 = torch.zeros_like(y)
        

        for vi in range(v):
            vp_dict = self.compress_one_vp(y[vi], z[vi], vi, vpct_i_1, vpct_j_1, vpct_i_2, vpct_j_2)
            vp_dict_list.append(vp_dict)
        
        return vp_dict_list
        
    def compress_one_vp(self, y, z, vp_pos, vpct_i_1, vpct_j_1, vpct_i_2, vpct_j_2):

        torch.backends.cudnn.deterministic = True
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        # y = self.g_a(x)
        # z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([y.size(0), y.size(1) * 2, y.size(2), y.size(3)], device=y.device)
        low_dim_hy_params = self.hy_vpct(hyper_params)
        if vp_pos == 0:
            vpct_i_1[vp_pos] = low_dim_hy_params.clone()
            vpct_j_1[vp_pos] = low_dim_hy_params.clone()
        else:
            vpct_i_1[vp_pos] = low_dim_hy_params.clone()
            
        # vpct_params_anchor = self.vpct_1(
        #     vpct_i_1,
        #     vpct_j_1
        # )[vp_pos]
        vpct_params_anchor = self.vpct(
            vpct_i_1,
            vpct_j_1
        )[vp_pos]

        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params, vpct_params_anchor], dim=1))

        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.compress_anchor(y, scales_anchor, means_anchor, symbols_list, indexes_list)
        
        ctx_params = self.context_prediction(anchor_hat)
        # mask anchor
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        low_dim_ctx_params = self.ctx_vpct(ctx_params)
        if vp_pos == 0:
            vpct_i_2[vp_pos] = low_dim_ctx_params.clone()
            vpct_j_2[vp_pos] = low_dim_ctx_params.clone()
        else:
            vpct_i_2[vp_pos] = low_dim_ctx_params.clone()

        # vpct_params_non_ahchor = self.vpct_2(
        #     vpct_i_2,
        #     vpct_j_2
        # )[vp_pos]
        vpct_params_non_ahchor = self.vpct(
            vpct_i_2,
            vpct_j_2
        )[vp_pos]

        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params, vpct_params_non_ahchor], dim=1))

        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self.compress_nonanchor(y, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)

        vpct_j_1[vp_pos] = (anchor_hat+nonanchor_hat).clone()
        vpct_j_2[vp_pos] = (anchor_hat+nonanchor_hat).clone()
        

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "y_hat": anchor_hat+nonanchor_hat, 
            "z_hat": z_hat,
            'vpct_params_anchor': vpct_params_anchor,
            "vpct_params_non_ahchor": vpct_params_non_ahchor,
            "gaussian_params": gaussian_params,
            "low_dim_hy_params": low_dim_hy_params,
            "ctx_params": ctx_params,
            "anchor_hat": anchor_hat,
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, data_dict_list, vpct_size, cuda=True):
        if cuda:
            vpct_i_1 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1).cuda()
            vpct_j_1 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1).cuda()
            vpct_i_2 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1).cuda()
            vpct_j_2 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1).cuda()
        else:
            vpct_i_1 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1)
            vpct_j_1 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1)
            vpct_i_2 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1)
            vpct_j_2 = torch.zeros(vpct_size[1]).unsqueeze(0).repeat(vpct_size[0], 1, 1, 1, 1)

        out_dict_list = []
        
        # for data_dict in data_dict_list:
        for vp_pos in range(len(data_dict_list)):
            out_dict = self.decompress_one_vp(data_dict_list[vp_pos], vp_pos, vpct_i_1, vpct_j_1, vpct_i_2, vpct_j_2, cuda=cuda)
            out_dict_list.append(out_dict)

        return out_dict_list

    def decompress_one_vp(self, data_dict, vp_pos, vpct_i_1, vpct_j_1, vpct_i_2, vpct_j_2, cuda=True):
        strings = data_dict['strings']
        shape = data_dict['shape']
        torch.backends.cudnn.deterministic = True

        y_strings = strings[0][0]
        z_strings = strings[1]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        if cuda:
            z_hat = self.entropy_bottleneck.decompress(z_strings, shape).cuda()
        else:
            z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([z_hat.size(0), self.M * 2, z_hat.size(2) * 4, z_hat.size(3) * 4], device=z_hat.device)
        low_dim_hy_params = self.hy_vpct(hyper_params)

        if vp_pos == 0:
            vpct_i_1[vp_pos] = low_dim_hy_params.clone()
            vpct_j_1[vp_pos] = low_dim_hy_params.clone()
        else:
            vpct_i_1[vp_pos] = low_dim_hy_params.clone()
        
        # vpct_params_anchor = self.vpct_1(
        #     vpct_i_1,
        #     vpct_j_1,
        # )[vp_pos]
        vpct_params_anchor = self.vpct(
            vpct_i_1,
            vpct_j_1,
        )[vp_pos]

        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params, vpct_params_anchor], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.decompress_anchor(scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
        
        ctx_params = self.context_prediction(anchor_hat)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        low_dim_ctx_params = self.ctx_vpct(ctx_params)
        if vp_pos == 0:
            vpct_i_2[vp_pos] = low_dim_ctx_params.clone()
            vpct_j_2[vp_pos] = low_dim_ctx_params.clone()
        else:
            vpct_i_2[vp_pos] = low_dim_ctx_params.clone()

        # vpct_params_non_ahchor = self.vpct_2(
        #     vpct_i_2, 
        #     vpct_j_2, 
        # )[vp_pos]
        vpct_params_non_ahchor = self.vpct(
            vpct_i_2, 
            vpct_j_2, 
        )[vp_pos]

        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params, vpct_params_non_ahchor], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self.decompress_nonanchor(scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)

        y_hat = anchor_hat + nonanchor_hat
        vpct_j_1[vp_pos] = y_hat.clone()
        vpct_j_2[vp_pos] = y_hat.clone()
        
        # x_hat = self.g_s(y_hat)


        # data_dict['x_hat'] = x_hat
        data_dict['d_y_hat'] = y_hat
        

        return data_dict

    def ckbd_anchor_sequeeze(self, y):
        B, C, H, W = y.shape
        anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        return anchor

    def ckbd_nonanchor_sequeeze(self, y):
        B, C, H, W = y.shape
        nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
        return nonanchor

    def ckbd_anchor_unsequeeze(self, anchor):
        B, C, H, W = anchor.shape
        y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
        y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
        y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
        return y_anchor

    def ckbd_nonanchor_unsequeeze(self, nonanchor):
        B, C, H, W = nonanchor.shape
        y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
        y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
        y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
        return y_nonanchor

    def compress_anchor(self, anchor, scales_anchor, means_anchor, symbols_list, indexes_list):
        # squeeze anchor to avoid non-anchor symbols
        anchor_squeeze = self.ckbd_anchor_sequeeze(anchor)
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = self.gaussian_conditional.quantize(anchor_squeeze, "symbols", means_anchor_squeeze)
        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
        return anchor_hat

    def compress_nonanchor(self, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list):
        nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(nonanchor)
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = self.gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
        return nonanchor_hat

    def decompress_anchor(self, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets):
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(scales_anchor.device) + means_anchor_squeeze
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat)
        return anchor_hat

    def decompress_nonanchor(self, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets):
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(scales_nonanchor.device) + means_nonanchor_squeeze
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat)
        return nonanchor_hat



