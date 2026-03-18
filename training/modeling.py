import torch
import torch.nn as nn

from compressai.zoo import image_models

from .models import VPCTCheng2020Attention


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access wrapped module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class ViewportForwardAdapter(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, x):
        target = x
        if x.dim() == 5:
            batch_size, num_viewports, channels, height, width = x.shape
            target = x.view(batch_size * num_viewports, channels, height, width)
            out = self.codec(target)
        else:
            out = self.codec(x)
            target = x

        out["target"] = target
        return out

    def aux_loss(self):
        return self.codec.aux_loss()


def _build_vpct_cheng2020_attn(quality, vpct_layers):
    pretrained_model = image_models["cheng2020-attn"](quality=quality, pretrained=True)
    channel_count = getattr(pretrained_model, "N", 192)

    model = VPCTCheng2020Attention(N=channel_count, vpct_layers=vpct_layers)
    pretrained_state = pretrained_model.state_dict()

    missing_keys, unexpected_keys = model.load_state_dict(pretrained_state, strict=False)

    print(
        "Initialized vpct-cheng2020-attn from CompressAI pretrained checkpoint. "
        f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
    )
    if missing_keys:
        print(f"Missing keys (kept random init): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {unexpected_keys}")

    return model


def build_model(args, device):
    if args.model == "vpct-cheng2020-attn":
        model = _build_vpct_cheng2020_attn(args.quality, args.vpct_layers)
    else:
        codec = image_models[args.model](quality=args.quality, pretrained=True)
        model = ViewportForwardAdapter(codec)

    frozen_params = freeze_ga_gs(model)
    print(f"Froze g_a/g_s parameters: {frozen_params}")

    model = model.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)

    return model


def freeze_ga_gs(model):
    module = model.module if isinstance(model, nn.DataParallel) else model
    frozen = 0
    for attr_name in ("g_a", "g_s"):
        if hasattr(module, attr_name):
            submodule = getattr(module, attr_name)
            for param in submodule.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
    return frozen


def configure_optimizers(net, args):
    params_dict = dict(net.named_parameters())
    net_param_names = sorted(
        name
        for name, param in params_dict.items()
        if param.requires_grad and not name.endswith(".quantiles")
    )
    aux_param_names = sorted(
        name
        for name, param in params_dict.items()
        if param.requires_grad and name.endswith(".quantiles")
    )

    if len(net_param_names) == 0:
        raise ValueError("No trainable net parameters found for optimizer.")

    net_optimizer = torch.optim.Adam(
        (params_dict[name] for name in net_param_names),
        lr=args.learning_rate,
    )

    if len(aux_param_names) == 0:
        raise ValueError("No trainable aux parameters found for optimizer.")

    aux_optimizer = torch.optim.Adam(
        (params_dict[name] for name in aux_param_names),
        lr=args.aux_learning_rate,
    )

    return net_optimizer, aux_optimizer
