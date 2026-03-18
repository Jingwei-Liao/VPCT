import argparse
import os
from contextlib import nullcontext

import torch
from compressai.zoo import image_models
from torch.utils.data import DataLoader

from training.args import CUSTOM_MODELS
from training.datasets import ERPViewportDataset
from training.engine import compute_bpi_from_likelihoods, compute_vp_psnr_mean
from training.modeling import build_model


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def update(self, value, n=1):
        self.sum += float(value) * int(n)
        self.count += int(n)


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with BPI/V-PSNR/V-SSIM/V-LPIPS.")
    parser.add_argument(
        "-m",
        "--model",
        default="vpct-cheng2020-attn",
        choices=sorted(set(image_models.keys()) | set(CUSTOM_MODELS)),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument("--test-split", type=str, default="test", help="Subdirectory name for test split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, default=8, help="Test batch size (ERP count)")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Dataloader prefetch factor")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent workers")
    parser.add_argument(
        "--vp-fov",
        type=float,
        nargs=2,
        default=(90.0, 90.0),
        help="Viewport FoV as (vertical horizontal)",
    )
    parser.add_argument("--num-viewports", type=int, default=6, help="Number of viewports per ERP")
    parser.add_argument("--quality", type=int, default=3, help="CompressAI quality index")
    parser.add_argument("--vpct-layers", type=int, default=1, help="Number of VPCT layers")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--amp", action="store_true", help="Enable AMP autocast for inference")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Autocast dtype when --amp is enabled (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _strip_module_prefix(state_dict):
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _add_module_prefix(state_dict):
    if all(not key.startswith("module.") for key in state_dict.keys()):
        return {f"module.{key}": value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_state(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if isinstance(model, torch.nn.DataParallel):
            model.load_state_dict(_add_module_prefix(_strip_module_prefix(state_dict)), strict=True)
        else:
            model.load_state_dict(_strip_module_prefix(state_dict), strict=True)


def compute_vp_ssim_mean(target, recon, erp_batch_size, num_viewports):
    from pytorch_msssim import ssim

    target = target.clamp(0, 1)
    recon = recon.clamp(0, 1)
    ssim_per_viewport = ssim(target, recon, data_range=1.0, size_average=False)
    ssim_per_image = ssim_per_viewport.view(erp_batch_size, num_viewports).mean(dim=1)
    return float(ssim_per_image.mean().item())


def compute_vp_lpips_mean(target, recon, erp_batch_size, num_viewports, lpips_model):
    target = target.clamp(0, 1)
    recon = recon.clamp(0, 1)
    target_lpips = target * 2.0 - 1.0
    recon_lpips = recon * 2.0 - 1.0
    lpips_per_viewport = lpips_model(target_lpips, recon_lpips).view(-1)
    lpips_per_image = lpips_per_viewport.view(erp_batch_size, num_viewports).mean(dim=1)
    return float(lpips_per_image.mean().item())


def evaluate(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    amp_enabled = bool(args.amp and device == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "Missing package 'lpips'. Install with: pip install lpips"
        ) from exc

    try:
        import pytorch_msssim  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Missing package 'pytorch-msssim'. Install with: pip install pytorch-msssim"
        ) from exc

    dataset = ERPViewportDataset(
        root=args.dataset,
        split=args.test_split,
        fov=args.vp_fov,
        num_viewports=args.num_viewports,
        random_rotate=False,
        random_viewport_subset=False,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(args, device)
    load_checkpoint_state(model, args.checkpoint, device)
    model.eval()

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    bpi = AverageMeter()
    v_psnr = AverageMeter()
    v_ssim = AverageMeter()
    v_lpips = AverageMeter()

    non_blocking = device == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            if batch.dim() != 5:
                raise RuntimeError(f"Expected test batch with shape [B, V, C, H, W], got {tuple(batch.shape)}")

            erp_batch_size, num_viewports, _, _, _ = batch.shape
            batch = batch.to(device, non_blocking=non_blocking)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                out_net = model(batch)

            target = out_net.get("target", batch).float()
            recon = out_net["x_hat"].float().clamp(0, 1)

            bpi_per_viewport = compute_bpi_from_likelihoods(out_net)
            bpi_value = bpi_per_viewport * num_viewports
            vpsnr_value = compute_vp_psnr_mean(target, recon, erp_batch_size, num_viewports)
            vssim_value = compute_vp_ssim_mean(target, recon, erp_batch_size, num_viewports)
            vlpips_value = compute_vp_lpips_mean(target, recon, erp_batch_size, num_viewports, lpips_model)

            bpi.update(bpi_value, erp_batch_size)
            v_psnr.update(vpsnr_value, erp_batch_size)
            v_ssim.update(vssim_value, erp_batch_size)
            v_lpips.update(vlpips_value, erp_batch_size)

    print("===== Evaluation Results =====")
    print(f"Samples: {len(dataset)}")
    print(f"BPI: {bpi.avg:.6f}")
    print(f"V-PSNR: {v_psnr.avg:.6f} dB")
    print(f"V-SSIM: {v_ssim.avg:.6f}")
    print(f"V-LPIPS: {v_lpips.avg:.6f}")


def main(argv=None):
    args = parse_eval_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
