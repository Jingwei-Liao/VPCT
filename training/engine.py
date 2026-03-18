import math

import torch


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_bpi_from_likelihoods(out_net):
    num_pixels = out_net["x_hat"].size(0)
    bits_per_sample = sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    ).item()
    return bits_per_sample / 8.0 / 1024.0


def compute_vp_psnr_mean(target, recon, erp_batch_size, num_viewports):
    recon = recon.clamp(0, 1)
    mse_per_viewport = ((target - recon) ** 2).mean(dim=(1, 2, 3))
    psnr_per_viewport = 10.0 * torch.log10(1.0 / torch.clamp(mse_per_viewport, min=1e-12))
    psnr_per_image = psnr_per_viewport.view(erp_batch_size, num_viewports).mean(dim=1)
    return float(psnr_per_image.mean().item())


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device
    non_blocking = device.type == "cuda"

    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(batch)
        criterion_target = out_net.get("target", batch)
        out_criterion = criterion(out_net, criterion_target)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(batch)}/{len(train_dataloader.dataset)}"
                f" ({100.0 * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    non_blocking = device.type == "cuda"

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    bpi = AverageMeter()
    mse_loss = AverageMeter()
    vp_psnr = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for batch in test_dataloader:
            if batch.dim() == 5:
                erp_batch_size, num_viewports, _, _, _ = batch.shape
            else:
                erp_batch_size = batch.size(0)
                num_viewports = 1

            batch = batch.to(device, non_blocking=non_blocking)
            out_net = model(batch)
            criterion_target = out_net.get("target", batch)
            out_criterion = criterion(out_net, criterion_target)

            viewport_batch_size = criterion_target.size(0)
            mse_value = float(out_criterion["mse_loss"].item())
            vp_psnr_value = compute_vp_psnr_mean(
                target=criterion_target,
                recon=out_net["x_hat"],
                erp_batch_size=erp_batch_size,
                num_viewports=num_viewports,
            )

            bpp_value = float(out_criterion["bpp_loss"].item())
            bpi_per_viewport = compute_bpi_from_likelihoods(out_net)
            bpi_value = bpi_per_viewport * num_viewports

            aux_loss.update(model.aux_loss(), viewport_batch_size)
            bpp_loss.update(bpp_value, viewport_batch_size)
            bpi.update(bpi_value, erp_batch_size)
            loss.update(out_criterion["loss"], viewport_batch_size)
            mse_loss.update(mse_value, viewport_batch_size)
            vp_psnr.update(vp_psnr_value, erp_batch_size)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tBPI: {bpi.avg:.2f} |"
        f"\tVP-PSNR: {vp_psnr.avg:.2f} dB |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg
