import random
import shutil
import json
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from compressai.losses import RateDistortionLoss

from .constants import QUALITY_TO_LMBDA
from .datasets import ERPViewportDataset
from .engine import test_epoch, train_one_epoch
from .modeling import build_model, configure_optimizers


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_filename="checkpoint_best_loss.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def _format_float(value):
    text = f"{float(value):g}"
    return text.replace(".", "p")


def _build_experiment_name(args):
    if args.experiment_name:
        return args.experiment_name

    fov_h, fov_w = args.vp_fov
    return (
        f"{args.model}"
        f"_q{args.quality}"
        f"_vpct{args.vpct_layers}"
        f"_bs{args.batch_size}"
        f"_nvp{args.num_viewports}"
        f"_fov{_format_float(fov_h)}x{_format_float(fov_w)}"
        f"_lr{_format_float(args.learning_rate)}"
    )


def _prepare_save_dir(args):
    experiment_name = _build_experiment_name(args)
    save_dir = os.path.join(args.save_root, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    args_path = os.path.join(save_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)

    return save_dir


def resolve_lambda(args):
    if args.lmbda is not None:
        print(f"Using manual lambda: lmbda={args.lmbda}")
        return args.lmbda

    if args.quality not in QUALITY_TO_LMBDA:
        raise ValueError(
            f"No default lmbda configured for quality={args.quality}. "
            f"Supported qualities: {sorted(QUALITY_TO_LMBDA.keys())}. "
            "Please set --lambda manually."
        )

    lmbda = QUALITY_TO_LMBDA[args.quality]
    print(f"Auto lambda from quality={args.quality}: lmbda={lmbda}")
    return lmbda


def run_training(args):
    args.lmbda = resolve_lambda(args)
    save_dir = _prepare_save_dir(args)
    print(f"Saving checkpoints to: {save_dir}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = ERPViewportDataset(
        root=args.dataset,
        split=args.train_split,
        fov=args.vp_fov,
        num_viewports=args.num_viewports,
        random_rotate=args.random_vp_rotate,
        random_viewport_subset=args.random_vp_subset,
    )
    test_dataset = ERPViewportDataset(
        root=args.dataset,
        split=args.test_split,
        fov=args.vp_fov,
        num_viewports=args.num_viewports,
        random_rotate=False,
        random_viewport_subset=False,
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device == "cuda"),
    }
    if args.num_workers > 0:
        common_loader_kwargs["persistent_workers"] = not args.no_persistent_workers
        common_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **common_loader_kwargs,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    net = build_model(args, device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename=os.path.join(save_dir, "checkpoint.pth.tar"),
                best_filename=os.path.join(save_dir, "checkpoint_best_loss.pth.tar"),
            )
