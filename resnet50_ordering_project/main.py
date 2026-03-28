import os
from pathlib import Path

import torch
import torch.nn as nn

from config import get_args
from datasets import build_imagenet_loaders
from engine import train_one_epoch, validate_one_epoch
from models import build_model
from profiler import profile_model
from utils import (
    append_csv,
    count_trainable_params,
    ensure_dir,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)


def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(args.epochs // 3, 1),
        gamma=0.1,
    )


def main():
    args = get_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("ResNet-50 Ordering Classification")
    print(f"Device            : {device}")
    print(f"Model type        : {args.model_type}")
    print(f"Ordering mode     : {args.ordering_mode}")
    print(f"Insert stages     : {args.insert_stages}")
    print(f"Ordering provider : {args.ordering_provider if args.ordering_provider else 'IdentityOrdering'}")
    print("=" * 100)

    train_loader, val_loader = build_imagenet_loaders(args)
    model = build_model(args).to(device)

    n_params = count_trainable_params(model)
    print(f"Trainable params  : {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_acc1 = 0.0
    history = []

    if args.resume:
        ckpt = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=None if args.eval_only else optimizer,
            scheduler=None if args.eval_only else scheduler,
            scaler=None if args.eval_only else scaler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0))
        best_acc1 = float(ckpt.get("best_acc1", 0.0))
        history = ckpt.get("history", [])
        print(f"Loaded checkpoint from {args.resume} | start_epoch={start_epoch} | best_acc1={best_acc1:.2f}")

    if args.eval_only:
        val_stats = validate_one_epoch(model, val_loader, criterion, device, args)
        profile_stats = profile_model(
            model=model,
            device=device,
            input_size=args.input_size,
            batch_size=args.profile_batch_size,
            warmup=args.profile_warmup,
            iters=args.profile_iters,
            amp=args.amp,
        )

        result = {
            "model_type": args.model_type,
            "ordering_mode": args.ordering_mode,
            "insert_stages": args.insert_stages,
            "acc1": val_stats["val_acc1"],
            "acc5": val_stats["val_acc5"],
            "throughput_img_per_s": profile_stats["throughput_img_per_s"],
            "peak_memory_mb": profile_stats["peak_memory_mb"],
            "trainable_params": n_params,
        }
        save_json(result, os.path.join(args.output_dir, "eval_result.json"))
        append_csv(result, os.path.join(args.output_dir, "summary.csv"))
        print(result)
        return

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        val_stats = validate_one_epoch(model, val_loader, criterion, device, args)
        scheduler.step()

        epoch_row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **train_stats,
            **val_stats,
        }
        history.append(epoch_row)
        append_csv(epoch_row, os.path.join(args.output_dir, "train_log.csv"))

        print("-" * 100)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train ACC@1 {train_stats['train_acc1']:.2f} | "
            f"Val ACC@1 {val_stats['val_acc1']:.2f} | "
            f"Best ACC@1 {best_acc1:.2f}"
        )
        print("-" * 100)

        latest_ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if args.amp else None,
            "best_acc1": best_acc1,
            "history": history,
            "args": vars(args),
        }
        save_checkpoint(latest_ckpt, os.path.join(args.output_dir, "latest.pth"))

        if val_stats["val_acc1"] > best_acc1:
            best_acc1 = float(val_stats["val_acc1"])
            best_ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "best_acc1": best_acc1,
                "history": history,
                "args": vars(args),
            }
            save_checkpoint(best_ckpt, os.path.join(args.output_dir, "best.pth"))
            print(f"[Saved] best.pth updated | best ACC@1={best_acc1:.2f}")

    # final evaluation with best checkpoint
    best_path = os.path.join(args.output_dir, "best.pth")
    if Path(best_path).exists():
        _ = load_checkpoint(best_path, model=model, map_location=device)

    val_stats = validate_one_epoch(model, val_loader, criterion, device, args)
    profile_stats = profile_model(
        model=model,
        device=device,
        input_size=args.input_size,
        batch_size=args.profile_batch_size,
        warmup=args.profile_warmup,
        iters=args.profile_iters,
        amp=args.amp,
    )

    result = {
        "model_type": args.model_type,
        "ordering_mode": args.ordering_mode,
        "insert_stages": args.insert_stages,
        "acc1": val_stats["val_acc1"],
        "acc5": val_stats["val_acc5"],
        "throughput_img_per_s": profile_stats["throughput_img_per_s"],
        "peak_memory_mb": profile_stats["peak_memory_mb"],
        "trainable_params": n_params,
        "best_acc1": best_acc1,
    }
    save_json(result, os.path.join(args.output_dir, "final_result.json"))
    append_csv(result, os.path.join(args.output_dir, "summary.csv"))

    print("=" * 100)
    print("Final Result")
    print(result)
    print("=" * 100)


if __name__ == "__main__":
    main()
