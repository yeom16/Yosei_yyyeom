import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from config import get_args
from datasets import build_imagenet_loaders
from engine import train_one_epoch, validate_one_epoch
from losses import build_train_criterion
from models import build_model
from profiler import profile_model
from schedulers import build_scheduler
from utils import (
    append_csv,
    cleanup_distributed,
    count_trainable_params,
    ensure_dir,
    init_distributed_mode,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)

def build_optimizer(args, model):
    params = model.parameters()
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    beta1, beta2 = args.betas
    return torch.optim.AdamW(params, lr=args.base_lr, betas=(beta1, beta2), eps=args.eps, weight_decay=args.weight_decay)

def get_state_dict_model(model):
    return model.module if hasattr(model, "module") else model

def main():
    args = get_args()
    init_distributed_mode(args)
    set_seed(args.seed + args.rank)

    if is_main_process(args):
        ensure_dir(args.output_dir)
        save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(args):
        print("=" * 100)
        print("ResNet-50 Ordering Classification v3 DDP")
        print(f"Device            : {device}")
        print(f"Distributed       : {args.distributed}")
        print(f"World size        : {args.world_size}")
        print(f"Model type        : {args.model_type}")
        print(f"Ordering mode     : {args.ordering_mode}")
        print(f"Insert stages     : {args.insert_stages}")
        print(f"Ordering provider : {args.ordering_provider if args.ordering_provider else 'IdentityOrdering'}")
        print(f"AMP               : {args.use_amp} ({args.amp_dtype})")
        print("=" * 100)

    train_loader, val_loader, train_sampler = build_imagenet_loaders(args)
    model = build_model(args).to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    n_params = count_trainable_params(get_state_dict_model(model))
    if is_main_process(args):
        print(f"Trainable params  : {n_params:,}")

    train_criterion = build_train_criterion(args).to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(args, get_state_dict_model(model))
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and args.amp_dtype == "fp16" and device.type == "cuda"))

    start_epoch = 0
    best_acc1 = 0.0
    history = []

    if args.resume:
        ckpt = load_checkpoint(
            args.resume,
            model=get_state_dict_model(model),
            optimizer=None if args.eval_only else optimizer,
            scheduler=None if args.eval_only else scheduler,
            scaler=None if args.eval_only else scaler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0))
        best_acc1 = float(ckpt.get("best_acc1", 0.0))
        history = ckpt.get("history", [])

    if args.eval_only:
        val_stats = validate_one_epoch(model, val_loader, val_criterion, device, args)
        if is_main_process(args):
            profile_stats = profile_model(
                model=get_state_dict_model(model),
                device=device,
                input_size=args.input_size,
                batch_size=args.profile_batch_size,
                warmup=args.profile_warmup,
                iters=args.profile_iters,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
            )
            result = {
                "model_type": args.model_type,
                "ordering_mode": args.ordering_mode,
                "insert_stages": args.insert_stages,
                "acc1": val_stats["val_acc1"],
                "acc5": val_stats["val_acc5"],
                "throughput_img_per_s": profile_stats["throughput_img_per_s"],
                "latency_ms_per_img": profile_stats["latency_ms_per_img"],
                "peak_memory_mb": profile_stats["peak_memory_mb"],
                "trainable_params": n_params,
            }
            save_json(result, os.path.join(args.output_dir, "eval_result.json"))
            append_csv(result, os.path.join(args.output_dir, "summary.csv"))
            print(result)
        cleanup_distributed()
        return

    for epoch in range(start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, train_loader, train_criterion, optimizer, scheduler, scaler, device, epoch, args)
        val_stats = validate_one_epoch(model, val_loader, val_criterion, device, args)

        if is_main_process(args):
            epoch_row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], **train_stats, **val_stats}
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
                "model": get_state_dict_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_acc1": best_acc1,
                "history": history,
                "args": vars(args),
            }
            save_checkpoint(latest_ckpt, os.path.join(args.output_dir, "latest.pth"))

            if val_stats["val_acc1"] > best_acc1:
                best_acc1 = float(val_stats["val_acc1"])
                best_ckpt = {
                    "epoch": epoch + 1,
                    "model": get_state_dict_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_acc1": best_acc1,
                    "history": history,
                    "args": vars(args),
                }
                save_checkpoint(best_ckpt, os.path.join(args.output_dir, "best.pth"))
                print(f"[Saved] best.pth updated | best ACC@1={best_acc1:.2f}")

    if is_main_process(args):
        best_path = os.path.join(args.output_dir, "best.pth")
        if Path(best_path).exists():
            load_checkpoint(best_path, model=get_state_dict_model(model), map_location=device)

        val_stats = validate_one_epoch(model, val_loader, val_criterion, device, args)
        profile_stats = profile_model(
            model=get_state_dict_model(model),
            device=device,
            input_size=args.input_size,
            batch_size=args.profile_batch_size,
            warmup=args.profile_warmup,
            iters=args.profile_iters,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
        )

        result = {
            "model_type": args.model_type,
            "ordering_mode": args.ordering_mode,
            "insert_stages": args.insert_stages,
            "acc1": val_stats["val_acc1"],
            "acc5": val_stats["val_acc5"],
            "throughput_img_per_s": profile_stats["throughput_img_per_s"],
            "latency_ms_per_img": profile_stats["latency_ms_per_img"],
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

    cleanup_distributed()

if __name__ == "__main__":
    main()
