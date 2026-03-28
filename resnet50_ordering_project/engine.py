import time
from typing import Dict

import torch
import torch.nn as nn

from metrics import AverageMeter, accuracy


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args) -> Dict[str, float]:
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for step, (images, target) in enumerate(loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if args.amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)

        loss_meter.update(loss.item(), bs)
        acc1_meter.update(acc1.item(), bs)
        acc5_meter.update(acc5.item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Train] Epoch {epoch+1}/{args.epochs} | "
                f"Step {step}/{len(loader)} | "
                f"LR {lr:.6f} | "
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"ACC@1 {acc1_meter.val:.2f} ({acc1_meter.avg:.2f}) | "
                f"ACC@5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f}) | "
                f"Data {data_time.val:.3f}s | Batch {batch_time.val:.3f}s"
            )

    return {
        "train_loss": loss_meter.avg,
        "train_acc1": acc1_meter.avg,
        "train_acc5": acc5_meter.avg,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, args) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for step, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)

        loss_meter.update(loss.item(), bs)
        acc1_meter.update(acc1.item(), bs)
        acc5_meter.update(acc5.item(), bs)

        if step % args.print_freq == 0:
            print(
                f"[Val] Step {step}/{len(loader)} | "
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"ACC@1 {acc1_meter.val:.2f} ({acc1_meter.avg:.2f}) | "
                f"ACC@5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f})"
            )

    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
    }
