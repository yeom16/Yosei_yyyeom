import time
import torch
from losses import MixupCutmixCollator
from metrics import AverageMeter, accuracy
from utils import get_autocast_context, reduce_tensor

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, args):
    model.train()
    batch_time = AverageMeter(); data_time = AverageMeter()
    loss_meter = AverageMeter(); acc1_meter = AverageMeter(); acc5_meter = AverageMeter()

    mix_collator = MixupCutmixCollator(
        num_classes=args.num_classes,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        switch_prob=args.switch_prob,
        label_smoothing=args.label_smoothing,
    )
    end = time.time()

    for step, (images, target) in enumerate(loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        raw_target = target.clone()

        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            images, target = mix_collator(images, target)

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = get_autocast_context(device, args.use_amp, args.amp_dtype)

        if args.use_amp and device.type == "cuda":
            with autocast_ctx:
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        if args.use_amp and args.amp_dtype == "fp16" and device.type == "cuda":
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

        acc1, acc5 = accuracy(output, raw_target, topk=(1, 5))

        loss_reduced = reduce_tensor(loss.detach(), average=True)
        acc1_reduced = reduce_tensor(acc1.detach(), average=True)
        acc5_reduced = reduce_tensor(acc5.detach(), average=True)

        bs = images.size(0)
        loss_meter.update(loss_reduced.item(), bs)
        acc1_meter.update(acc1_reduced.item(), bs)
        acc5_meter.update(acc5_reduced.item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0 and args.rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Train] Epoch {epoch+1}/{args.epochs} | Step {step}/{len(loader)} | "
                f"LR {lr:.6f} | Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"ACC@1 {acc1_meter.val:.2f} ({acc1_meter.avg:.2f}) | "
                f"ACC@5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f}) | "
                f"Data {data_time.val:.3f}s | Batch {batch_time.val:.3f}s"
            )

    if scheduler is not None:
        scheduler.step()

    return {"train_loss": loss_meter.avg, "train_acc1": acc1_meter.avg, "train_acc5": acc5_meter.avg}

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, args):
    model.eval()
    loss_meter = AverageMeter(); acc1_meter = AverageMeter(); acc5_meter = AverageMeter()

    for step, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        autocast_ctx = get_autocast_context(device, args.use_amp, args.amp_dtype)
        if args.use_amp and device.type == "cuda":
            with autocast_ctx:
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_reduced = reduce_tensor(loss.detach(), average=True)
        acc1_reduced = reduce_tensor(acc1.detach(), average=True)
        acc5_reduced = reduce_tensor(acc5.detach(), average=True)

        bs = images.size(0)
        loss_meter.update(loss_reduced.item(), bs)
        acc1_meter.update(acc1_reduced.item(), bs)
        acc5_meter.update(acc5_reduced.item(), bs)

        if step % args.print_freq == 0 and args.rank == 0:
            print(
                f"[Val] Step {step}/{len(loader)} | "
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"ACC@1 {acc1_meter.val:.2f} ({acc1_meter.avg:.2f}) | "
                f"ACC@5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f})"
            )

    return {"val_loss": loss_meter.avg, "val_acc1": acc1_meter.avg, "val_acc5": acc5_meter.avg}
