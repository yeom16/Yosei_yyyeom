import time
from typing import Dict

import torch


@torch.no_grad()
def measure_throughput(model, device, input_size=224, batch_size=64, warmup=30, iters=100, amp=False) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    for _ in range(warmup):
        if amp:
            with torch.cuda.amp.autocast():
                _ = model(x)
        else:
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        if amp:
            with torch.cuda.amp.autocast():
                _ = model(x)
        else:
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_images = batch_size * iters
    throughput = total_images / elapsed
    return throughput


@torch.no_grad()
def measure_peak_memory(model, device, input_size=224, batch_size=64, warmup=10, iters=30, amp=False) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    if device.type != "cuda":
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        if amp:
            with torch.cuda.amp.autocast():
                _ = model(x)
        else:
            _ = model(x)

    torch.cuda.synchronize(device)

    for _ in range(iters):
        if amp:
            with torch.cuda.amp.autocast():
                _ = model(x)
        else:
            _ = model(x)

    torch.cuda.synchronize(device)
    mem_bytes = torch.cuda.max_memory_allocated(device)
    mem_mb = mem_bytes / (1024 ** 2)
    return mem_mb


def profile_model(model, device, input_size=224, batch_size=64, warmup=30, iters=100, amp=False) -> Dict[str, float]:
    throughput = measure_throughput(
        model=model,
        device=device,
        input_size=input_size,
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
        amp=amp,
    )
    mem_mb = measure_peak_memory(
        model=model,
        device=device,
        input_size=input_size,
        batch_size=batch_size,
        warmup=max(10, warmup // 3),
        iters=max(30, iters // 3),
        amp=amp,
    )
    return {
        "throughput_img_per_s": throughput,
        "peak_memory_mb": mem_mb,
    }
