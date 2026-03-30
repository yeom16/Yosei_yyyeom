import time
import torch
from utils import get_autocast_context

@torch.no_grad()
def measure_throughput_and_latency(model, device, input_size=224, batch_size=64, warmup=50, iters=200, use_amp=False, amp_dtype="bf16"):
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    for _ in range(warmup):
        autocast_ctx = get_autocast_context(device, use_amp, amp_dtype)
        if use_amp and device.type == "cuda":
            with autocast_ctx:
                _ = model(x)
        else:
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        autocast_ctx = get_autocast_context(device, use_amp, amp_dtype)
        if use_amp and device.type == "cuda":
            with autocast_ctx:
                _ = model(x)
        else:
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    total_images = batch_size * iters
    return total_images / elapsed, (elapsed / total_images) * 1000.0

@torch.no_grad()
def measure_peak_memory(model, device, input_size=224, batch_size=64, warmup=50, iters=200, use_amp=False, amp_dtype="bf16"):
    model.eval()
    if device.type != "cuda":
        return 0.0
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        autocast_ctx = get_autocast_context(device, use_amp, amp_dtype)
        if use_amp:
            with autocast_ctx:
                _ = model(x)
        else:
            _ = model(x)
    torch.cuda.synchronize(device)

    for _ in range(iters):
        autocast_ctx = get_autocast_context(device, use_amp, amp_dtype)
        if use_amp:
            with autocast_ctx:
                _ = model(x)
        else:
            _ = model(x)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)

def profile_model(model, device, input_size=224, batch_size=64, warmup=50, iters=200, use_amp=False, amp_dtype="bf16"):
    imgps, msimg = measure_throughput_and_latency(model, device, input_size, batch_size, warmup, iters, use_amp, amp_dtype)
    mem = measure_peak_memory(model, device, input_size, batch_size, warmup, iters, use_amp, amp_dtype)
    return {"throughput_img_per_s": imgps, "latency_ms_per_img": msimg, "peak_memory_mb": mem}
