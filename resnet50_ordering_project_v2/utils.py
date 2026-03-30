import csv, json, os, random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.distributed as dist

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_csv(row: Dict[str, Any], path: str):
    ensure_dir(str(Path(path).parent))
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state: Dict[str, Any], path: str):
    ensure_dir(str(Path(path).parent))
    torch.save(state, path)

def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt

def get_autocast_dtype(amp_dtype: str):
    return torch.bfloat16 if amp_dtype == "bf16" else torch.float16

def get_autocast_context(device: torch.device, enabled: bool, amp_dtype: str):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=get_autocast_dtype(amp_dtype), enabled=True)

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        return

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process(args):
    return args.rank == 0

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

@torch.no_grad()
def reduce_tensor(tensor: torch.Tensor, average: bool = True):
    if not is_dist_avail_and_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= dist.get_world_size()
    return rt
