import importlib.util, json
from pathlib import Path
import torch.nn as nn

class IdentityOrdering(nn.Module):
    def __init__(self, channels: int, stage_idx: int, ordering_mode: str = "identity", **kwargs):
        super().__init__()
    def forward(self, x):
        return x

def build_identity_ordering(channels: int, stage_idx: int, ordering_mode: str = "identity", **kwargs) -> nn.Module:
    return IdentityOrdering(channels=channels, stage_idx=stage_idx, ordering_mode=ordering_mode, **kwargs)

def load_external_factory(provider_path: str, factory_name: str = "build_ordering_module"):
    provider_path = str(provider_path).strip()
    if provider_path == "":
        return build_identity_ordering
    path = Path(provider_path)
    if not path.exists():
        raise FileNotFoundError(f"Ordering provider file not found: {provider_path}")
    spec = importlib.util.spec_from_file_location("external_ordering_provider", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, factory_name):
        raise AttributeError(f"{provider_path} does not define factory '{factory_name}'")
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"Factory '{factory_name}' is not callable")
    return factory

def parse_ordering_kwargs(raw: str) -> dict:
    raw = raw.strip()
    return {} if raw == "" else json.loads(raw)
