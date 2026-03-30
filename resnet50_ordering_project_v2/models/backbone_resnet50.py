from dataclasses import dataclass
from typing import Iterable, Optional, Set
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from .order_interface import build_identity_ordering

@dataclass
class BackboneFeatures:
    stem: torch.Tensor
    stage1: torch.Tensor
    stage2: torch.Tensor
    stage3: torch.Tensor
    stage4: torch.Tensor

class OrderedStage(nn.Module):
    def __init__(self, stage: nn.Module, ordering_module: nn.Module):
        super().__init__()
        self.stage = stage
        self.ordering = ordering_module
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage(x)
        x = self.ordering(x)
        return x

class ResNet50StageBackbone(nn.Module):
    stage_channels = {1: 256, 2: 512, 3: 1024, 4: 2048}
    def __init__(self, pretrained: bool = False, zero_init_residual: bool = False,
                 insert_stages: Optional[Iterable[int]] = None, ordering_factory=None,
                 ordering_mode: str = "identity", ordering_kwargs: Optional[dict] = None):
        super().__init__()
        if insert_stages is None:
            insert_stages = [1, 2, 3, 4]
        self.insert_stages: Set[int] = set(insert_stages)
        self.ordering_mode = ordering_mode
        self.ordering_kwargs = ordering_kwargs or {}

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights, zero_init_residual=zero_init_residual)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        if ordering_factory is None:
            ordering_factory = build_identity_ordering

        self.layer1 = OrderedStage(base.layer1, self._build_ordering_module(ordering_factory, 1))
        self.layer2 = OrderedStage(base.layer2, self._build_ordering_module(ordering_factory, 2))
        self.layer3 = OrderedStage(base.layer3, self._build_ordering_module(ordering_factory, 3))
        self.layer4 = OrderedStage(base.layer4, self._build_ordering_module(ordering_factory, 4))
        self.out_channels = 2048

    def _build_ordering_module(self, ordering_factory, stage_idx: int) -> nn.Module:
        channels = self.stage_channels[stage_idx]
        if stage_idx in self.insert_stages:
            return ordering_factory(channels=channels, stage_idx=stage_idx, ordering_mode=self.ordering_mode, **self.ordering_kwargs)
        return build_identity_ordering(channels=channels, stage_idx=stage_idx, ordering_mode="identity")

    def forward_features(self, x: torch.Tensor) -> BackboneFeatures:
        stem = self.stem(x)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return BackboneFeatures(stem=stem, stage1=s1, stage2=s2, stage3=s3, stage4=s4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x).stage4
