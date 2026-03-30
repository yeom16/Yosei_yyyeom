import torch
import torch.nn as nn
from .backbone_resnet50 import ResNet50StageBackbone

class ResNet50OrderedClassifier(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained_backbone: bool = False,
                 zero_init_residual: bool = False, insert_stages=(1, 2, 3, 4),
                 ordering_factory=None, ordering_mode: str = "identity",
                 ordering_kwargs=None, drop_rate: float = 0.0):
        super().__init__()
        self.backbone = ResNet50StageBackbone(
            pretrained=pretrained_backbone,
            zero_init_residual=zero_init_residual,
            insert_stages=insert_stages,
            ordering_factory=ordering_factory,
            ordering_mode=ordering_mode,
            ordering_kwargs=ordering_kwargs,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet50BaselineClassifier(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained_backbone: bool = False,
                 zero_init_residual: bool = False, drop_rate: float = 0.0):
        super().__init__()
        self.model = ResNet50OrderedClassifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            zero_init_residual=zero_init_residual,
            insert_stages=[],
            ordering_factory=None,
            ordering_mode="identity",
            ordering_kwargs={},
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
