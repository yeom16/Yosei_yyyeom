import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        return torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()

def one_hot(target: torch.Tensor, num_classes: int, device=None):
    if device is None:
        device = target.device
    out = torch.zeros((target.size(0), num_classes), device=device, dtype=torch.float32)
    out.scatter_(1, target.view(-1, 1), 1.0)
    return out

class MixupCutmixCollator:
    def __init__(self, num_classes: int, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0,
                 mix_prob: float = 1.0, switch_prob: float = 0.5, label_smoothing: float = 0.1):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing

    def _smooth_target(self, y: torch.Tensor) -> torch.Tensor:
        y = one_hot(y, self.num_classes)
        off_value = self.label_smoothing / self.num_classes
        on_value = 1.0 - self.label_smoothing + off_value
        return y * (on_value - off_value) + off_value

    def _rand_bbox(self, W, H, lam):
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
        cx = np.random.randint(W); cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if np.random.rand() > self.mix_prob:
            return x, self._smooth_target(y)
        bs = x.size(0)
        perm = torch.randperm(bs, device=x.device)
        y1 = self._smooth_target(y); y2 = self._smooth_target(y[perm])
        use_cutmix = (self.cutmix_alpha > 0) and (np.random.rand() < self.switch_prob)
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            _, _, H, W = x.shape
            x1, y1_box, x2, y2_box = self._rand_bbox(W, H, lam)
            x[:, :, y1_box:y2_box, x1:x2] = x[perm, :, y1_box:y2_box, x1:x2]
            box_area = max(0, x2 - x1) * max(0, y2_box - y1_box)
            lam = 1.0 - box_area / float(W * H)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = x * lam + x[perm] * (1.0 - lam)
        return x, y1 * lam + y2 * (1.0 - lam)

def build_train_criterion(args):
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        return SoftTargetCrossEntropy()
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
