import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0, dice_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        dice_loss = self.dice(logits, targets)
        return bce_loss + self.dice_weight * dice_loss


def build_loss(loss_name: str = "bce_dice", pos_weight: float = 5.0) -> nn.Module:
    key = loss_name.lower()

    if key in {"weighted_bce", "bce"}:
        return WeightedBCELoss(pos_weight=pos_weight)

    if key in {"bce_dice", "dice_bce"}:
        return BCEDiceLoss(pos_weight=pos_weight)

    raise ValueError(f"Unsupported loss_name: {loss_name}")


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
