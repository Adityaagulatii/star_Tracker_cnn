import torch
import torch.nn as nn


class SegLoss(nn.Module):
    """Binary cross-entropy — used by UNet and MobileUNet (single-channel output)."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class DualLoss(nn.Module):
    """
    Combined loss for ELUNet dual-output (seg + dist map).
    Based on: arXiv:2404.19108

    pred    : (B, 2, H, W)  — ch0 = seg prob (sigmoid), ch1 = dist (raw)
    seg_gt  : (B, 1, H, W)  — binary ground-truth segmentation
    dist_gt : (B, 1, H, W)  — distance transform in pixels

    total = 2.5 * MSE(dist_pred * mask, dist_gt * mask) + BCE(seg_pred, seg_gt)
    """
    def __init__(self, dist_weight: float = 2.5):
        super().__init__()
        self.dist_weight = dist_weight
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, seg_gt, dist_gt):
        seg_pred  = pred[:, 0:1]
        dist_pred = pred[:, 1:2]
        seg_loss  = self.bce(seg_pred, seg_gt)
        mask      = seg_gt
        dist_loss = self.mse(dist_pred * mask, dist_gt * mask)
        total     = self.dist_weight * dist_loss + seg_loss
        return total, seg_loss, dist_loss
