import torch.nn as nn


class SegLoss(nn.Module):
    """Binary cross-entropy for segmentation maps."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target)
