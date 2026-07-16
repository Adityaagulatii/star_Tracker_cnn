"""
Train ELUNet (dual-output) from scratch on simulator data.

Usage:
    python -m quantize.train_fp32

Produces: checkpoints/elunet_fp32.pth
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt

from model import ELUNet
from loss  import DualLoss

# ---- Paths ----------------------------------------------------------------
SIM_TRAIN = "C:/Users/adity/Downloads/star_tracker_quant/data/sim_train.npz"
SIM_VAL   = "C:/Users/adity/Downloads/star_tracker_quant/data/sim_val.npz"
CKPT_OUT  = "checkpoints/elunet_fp32.pth"
EPOCHS    = 40
BATCH     = 8
LR        = 1e-3


class SimDualDataset(Dataset):
    """Loads pre-generated npz data; computes distance maps on-the-fly."""
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.images = d["images"]   # (N, 1, H, W) float32 [0,1]
        self.segs   = d["segs"]     # (N, 1, H, W) float32 binary

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]        # (1, H, W)
        seg = self.segs[i]          # (1, H, W)
        dist = distance_transform_edt(seg[0]).astype(np.float32)
        return (torch.from_numpy(img),
                torch.from_numpy(seg),
                torch.from_numpy(dist[np.newaxis]))   # (1, H, W)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model     = ELUNet().to(device)
    criterion = DualLoss(dist_weight=2.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_dl = DataLoader(SimDualDataset(SIM_TRAIN), batch_size=BATCH,
                          shuffle=True, num_workers=0)
    val_dl   = DataLoader(SimDualDataset(SIM_VAL),   batch_size=BATCH,
                          num_workers=0)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr = 0.0
        for imgs, segs, dists in train_dl:
            imgs  = imgs.to(device)
            segs  = segs.to(device)
            dists = dists.to(device)
            optimizer.zero_grad()
            loss, _, _ = criterion(model(imgs), segs, dists)
            loss.backward()
            optimizer.step()
            tr += loss.item()
        tr /= len(train_dl)
        scheduler.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for imgs, segs, dists in val_dl:
                imgs  = imgs.to(device)
                segs  = segs.to(device)
                dists = dists.to(device)
                loss, _, _ = criterion(model(imgs), segs, dists)
                vl += loss.item()
        vl /= len(val_dl)

        print(f"Epoch {epoch:3d}/{EPOCHS}  train={tr:.4f}  val={vl:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"  -> saved {CKPT_OUT}")

    print(f"\nDone. Best val={best_val:.4f}")


if __name__ == "__main__":
    train()
