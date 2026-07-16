"""
Fine-tune FP32 ELUNet with LSQ INT8 quantization.

Usage:
    python -m quantize.train_lsq

Requires: checkpoints/elunet_fp32.pth  (run train_fp32.py first)
Produces: checkpoints/elunet_lsq.pth
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt

from model             import ELUNet
from loss              import DualLoss
from quantize.lsq      import replace_conv2d_with_lsq

SIM_TRAIN = "C:/Users/adity/Downloads/star_tracker_quant/data/sim_train.npz"
SIM_VAL   = "C:/Users/adity/Downloads/star_tracker_quant/data/sim_val.npz"
CKPT_IN   = "checkpoints/elunet_fp32.pth"
CKPT_OUT  = "checkpoints/elunet_lsq.pth"
EPOCHS    = 15
BATCH     = 8
LR        = 1e-4


class SimDualDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.images = d["images"]
        self.segs   = d["segs"]

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        img  = self.images[i]
        seg  = self.segs[i]
        dist = distance_transform_edt(seg[0]).astype(np.float32)
        return (torch.from_numpy(img),
                torch.from_numpy(seg),
                torch.from_numpy(dist[np.newaxis]))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ELUNet()
    model.load_state_dict(torch.load(CKPT_IN, map_location="cpu", weights_only=True))
    model = replace_conv2d_with_lsq(model, n_bits=8)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"LSQ model params: {params:,}")

    criterion = DualLoss(dist_weight=2.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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

        print(f"Epoch {epoch:2d}/{EPOCHS}  train={tr:.4f}  val={vl:.4f}  {time.time()-t0:.1f}s")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"  -> saved {CKPT_OUT}")

    print(f"\nDone. Best val={best_val:.4f}")


if __name__ == "__main__":
    train()
