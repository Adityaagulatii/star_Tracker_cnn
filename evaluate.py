"""
evaluate.py — test all models; ELUNet uses trilateration on the dist map.

Usage:  python evaluate.py
"""

import time, random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label, center_of_mass, minimum_filter
from torch.utils.data import DataLoader

from dataset import StarDataset
from model   import UNet, MobileUNet, ELUNet

MODELS       = {'unet': UNet, 'mobileunet': MobileUNet, 'elunet': ELUNet}
THRESHOLD    = 0.5
MATCH_RADIUS = 5


# ── Centroid extraction ───────────────────────────────────────────────────────

def get_centroids_com(seg_map: np.ndarray) -> list[tuple[float, float]]:
    """Center-of-mass — used for UNet and MobileUNet (single-channel output)."""
    binary     = (seg_map > THRESHOLD).astype(np.int32)
    labeled, n = label(binary)
    return [(float(cx), float(cy))
            for cy, cx in [center_of_mass(binary, labels=labeled, index=i)
                           for i in range(1, n + 1)]]


def get_centroids_trilateration(seg: np.ndarray, dist: np.ndarray,
                                radius: int = 7) -> list[tuple[float, float]]:
    """
    Per-blob: peak of dist map gives seed, trilateration refines to sub-pixel.
    Based on: arXiv:2404.19108
    """
    binary    = (seg > THRESHOLD).astype(np.float32)
    labeled, n = label(binary)
    if n == 0:
        return []
    centroids = []
    for i in range(1, n + 1):
        mask_blob = labeled == i
        if mask_blob.sum() < 2:
            continue
        dist_blob = np.where(mask_blob, dist, -np.inf)
        r0, c0   = np.unravel_index(np.argmax(dist_blob), dist_blob.shape)
        H, W     = binary.shape
        rs, re   = max(0, r0-radius), min(H, r0+radius+1)
        cs, ce   = max(0, c0-radius), min(W, c0+radius+1)
        pseg     = binary[rs:re, cs:ce]
        pdist    = dist[rs:re, cs:ce]
        rows, cols = np.where(pseg > 0)
        if len(rows) < 3:
            centroids.append((float(c0), float(r0)))
            continue
        ar = rows + rs; ac = cols + cs; ds = pdist[rows, cols]
        pos = ds > 0
        if not pos.any():
            centroids.append((float(c0), float(r0)))
            continue
        ref  = np.where(pos)[0][np.argmin(ds[pos])]
        y0_  = float(ar[ref]); x0_ = float(ac[ref]); d0_ = float(ds[ref])
        m    = np.arange(len(rows)) != ref
        yi   = ar[m].astype(float); xi = ac[m].astype(float); di = ds[m]
        A    = 2 * np.column_stack([xi - x0_, yi - y0_])
        b    = (xi**2 - x0_**2) + (yi**2 - y0_**2) - (di**2 - d0_**2)
        if A.shape[0] >= 2:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx = float(np.clip(x0_ + res[0], 0, W-1))
            cy = float(np.clip(y0_ + res[1], 0, H-1))
        else:
            cx, cy = float(c0), float(r0)
        centroids.append((cx, cy))
    return centroids


def match(pred_cents, true_cents, radius):
    matched = set()
    tp, errors = 0, []
    for px, py in pred_cents:
        best_d, best_j = float('inf'), -1
        for j, (tx, ty) in enumerate(true_cents):
            if j in matched:
                continue
            d = np.hypot(px - tx, py - ty)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= radius:
            tp += 1; matched.add(best_j); errors.append(best_d)
    return tp, len(pred_cents) - tp, len(true_cents) - tp, errors


# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate(model_name, device):
    ckpt = Path(f"checkpoints/{model_name}_best.pth")
    if not ckpt.exists():
        return None

    model     = MODELS[model_name]().to(device)
    dual_out  = (model_name == 'elunet')
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(StarDataset('data/val'), batch_size=1,
                        shuffle=False, num_workers=0)

    tp_tot = fp_tot = fn_tot = 0
    all_errors, times = [], []

    with torch.no_grad():
        for images, segs, _ in loader:
            images = images.to(device)
            t0     = time.perf_counter()
            pred   = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

            true_np = segs[0, 0].numpy()
            true_c  = get_centroids_com(true_np)

            if dual_out:
                pred_c = get_centroids_trilateration(
                    pred[0, 0].cpu().numpy(),
                    pred[0, 1].cpu().numpy()
                )
            else:
                pred_c = get_centroids_com(pred[0, 0].cpu().numpy())

            tp, fp, fn, errs = match(pred_c, true_c, MATCH_RADIUS)
            tp_tot += tp; fp_tot += fp; fn_tot += fn
            all_errors.extend(errs)

    precision = tp_tot / (tp_tot + fp_tot + 1e-8)
    recall    = tp_tot / (tp_tot + fn_tot + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    mean_err  = float(np.mean(all_errors)) if all_errors else float('nan')
    mean_time = float(np.mean(times[5:])) if len(times) > 5 else float(np.mean(times))
    params    = sum(p.numel() for p in model.parameters())
    return dict(f1=f1, precision=precision, recall=recall,
                mean_err=mean_err, time_ms=mean_time, params=params,
                centroid='trilateration' if dual_out else 'center-of-mass')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating on {device}\n")
    print(f"{'Model':<14}{'F1':>7}{'Err(px)':>10}{'ms/img':>9}{'Centroiding'}")
    print('-' * 55)

    for name in MODELS:
        r = evaluate(name, device)
        if r is None:
            print(f"{name:<14}  no checkpoint")
            continue
        print(f"{name:<14}{r['f1']:>7.3f}{r['mean_err']:>10.2f}"
              f"{r['time_ms']:>9.1f}  {r['centroid']}")
