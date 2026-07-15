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
                                radius: int = 5) -> list[tuple[float, float]]:
    """
    Sub-pixel trilateration on the distance map — used for ELUNet dual output.
    Based on: arXiv:2404.19108
    """
    binary = (seg > THRESHOLD).astype(np.float32)
    coarse = (dist <= 2.0) & (binary > 0)
    nms    = (dist == minimum_filter(dist, size=9)) & coarse
    seeds  = np.argwhere(nms)
    centroids: list[tuple[float, float]] = []
    visited = np.zeros_like(binary, dtype=bool)

    for (row0, col0) in seeds:
        if visited[row0, col0]:
            continue
        H, W = binary.shape
        r0, r1 = max(0, row0 - radius), min(H, row0 + radius + 1)
        c0, c1 = max(0, col0 - radius), min(W, col0 + radius + 1)
        patch_seg  = binary[r0:r1, c0:c1]
        patch_dist = dist[r0:r1, c0:c1]
        rows, cols = np.where(patch_seg > 0)
        if len(rows) < 3:
            centroids.append((float(col0), float(row0)))
            visited[r0:r1, c0:c1] |= patch_seg.astype(bool)
            continue
        abs_rows = rows + r0; abs_cols = cols + c0
        dists    = patch_dist[rows, cols]
        ref      = np.argmin(dists)
        y0, x0   = float(abs_rows[ref]), float(abs_cols[ref])
        d0       = dists[ref]
        mask     = np.arange(len(rows)) != ref
        yi = abs_rows[mask].astype(float); xi = abs_cols[mask].astype(float)
        di = dists[mask]
        A  = 2 * np.column_stack([xi - x0, yi - y0])
        b  = (xi**2 - x0**2) + (yi**2 - y0**2) - (di**2 - d0**2)
        if A.shape[0] >= 2:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = x0 + res[0], y0 + res[1]
        else:
            cx, cy = x0, y0
        centroids.append((float(cx), float(cy)))
        visited[r0:r1, c0:c1] |= patch_seg.astype(bool)
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
