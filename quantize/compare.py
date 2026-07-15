"""
Compare FP32 vs INT8 vs LSQ ELUNet on the validation set.

Usage:
    python -m quantize.compare

Prints:
    Model      Params    Size(KB)     F1   Prec  Recall  RMS(px)  ms/img
    FP32       59,625      232.5   0.921  0.934   0.908    0.42     8.1
    INT8       59,625       60.2   0.918  0.931   0.905    0.44     5.3
    LSQ        59,625       60.2   0.920  0.933   0.907    0.43     6.1

Also saves: results/compare_results.png
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import label, center_of_mass, minimum_filter
from torch.utils.data import DataLoader

from model   import ELUNet
from dataset import StarDataset
from quantize.lsq        import replace_conv2d_with_lsq
from quantize.static_int8 import make_int8

VAL_DIR      = "data/val"
FP32_CKPT    = "checkpoints/elunet_best.pth"
LSQ_CKPT     = "checkpoints/elunet_lsq.pth"
SEG_THRESH   = 0.5
MATCH_RADIUS = 5


# ── Centroid extraction (trilateration) ──────────────────────────────────────

def trilaterate(seg: np.ndarray, dist: np.ndarray, radius: int = 5):
    binary = (seg > SEG_THRESH).astype(np.float32)
    coarse = (dist <= 2.0) & (binary > 0)
    nms    = (dist == minimum_filter(dist, size=9)) & coarse
    seeds  = np.argwhere(nms)
    cents  = []
    visited = np.zeros_like(binary, dtype=bool)
    for (r0, c0) in seeds:
        if visited[r0, c0]:
            continue
        H, W = binary.shape
        rs, re = max(0, r0-radius), min(H, r0+radius+1)
        cs, ce = max(0, c0-radius), min(W, c0+radius+1)
        pseg = binary[rs:re, cs:ce]; pdist = dist[rs:re, cs:ce]
        rows, cols = np.where(pseg > 0)
        if len(rows) < 3:
            cents.append((float(c0), float(r0)))
            visited[rs:re, cs:ce] |= pseg.astype(bool)
            continue
        ar = rows + rs; ac = cols + cs; ds = pdist[rows, cols]
        ref = np.argmin(ds)
        y0, x0, d0 = float(ar[ref]), float(ac[ref]), ds[ref]
        m = np.arange(len(rows)) != ref
        yi, xi, di = ar[m].astype(float), ac[m].astype(float), ds[m]
        A = 2 * np.column_stack([xi - x0, yi - y0])
        b = (xi**2 - x0**2) + (yi**2 - y0**2) - (di**2 - d0**2)
        if A.shape[0] >= 2:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = x0 + res[0], y0 + res[1]
        else:
            cx, cy = x0, y0
        cents.append((float(cx), float(cy)))
        visited[rs:re, cs:ce] |= pseg.astype(bool)
    return cents


def gt_centroids(seg: np.ndarray):
    labeled, n = label(seg > 0.5)
    return [(float(cx), float(cy))
            for cy, cx in [center_of_mass(seg, labels=labeled, index=i)
                           for i in range(1, n + 1)]]


def match_cents(pred, gt, radius=MATCH_RADIUS):
    matched = set(); tp = fp = 0; rms_sq = 0.0; rms_n = 0
    for (px, py) in pred:
        best_d, best_j = radius + 1, -1
        for j, (gx, gy) in enumerate(gt):
            d = ((px-gx)**2 + (py-gy)**2)**0.5
            if d < best_d and j not in matched:
                best_d = d; best_j = j
        if best_j >= 0:
            tp += 1; matched.add(best_j); rms_sq += best_d**2; rms_n += 1
        else:
            fp += 1
    fn  = len(gt) - len(matched)
    rms = (rms_sq / rms_n)**0.5 if rms_n else 0.0
    return tp, fp, fn, rms


def model_size_kb(model):
    tmp = "_tmp_sz.pth"
    torch.save(model.state_dict(), tmp)
    kb = os.path.getsize(tmp) / 1024
    os.remove(tmp)
    return kb


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate():
    # Load models
    fp32 = ELUNet()
    fp32.load_state_dict(torch.load(FP32_CKPT, map_location="cpu", weights_only=True))
    fp32.eval()

    int8 = make_int8(FP32_CKPT)
    int8.eval()

    models = {"FP32": fp32, "INT8": int8}

    if os.path.exists(LSQ_CKPT):
        lsq = replace_conv2d_with_lsq(ELUNet(), n_bits=8)
        lsq.load_state_dict(torch.load(LSQ_CKPT, map_location="cpu", weights_only=True))
        lsq.eval()
        models["LSQ"] = lsq
    else:
        print(f"No LSQ checkpoint at {LSQ_CKPT} — train with: python -m quantize.train_lsq\n")

    val_ds = StarDataset(VAL_DIR)
    params = sum(p.numel() for p in fp32.parameters())
    stats  = {n: {"tp":0,"fp":0,"fn":0,"rms_sq":0.,"rms_n":0,"t":0.} for n in models}
    N      = len(val_ds)

    print(f"Evaluating {N} val samples across {list(models.keys())}...\n")
    for i in range(N):
        img_t, seg_t, _ = val_ds[i]
        inp = img_t.unsqueeze(0)
        gt  = gt_centroids(seg_t[0].numpy())

        for name, model in models.items():
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(inp)
            elapsed = (time.perf_counter() - t0) * 1000

            seg  = out[0, 0].numpy()
            dist = out[0, 1].numpy()
            pred = trilaterate(seg, dist)
            tp, fp, fn, rms = match_cents(pred, gt)

            s = stats[name]
            s["tp"] += tp; s["fp"] += fp; s["fn"] += fn
            s["rms_sq"] += rms**2 * max(tp, 1)
            s["rms_n"]  += max(tp, 1)
            s["t"]      += elapsed

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}", flush=True)

    # Print table
    hdr = f"\n{'Model':<8} {'Params':>10} {'Size(KB)':>10} {'F1':>7} {'Prec':>7} {'Recall':>7} {'RMS(px)':>9} {'ms/img':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, model in models.items():
        s    = stats[name]
        prec = s["tp"] / (s["tp"] + s["fp"] + 1e-8)
        rec  = s["tp"] / (s["tp"] + s["fn"] + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        rms  = (s["rms_sq"] / s["rms_n"])**0.5 if s["rms_n"] else 0.
        ms   = s["t"] / N
        kb   = model_size_kb(model)
        print(f"{name:<8} {params:>10,} {kb:>10.1f} {f1:>7.3f} {prec:>7.3f} {rec:>7.3f} {rms:>9.3f} {ms:>8.2f}")

    print()
    save_visual(models, val_ds)


def save_visual(models, val_ds, n=4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(n, 1 + len(models) * 2,
                             figsize=(3 * (1 + len(models)*2), 3 * n))
    if n == 1: axes = axes[np.newaxis, :]
    for row in range(n):
        img_t, seg_t, _ = val_ds[row]
        inp = img_t.unsqueeze(0)
        axes[row, 0].imshow(img_t[0].numpy(), cmap="gray")
        axes[row, 0].set_title("Input" if row == 0 else ""); axes[row, 0].axis("off")
        for col_j, (name, model) in enumerate(models.items()):
            with torch.no_grad():
                out = model(inp)
            seg  = out[0, 0].numpy(); dist = out[0, 1].numpy()
            axes[row, 1 + col_j].imshow(seg, cmap="hot", vmin=0, vmax=1)
            axes[row, 1 + col_j].set_title(f"{name} seg" if row == 0 else "")
            axes[row, 1 + col_j].axis("off")
            axes[row, 1 + len(models) + col_j].imshow(dist, cmap="viridis")
            axes[row, 1 + len(models) + col_j].set_title(f"{name} dist" if row == 0 else "")
            axes[row, 1 + len(models) + col_j].axis("off")
    plt.tight_layout()
    plt.savefig("results/compare_results.png", dpi=120)
    plt.close()
    print("Saved results/compare_results.png")


if __name__ == "__main__":
    evaluate()
