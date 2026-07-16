"""
Generate visual comparison: FP32 vs INT8 (dynamic) vs LSQ INT8.
Saves: results/model_comparison.png

Usage: python -m quantize.show_results
"""

import sys, os, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy.ndimage import label, center_of_mass, distance_transform_edt

from model              import ELUNet
from quantize.lsq       import replace_conv2d_with_lsq
from quantize.static_int8 import make_int8

FP32_CKPT  = "checkpoints/elunet_fp32.pth"
LSQ_CKPT   = "checkpoints/elunet_lsq.pth"
SIM_VAL    = "C:/Users/adity/Downloads/star_tracker_quant/data/sim_val.npz"
OUT_DIR    = "results"
SEG_THRESH = 0.5


def model_size_kb(m):
    f = tempfile.mktemp(suffix=".pth")
    torch.save(m.state_dict(), f)
    kb = os.path.getsize(f) / 1024
    os.remove(f)
    return kb


def trilaterate(seg, dist, radius=7):
    binary     = (seg > SEG_THRESH).astype(np.float32)
    labeled, n = label(binary)
    if n == 0:
        return []
    cents = []
    for i in range(1, n + 1):
        blob      = labeled == i
        if blob.sum() < 2: continue
        dist_blob = np.where(blob, dist, -np.inf)
        r0, c0    = np.unravel_index(np.argmax(dist_blob), dist_blob.shape)
        H, W      = binary.shape
        rs, re    = max(0, r0-radius), min(H, r0+radius+1)
        cs, ce    = max(0, c0-radius), min(W, c0+radius+1)
        pseg      = binary[rs:re, cs:ce]
        pdist     = dist[rs:re, cs:ce]
        rows, cols = np.where(pseg > 0)
        if len(rows) < 3:
            cents.append((float(c0), float(r0))); continue
        ar = rows+rs; ac = cols+cs; ds = pdist[rows, cols]
        pos = ds > 0
        if not pos.any():
            cents.append((float(c0), float(r0))); continue
        ref      = np.where(pos)[0][np.argmin(ds[pos])]
        y0_, x0_ = float(ar[ref]), float(ac[ref]); d0_ = float(ds[ref])
        m        = np.arange(len(rows)) != ref
        yi = ar[m].astype(float); xi = ac[m].astype(float); di = ds[m]
        A  = 2 * np.column_stack([xi-x0_, yi-y0_])
        b  = (xi**2-x0_**2) + (yi**2-y0_**2) - (di**2-d0_**2)
        if A.shape[0] >= 2:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx = float(np.clip(x0_+res[0], 0, W-1))
            cy = float(np.clip(y0_+res[1], 0, H-1))
        else:
            cx, cy = float(c0), float(r0)
        cents.append((cx, cy))
    return cents


def infer(model, img_np):
    t = torch.from_numpy(img_np).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(t)
    ms = (time.perf_counter() - t0) * 1000
    return out[0,0].numpy(), out[0,1].numpy(), ms


def main():
    # ---- Load models ----
    fp32 = ELUNet()
    fp32.load_state_dict(torch.load(FP32_CKPT, map_location="cpu", weights_only=True))
    fp32.eval()

    int8 = make_int8(FP32_CKPT)
    int8.eval()

    lsq = replace_conv2d_with_lsq(ELUNet(), n_bits=8)
    lsq.load_state_dict(torch.load(LSQ_CKPT, map_location="cpu", weights_only=True))
    lsq.eval()

    params   = sum(p.numel() for p in fp32.parameters())
    fp32_kb  = model_size_kb(fp32)
    int8_kb  = params / 1024
    lsq_kb   = model_size_kb(lsq)

    print(f"Parameters  : {params:,}")
    print(f"FP32 file   : {fp32_kb:.1f} KB")
    print(f"INT8 weights: ~{int8_kb:.1f} KB  (1 byte/param)")
    print(f"LSQ file    : {lsq_kb:.1f} KB")

    models = {
        f"FP32\n{fp32_kb:.0f} KB  |  baseline": fp32,
        f"INT8\n~{int8_kb:.0f} KB  |  4x smaller weights": int8,
        f"LSQ INT8\n{lsq_kb:.0f} KB  |  learned quant": lsq,
    }

    # ---- Load val samples ----
    d = np.load(SIM_VAL)
    imgs = d["images"]; segs = d["segs"]
    sample_idx = [0, 4, 8, 14]
    samples = []
    for i in sample_idx:
        seg_np   = segs[i,0]
        labeled, n = label(seg_np > 0.5)
        gt_cents = [(float(cx), float(cy))
                    for cy, cx in [center_of_mass(seg_np, labels=labeled, index=j)
                                   for j in range(1, n+1)]]
        samples.append((imgs[i], seg_np, gt_cents))

    n_samples = len(samples)
    n_models  = len(models)

    # ---- Big figure: 3 panels per sample (input+GT | seg maps | centroid overlay) ----
    fig = plt.figure(figsize=(5 * (1 + n_models), 4.5 * n_samples), facecolor="#0d0d0d")

    for row, (img_np, seg_gt, gt_cents) in enumerate(samples):
        row_axes = []
        for col in range(1 + n_models):
            ax = fig.add_subplot(n_samples, 1 + n_models,
                                 row * (1 + n_models) + col + 1)
            ax.set_facecolor("black"); ax.axis("off")
            row_axes.append(ax)

        # col 0: input + gt centroids
        ax0 = row_axes[0]
        ax0.imshow(img_np[0], cmap="gray", vmin=0, vmax=1, origin="upper")
        for (gx, gy) in gt_cents:
            ax0.add_patch(Circle((gx, gy), 6, fill=False,
                                  edgecolor="red", linewidth=1.5))
        ax0.set_title("Input\n(red = GT stars)", color="white", fontsize=9) if row == 0 else None

        for mi, (name, model) in enumerate(models.items()):
            ax = row_axes[1 + mi]
            seg, dist, ms = infer(model, img_np)
            cents = trilaterate(seg, dist)

            # overlay: raw image + coloured seg heatmap + centroid circles
            ax.imshow(img_np[0], cmap="gray", vmin=0, vmax=1, origin="upper")
            ax.imshow(seg, cmap="hot", alpha=0.45, vmin=0, vmax=1, origin="upper")

            # predicted centroids
            for (cx, cy) in cents:
                ax.add_patch(Circle((cx, cy), 5, fill=False,
                                     edgecolor="lime", linewidth=2))
            # GT centroids
            for (gx, gy) in gt_cents:
                ax.add_patch(Circle((gx, gy), 7, fill=False,
                                     edgecolor="red", linewidth=1.5,
                                     linestyle="--"))

            label_str = (f"Det:{len(cents)}  GT:{len(gt_cents)}  {ms:.1f}ms"
                         if len(cents) else f"None  GT:{len(gt_cents)}  {ms:.1f}ms")
            ax.text(3, 12, label_str, color="yellow", fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6))
            if row == 0:
                ax.set_title(name, color="white", fontsize=9)

    legend_text = ("lime circle = predicted centroid    red dashed = ground truth\n"
                   "Heatmap = segmentation probability  |  Training: 200 sim images, 40 epochs")
    fig.text(0.5, 0.01, legend_text, ha="center", color="#aaaaaa", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 1], pad=0.6)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "model_comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\nSaved: {out}")

    # ---- Print centroid table ----
    print(f"\n{'Sample':<8}", end="")
    for name in models: print(f"  {name.split(chr(10))[0]:<6}", end="")
    print("  GT")
    for i, (img_np, seg_gt, gt_cents) in enumerate(samples):
        print(f"  S{i:<5}", end="")
        for name, model in models.items():
            seg, dist, ms = infer(model, img_np)
            print(f"  {len(trilaterate(seg, dist)):<6}", end="")
        print(f"  {len(gt_cents)}")

    print(f"\nModel size summary:")
    print(f"  FP32 on disk  : {model_size_kb(fp32):.1f} KB")
    print(f"  INT8 weights  : ~{params/1024:.1f} KB  (true int8 footprint)")
    print(f"  Size reduction: {model_size_kb(fp32) / (params/1024):.1f}x")


if __name__ == "__main__":
    main()
