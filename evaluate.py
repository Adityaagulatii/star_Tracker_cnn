"""
evaluate.py — test all three models on the validation set.

Usage:  python evaluate.py
"""

import time
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label, center_of_mass
from torch.utils.data import DataLoader

from dataset import StarDataset
from model   import UNet, MobileUNet, ELUNet

MODELS       = {'unet': UNet, 'mobileunet': MobileUNet, 'elunet': ELUNet}
THRESHOLD    = 0.5    # probability cutoff to call a pixel a star
MATCH_RADIUS = 5      # pixels — predicted star must be within this to count as TP


# ── Centroid extraction ───────────────────────────────────────────────────────

def get_centroids(seg_map):
    """Find star centres from a binary-ish seg map using connected components."""
    binary         = (seg_map > THRESHOLD).astype(np.int32)
    labeled, n     = label(binary)
    cents = []
    for i in range(1, n + 1):
        cy, cx = center_of_mass(binary, labels=labeled, index=i)
        cents.append((cx, cy))
    return cents


def match(pred_cents, true_cents, radius):
    """Greedy nearest-neighbour matching. Returns TP, FP, FN, error list."""
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
            tp += 1
            matched.add(best_j)
            errors.append(best_d)
    fp = len(pred_cents) - tp
    fn = len(true_cents)  - tp
    return tp, fp, fn, errors


# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate(model_name, device):
    ckpt = Path(f"checkpoints/{model_name}_best.pth")
    if not ckpt.exists():
        return None

    model = MODELS[model_name]().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loader = DataLoader(StarDataset('data/val'), batch_size=1,
                        shuffle=False, num_workers=0)

    tp_tot = fp_tot = fn_tot = 0
    all_errors, times = [], []

    with torch.no_grad():
        for images, segs in loader:
            images = images.to(device)

            t0 = time.perf_counter()
            pred = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

            pred_np = pred[0, 0].cpu().numpy()
            true_np = segs[0, 0].numpy()

            pred_c = get_centroids(pred_np)
            true_c = get_centroids(true_np)

            tp, fp, fn, errs = match(pred_c, true_c, MATCH_RADIUS)
            tp_tot += tp
            fp_tot += fp
            fn_tot += fn
            all_errors.extend(errs)

    precision = tp_tot / (tp_tot + fp_tot + 1e-8)
    recall    = tp_tot / (tp_tot + fn_tot + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    mean_err  = float(np.mean(all_errors)) if all_errors else float('nan')
    mean_time = float(np.mean(times[5:])) if len(times) > 5 else float(np.mean(times))
    params    = sum(p.numel() for p in model.parameters())

    return dict(f1=f1, precision=precision, recall=recall,
                mean_err=mean_err, time_ms=mean_time, params=params)


# ── Visual predictions on 2 random val images ─────────────────────────────────

def save_img(arr, path):
    """Save a single 2D array as a clean black/white PNG, no borders."""
    binary = (arr > THRESHOLD).astype(np.float32)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('black')
    ax.imshow(binary, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    plt.savefig(path, dpi=150, bbox_inches='tight',
                pad_inches=0, facecolor='black')
    plt.close()


def visualise(device):
    val_paths = sorted(Path("data/val").glob("image_*.npy"))
    random.seed(99)
    picks = random.sample(val_paths, min(2, len(val_paths)))

    loaded = {}
    for name, Cls in MODELS.items():
        ckpt = Path(f"checkpoints/{name}_best.pth")
        if not ckpt.exists():
            continue
        m = Cls().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        loaded[name] = m

    for path in picks:
        idx   = path.stem.split("_")[1]
        image = np.load(path)
        seg   = np.load(Path("data/val") / f"seg_{idx}.npy")
        img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

        # Raw image
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('black')
        ax.imshow(image, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        plt.savefig(f"val_{idx}_input.png", dpi=150,
                    bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()

        # Ground truth
        save_img(seg, f"val_{idx}_groundtruth.png")

        # Each model
        for name, m in loaded.items():
            with torch.no_grad():
                pred = m(img_t)[0, 0].cpu().numpy()
            save_img(pred, f"val_{idx}_{name}.png")

        print(f"Saved 5 images for sample {idx}")


# ── Unseen image test ────────────────────────────────────────────────────────

def unseen_test(device, out="unseen_test.png"):
    """Download one brand-new sky image and show all three model outputs."""
    from prepare_data import download_image, make_seg
    import random as _r

    print("\nDownloading unseen test image...")
    _r.seed(2025)
    image = None
    while image is None:
        ra, dec = _r.uniform(0, 360), _r.uniform(-60, 60)
        image   = download_image(ra, dec)

    seg_gt = make_seg(image)
    img_t  = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, len(MODELS) + 2, figsize=(5 * (len(MODELS) + 2), 5))
    fig.patch.set_facecolor('#0d0d0d')

    # Raw input
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('black')
    ax.imshow(image, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    plt.savefig("unseen_input.png", dpi=150,
                bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

    # Ground truth
    save_img(seg_gt, "unseen_groundtruth.png")

    # Each model
    for name, Cls in MODELS.items():
        ckpt = Path(f"checkpoints/{name}_best.pth")
        if not ckpt.exists():
            continue
        m = Cls().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        with torch.no_grad():
            pred = m(img_t)[0, 0].cpu().numpy()
        save_img(pred, f"unseen_{name}.png")

    print("Saved: unseen_input.png  unseen_groundtruth.png  unseen_unet.png  unseen_mobileunet.png  unseen_elunet.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating on {device}\n")

    # Metrics table
    header = f"{'Model':<14}{'F1':>7}{'Err(px)':>10}{'ms/img':>9}"
    print(header)
    print('-' * len(header))

    for name in MODELS:
        r = evaluate(name, device)
        if r is None:
            print(f"{name:<14}  no checkpoint found")
            continue
        print(f"{name:<14}"
              f"{r['f1']:>7.3f}"
              f"{r['mean_err']:>10.2f}"
              f"{r['time_ms']:>9.1f}")

    # Visual output on val samples
    visualise(device)

    # Unseen image test
    unseen_test(device)
