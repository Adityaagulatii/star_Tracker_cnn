"""
prepare_data.py  —  download sky images, generate seg maps, visualize samples.

Run:  python prepare_data.py
"""

import random, warnings, time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, label
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

IMG_SIZE    = 256      # pixels per side
FOV_ARCMIN  = 6.0      # sky field width in arcminutes
N_TRAIN     = 200      # training images
N_VAL       = 50       # validation images
DATA_DIR    = Path("data")
SEED        = 42

THRESHOLD   = 0.55     # brightness cutoff for star detection
MIN_SIZE    = 3        # minimum blob size in pixels (smaller = noise)
MAX_SIZE    = 80       # maximum blob size in pixels (larger = galaxy/artifact)
STAR_RADIUS = 3        # radius of circle drawn around each star center

random.seed(SEED)
np.random.seed(SEED)

# ── Step 1: Download a sky image ──────────────────────────────────────────────

def download_image(ra, dec, retries=3):
    """Download a DSS2 Red image patch centred on (ra, dec). Returns (256,256)
    float32 array normalised to [0,1], or None on failure."""
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    for attempt in range(retries):
        try:
            result = SkyView.get_images(
                position=coord,
                survey=['DSS2 Red'],
                pixels=str(IMG_SIZE),
                width=FOV_ARCMIN * u.arcmin,
                height=FOV_ARCMIN * u.arcmin,
            )
            break
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2)

    if not result:
        return None

    image = result[0][0].data.squeeze().astype(np.float32)
    if image.ndim != 2:
        return None

    p1, p99 = np.percentile(image, 1), np.percentile(image, 99)
    if p99 <= p1:
        return None

    return np.clip((image - p1) / (p99 - p1), 0, 1).astype(np.float32)


# ── Step 2: Generate segmentation map from the image ─────────────────────────

def make_seg(image):
    """Detect stars via thresholding and return a binary seg map.

    Pipeline:
      1. Gaussian blur  — smooths pixel noise
      2. Threshold      — bright blobs become candidates
      3. Size filter    — remove noise specks and large artifacts
      4. Draw circles   — paint STAR_RADIUS circles at each centre
    """
    smoothed        = gaussian_filter(image, sigma=1.0)
    binary          = (smoothed > THRESHOLD).astype(np.int32)
    labeled, n_blobs = label(binary)

    H, W = image.shape
    Y, X = np.ogrid[:H, :W]
    seg  = np.zeros((H, W), dtype=np.float32)

    for i in range(1, n_blobs + 1):
        mask = labeled == i
        size = mask.sum()
        if size < MIN_SIZE or size > MAX_SIZE:
            continue
        ys, xs = np.where(mask)
        cy, cx  = ys.mean(), xs.mean()
        seg[np.sqrt((X - cx)**2 + (Y - cy)**2) <= STAR_RADIUS] = 1.0

    return seg


# ── Step 3: Download + label N images for one split ──────────────────────────

def generate_dataset(n, split_dir):
    """Download n images and save image/seg pairs to split_dir.
    Resumes automatically if some files already exist."""
    split_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(split_dir.glob("image_*.npy")))
    if existing >= n:
        print(f"{split_dir.name}: {existing} samples already present, skipping.")
        return

    print(f"\nDownloading {n} samples -> {split_dir}  (resuming from {existing})")
    saved    = existing
    attempts = 0

    with tqdm(total=n, initial=existing, unit="img") as pbar:
        while saved < n:
            attempts += 1
            if attempts > n * 8:
                print(f"Warning: stopped at {saved}/{n} after too many failed attempts.")
                break

            ra, dec = random.uniform(0, 360), random.uniform(-60, 60)
            image   = download_image(ra, dec)
            if image is None:
                continue

            seg = make_seg(image)
            np.save(split_dir / f"image_{saved:04d}.npy", image)
            np.save(split_dir / f"seg_{saved:04d}.npy",   seg)
            saved += 1
            pbar.update(1)
            time.sleep(0.1)

    print(f"Done — {saved} samples in {split_dir}")


# ── Step 4: Visualise two random samples ─────────────────────────────────────

def show_samples(data_dir=DATA_DIR / "train", n=2, out="sample.png"):
    """Save a side-by-side figure of n random image/seg pairs."""
    paths = sorted((data_dir).glob("image_*.npy"))
    if not paths:
        print("No images found — run generate_dataset first.")
        return

    picks = random.sample(paths, min(n, len(paths)))

    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    fig.patch.set_facecolor('#0d0d0d')
    axes[0, 0].set_title("Sky Image",        color='white', fontsize=13, pad=8)
    axes[0, 1].set_title("Segmentation Map", color='white', fontsize=13, pad=8)

    for row, path in enumerate(picks):
        idx   = path.stem.split("_")[1]
        image = np.load(data_dir / f"image_{idx}.npy")
        seg   = np.load(data_dir / f"seg_{idx}.npy")

        axes[row, 0].imshow(image, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"Sample {idx}", color='white', fontsize=10,
                                rotation=0, labelpad=55, va='center')

        axes[row, 1].imshow(seg, cmap="hot", origin="lower", vmin=0, vmax=1)
        axes[row, 1].axis("off")

    plt.suptitle("Star Detection — Ground Truth Labels", color='white',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset(N_TRAIN, DATA_DIR / "train")
    generate_dataset(N_VAL,   DATA_DIR / "val")
    show_samples()
