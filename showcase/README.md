# Star Tracker CNN — Showcase

This folder contains labeled inputs and model outputs for 3 validation sky images (samples 0, 5, 12).

---

## inputs/

| File | Description |
|---|---|
| `input_sample00/05/12.png` | Raw 256×256 grayscale sky images from NASA DSS2 Red Survey. These are the images fed into each model. Pixel values normalized to [0, 1]. |
| `groundtruth_sample00/05/12.png` | Binary segmentation maps hand-labeled from Gaia DR3. White dots = star regions (3-px circles). These are the targets the models are trained to predict. |

---

## outputs/

Each model receives the corresponding `input_sampleXX.png` and outputs a **segmentation probability map** — brighter pixels = higher confidence of a star.

| File | Model | Params | Notes |
|---|---|---|---|
| `unet_sampleXX.png` | UNet | 481,745 | Largest model. Best F1 (0.928), lowest centroid error (0.39 px). |
| `mobileunet_sampleXX.png` | MobileUNet | 254,969 | Uses depthwise-separable convolutions. Fastest to train, lowest F1 (0.884). |
| `elunet_sampleXX.png` | ELUNet | 59,625 | Smallest model (232 KB). F1 = 0.910. Designed for embedded CubeSat deployment. |

---

## full_grid.png

Side-by-side comparison of all 3 samples across all columns:
`Sky Image → Ground Truth → UNet → MobileUNet → ELUNet`

---

## How outputs are generated

1. Input image is loaded as `(1, 256, 256)` float32 tensor
2. Passed through the trained model
3. Output channel 0 = sigmoid segmentation probability map `[0, 1]`
4. Bright (hot colormap) = star detected, dark = background

Centroid extraction uses center-of-mass on blobs above threshold 0.5.
