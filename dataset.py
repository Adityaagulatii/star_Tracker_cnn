import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import distance_transform_edt


class StarDataset(Dataset):
    """
    Loads DSS2 star images from data/train or data/val.
    Returns (image, seg, dist_map) — dist_map computed on-the-fly from seg.

    Shapes:
        image    : (1, H, W)  float32  normalized [0, 1]
        seg      : (1, H, W)  float32  binary (1 = star pixel)
        dist_map : (1, H, W)  float32  distance transform in pixels
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.indices  = sorted(
            int(f.stem.split('_')[1])
            for f in self.data_dir.glob('image_*.npy')
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        image = np.load(self.data_dir / f'image_{idx:04d}.npy').astype(np.float32)
        seg   = np.load(self.data_dir / f'seg_{idx:04d}.npy').astype(np.float32)

        dist_map = distance_transform_edt(seg).astype(np.float32)

        image    = torch.from_numpy(image).unsqueeze(0)
        seg      = torch.from_numpy(seg).unsqueeze(0)
        dist_map = torch.from_numpy(dist_map).unsqueeze(0)
        return image, seg, dist_map
