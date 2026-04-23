import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class StarDataset(Dataset):
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
        image = np.load(self.data_dir / f'image_{idx:04d}.npy')
        seg   = np.load(self.data_dir / f'seg_{idx:04d}.npy')

        # (H, W) -> (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        seg   = torch.from_numpy(seg).unsqueeze(0)
        return image, seg
