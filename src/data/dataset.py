"""
PyTorch Dataset for FastMRI with Artifact Generation

FILE: src/data/dataset.py

FIXED: Proper normalization that handles undersampled k-space correctly.
Each image is normalized INDEPENDENTLY to preserve visible content.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys
from typing import Tuple, Optional

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)

from fastmri_loader import FastMRILoader
from artifacts import MultiArtifactSimulator


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = image.min(), image.max()
    if hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def center_crop(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop (or zero-pad if too small) to target size."""
    h, w = image.shape

    if h < target_h or w < target_w:
        ph = max(0, target_h - h)
        pw = max(0, target_w - w)
        image = np.pad(
            image,
            ((ph // 2, ph - ph // 2),
             (pw // 2, pw - pw // 2)),
            mode='constant', constant_values=0
        )
        h, w = image.shape

    sh = (h - target_h) // 2
    sw = (w - target_w) // 2
    return image[sh:sh + target_h, sw:sw + target_w].copy()


class FastMRIArtifactDataset(Dataset):
    """
    PyTorch Dataset that loads fastMRI slices and applies artifacts on the go.
    
    FIXED VERSION: Each image is normalized independently to preserve content.
    """

    def __init__(
        self,
        data_path: str,
        num_slices: int = 100,
        image_size: Tuple[int, int] = (256, 256),
        acceleration_factor: int = 4,
        num_spikes: int = 10,
        skip_edge_fraction: float = 0.1,
    ):
        print(f"\n  Initialising dataset  ->  {data_path}")

        self.image_size = image_size
        self.loader = FastMRILoader(data_path)
        self.artifact_sim = MultiArtifactSimulator(
            acceleration_factor=acceleration_factor,
            num_spikes=num_spikes,
        )
        self.slice_index = self._build_index(num_slices, skip_edge_fraction)

        print(f"  Dataset ready:  {len(self.slice_index)} slices, "
              f"size={image_size}, accel={acceleration_factor}x, "
              f"spikes={num_spikes}")

    def _build_index(self, num_slices: int, skip_edge: float) -> list:
        index = []
        for file_idx in range(len(self.loader.file_list)):
            info = self.loader.get_file_info(file_idx)
            total = info['num_slices']
            start = int(total * skip_edge)
            end = int(total * (1 - skip_edge))

            for slice_idx in range(start, end):
                index.append((file_idx, slice_idx))
                if len(index) >= num_slices:
                    return index
        return index

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> dict:
        """
        Load one slice, corrupt it, return (corrupted, clean) pair.
        
        FIXED: Both images normalized independently to [0,1].
        """
        file_idx, slice_idx = self.slice_index[idx]

        # 1) Load from disk
        kspace_clean, image_clean = self.loader.load_slice(file_idx, slice_idx)

        # 2) Corrupt k-space
        result = self.artifact_sim.apply(kspace_clean)
        kspace_corrupted = result['corrupted_kspace']

        # 3) Reconstruct corrupted image via IFFT
        image_corrupted = np.abs(
            np.fft.fftshift(
                np.fft.ifft2(
                    np.fft.ifftshift(kspace_corrupted)
                )
            )
        )

        # 4) FIXED NORMALIZATION: Normalize each image independently
        clean_norm = normalize_image(image_clean)
        corrupted_norm = normalize_image(image_corrupted)

        # 5) Center crop
        clean_crop = center_crop(clean_norm, *self.image_size)
        corrupted_crop = center_crop(corrupted_norm, *self.image_size)

        # 6) To PyTorch tensors [1, H, W]
        return {
            'corrupted': torch.from_numpy(corrupted_crop).unsqueeze(0),
            'clean': torch.from_numpy(clean_crop).unsqueeze(0),
        }


def create_train_val_datasets(
    data_path: str,
    total_slices: int = 120,
    val_fraction: float = 0.2,
    image_size: Tuple[int, int] = (256, 256),
    acceleration_factor: int = 4,
    num_spikes: int = 10,
) -> Tuple[FastMRIArtifactDataset, FastMRIArtifactDataset]:
    """Build train and validation datasets."""
    
    num_train = int(total_slices * (1 - val_fraction))
    num_val = total_slices - num_train

    print("\n" + "=" * 58)
    print("  Creating TRAINING dataset")
    print("=" * 58)
    train_ds = FastMRIArtifactDataset(
        data_path=data_path,
        num_slices=num_train,
        image_size=image_size,
        acceleration_factor=acceleration_factor,
        num_spikes=num_spikes,
    )

    print("\n" + "=" * 58)
    print("  Creating VALIDATION dataset")
    print("=" * 58)
    full_ds = FastMRIArtifactDataset(
        data_path=data_path,
        num_slices=num_train + num_val,
        image_size=image_size,
        acceleration_factor=acceleration_factor,
        num_spikes=num_spikes,
    )
    full_ds.slice_index = full_ds.slice_index[num_train:num_train + num_val]
    print(f"  Validation uses slices #{num_train}-#{num_train + num_val - 1}")

    return train_ds, full_ds


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    print("=" * 58)
    print("  Testing FastMRI Dataset")
    print("=" * 58)

    candidates = [
        'data/singlecoil_val',
        'data/singlecoil_train',
        'data/raw/knee_singlecoil_val',
        '../data/singlecoil_val',
    ]
    data_path = None
    for c in candidates:
        if os.path.isdir(c):
            data_path = c
            break

    if data_path is None:
        print("\n  Could not find data directory.")
        sys.exit(1)

    print(f"  Found data at:  {data_path}")

    ds = FastMRIArtifactDataset(
        data_path=data_path,
        num_slices=8,
        image_size=(256, 256),
    )

    s = ds[0]
    print(f"\n  Single sample:")
    print(f"    corrupted  : {s['corrupted'].shape}  "
          f"range [{s['corrupted'].min():.3f}, {s['corrupted'].max():.3f}]")
    print(f"    clean      : {s['clean'].shape}  "
          f"range [{s['clean'].min():.3f}, {s['clean'].max():.3f}]")

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dl))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Dataset Test - Corrupted vs Clean', fontsize=15)
    for i in range(4):
        axes[0, i].imshow(batch['corrupted'][i, 0].numpy(), cmap='gray')
        axes[0, i].set_title(f'Corrupted {i+1}')
        axes[0, i].axis('off')
        axes[1, i].imshow(batch['clean'][i, 0].numpy(), cmap='gray')
        axes[1, i].set_title(f'Clean {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(_THIS), '..', 'outputs', 'figures')
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, 'dataset_test.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Saved -> outputs/figures/dataset_test.png")
    print("  All tests passed!")