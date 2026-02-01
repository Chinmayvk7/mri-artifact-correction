"""
PyTorch Dataset for FastMRI with Artifact Generation

FILE: src/data/dataset.py

We will generate artifacts on-the-go instead of pre-saving them because:

    1. MORE VARIETY   — each epoch sees DIFFERENT random artifacts on the same slice
                        → the network learns to handle many artifact patterns, not just one
    2. DISK SPACE     — no need to store 100 corrupted copies on disk
    3. FLEXIBILITY    — change artifact severity without re-generating anything
    4. GENERALIZATION — exposure to more patterns = better real-world performance

HOW IT CONNECTS TO OUR EXISTING CODE:
    • Uses our FastMRILoader  → loads .h5 files, returns (kspace, image)
    • Uses our MultiArtifactSimulator → corrupts k-space, returns dict

EXPECTED INTERFACES (must match our existing files):

    loader = FastMRILoader(data_path)
        loader.file_list                     → list of .h5 filenames
        loader.get_file_info(file_idx)       → dict with 'num_slices' key
        loader.load_slice(file_idx, slice_idx) → (kspace [H,W] complex, image [H,W] real)

    sim = MultiArtifactSimulator(acceleration_factor=4, num_spikes=10)
        sim.apply(kspace)                    → dict with 'corrupted_kspace' key
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys
from typing import Tuple, Optional

# so imports work regardless of where you run from 
_THIS   = os.path.dirname(os.path.abspath(__file__))   # data/
sys.path.insert(0, _THIS)

from fastmri_loader import FastMRILoader
from artifacts      import MultiArtifactSimulator



# HELPER FUNCTIONS


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Min-max normalize to [0, 1].

    We normalize each slice independently (not globally) because different
    MRI slices have wildly different intensity ranges depending on anatomy.


        image: 2D array, any range.
    Returns:
        Normalized array in [0, 1].
    """
    lo, hi = image.min(), image.max()
    if hi - lo == 0:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def center_crop(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Center-crop (or zero-pad if too small) to target size.

    We center crop instead of random crop:
        MRI anatomy is centered — random crop might chop off important tissue.

        image:    2D array [H, W].
        target_h: Desired height.
        target_w: Desired width.
    Returns:
        Cropped / padded array [target_h, target_w].
    """
    h, w = image.shape

    # pad if necessary to maintain the spatial shape.
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

    # crop
    sh = (h - target_h) // 2
    sw = (w - target_w) // 2
    return image[sh:sh + target_h, sw:sw + target_w].copy()



# DATASET CLASS


class FastMRIArtifactDataset(Dataset):
    """
    PyTorch Dataset that loads fastMRI slices and applies artifacts on the go.

    Each call to __getitem__ returns:
        {
            'corrupted' : FloatTensor [1, H, W]   ← input to the network
            'clean'     : FloatTensor [1, H, W]   ← ground truth target
            'kspace_clean'     : np array (complex) ← for visualization only
            'kspace_corrupted' : np array (complex) ← for visualization only
        }

    USAGE:
        ds = FastMRIArtifactDataset(
            data_path='data/raw/knee_singlecoil_val',
            num_slices=100,
            image_size=(256, 256),
        )
        sample = ds[0]
        print(sample['corrupted'].shape)   # torch.Size([1, 256, 256])
    """

    def __init__(
        self,
        data_path:           str,
        num_slices:          int            = 100,
        image_size:          Tuple[int,int] = (256, 256),
        acceleration_factor: int            = 4,
        num_spikes:          int            = 10,
        skip_edge_fraction:  float          = 0.1,
    ):
        """
        
            data_path:           Directory containing fastMRI .h5 files.
            num_slices:          How many slices total to put in this dataset.
                                 80 for train, 20 for val is a good split.
            image_size:          (H, W) after center cropping. ( 256×256 )
                                 
            acceleration_factor: Undersampling rate (4 = keep ¼ of k-space lines).
            num_spikes:          How many random spike artifacts to put.
            skip_edge_fraction:  Skip top/bottom fraction of each volume
        
        """
        print(f"\n  Initialising dataset  →  {data_path}")

        self.image_size = image_size

        # existing classes
        self.loader      = FastMRILoader(data_path)
        self.artifact_sim = MultiArtifactSimulator(
            acceleration_factor=acceleration_factor,
            num_spikes=num_spikes,
        )

        # build flat index: (file_idx, slice_idx) pairs
        self.slice_index = self._build_index(num_slices, skip_edge_fraction)

        print(f"  Dataset ready:  {len(self.slice_index)} slices, "
              f"size={image_size}, accel={acceleration_factor}×, "
              f"spikes={num_spikes}")

    # index building
    def _build_index(self, num_slices: int, skip_edge: float) -> list:
        """
        Walk through available volumes, pick slices (skipping edges).

        Returns a flat list so that dataset[i] can access (file, slice)
        in O(1).
        """
        index = []
        for file_idx in range(len(self.loader.file_list)):
            info        = self.loader.get_file_info(file_idx)
            total       = info['num_slices']
            start       = int(total * skip_edge)
            end         = int(total * (1 - skip_edge))

            for slice_idx in range(start, end):
                index.append((file_idx, slice_idx))
                if len(index) >= num_slices:
                    return index
        return index

    # required Dataset interface
    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> dict:
        """
        Load one slice, corrupt it, return (corrupted, clean) pair.

        Step-by-step:
            1. Look up which file & slice this index maps to
            2. Load clean k-space + magnitude image from disk
            3. Apply random artifacts to k-space wehre new artifacts for evry call.
            4. Reconstruct corrupted image via IFFT
            5. Normalize both images using the CLEAN image's range
               (so the intensity scales match — critical for fair loss computation)
            6. Center-crop both to image_size
            7. Convert to PyTorch tensors  [1, H, W]

    
            idx: integer index into slice_index.

        Returns:
            dict with 'corrupted', 'clean' tensors + raw k-space arrays.
        """
        file_idx, slice_idx = self.slice_index[idx]

        # 1) load from disk
        kspace_clean, image_clean = self.loader.load_slice(file_idx, slice_idx)

        # 2) corrupt k-space
        result           = self.artifact_sim.apply(kspace_clean)
        kspace_corrupted = result['corrupted_kspace']

        # 3) reconstruct corrupted image
        #    ifftshift → ifft2 → fftshift → abs  (same convention as fft.py)
        image_corrupted = np.abs(
            np.fft.fftshift(
                np.fft.ifft2(
                    np.fft.ifftshift(kspace_corrupted)
                )
            )
        )

        # 4) normalize BOTH images using the CLEAN image's range
        # This keeps them on the same scale — essential for loss computation.
        lo  = image_clean.min()
        hi  = image_clean.max()
        rng = hi - lo if (hi - lo) > 0 else 1.0

        clean_norm     = ((image_clean     - lo) / rng).astype(np.float32)
        corrupted_norm = np.clip(
            ((image_corrupted - lo) / rng).astype(np.float32),
            0.0, 1.0                        # corruption can push values outside [0,1]
        )

        # 5) center crop
        clean_crop     = center_crop(clean_norm,     *self.image_size)
        corrupted_crop = center_crop(corrupted_norm, *self.image_size)

        # 6) → PyTorch tensors  [1, H, W]   (channel-first format)
        return {
            'corrupted':        torch.from_numpy(corrupted_crop).unsqueeze(0),
            'clean':            torch.from_numpy(clean_crop).unsqueeze(0),
            'kspace_clean':     kspace_clean,      # raw complex — for viz only
            'kspace_corrupted': kspace_corrupted,
        }



# Create train + val datasets with proper split


def create_train_val_datasets(
    data_path:           str,
    total_slices:        int            = 120,
    val_fraction:        float          = 0.2,
    image_size:          Tuple[int,int] = (256, 256),
    acceleration_factor: int            = 4,
    num_spikes:          int            = 10,
) -> Tuple[FastMRIArtifactDataset, FastMRIArtifactDataset]:
    """
    Build train and validation datasets.

    We do NOT randomly shuffle slices from the same volume into both sets —
    that would be data leakage (slices from the same scan are very similar).
    Instead the first N slices go to train, the next M go to val.

    
        data_path:           Path to fastMRI .h5 directory.
        total_slices:        Train + val combined.
        val_fraction:        Fraction reserved for validation.
        image_size:          Cropped size.
        acceleration_factor: Undersampling rate.
        num_spikes:          Spike count.

    Returns:
        (train_dataset, val_dataset)
    """
    num_train = int(total_slices * (1 - val_fraction))
    num_val   = total_slices - num_train

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
    # Load train+val slices, then keep only the val portion
    full_ds = FastMRIArtifactDataset(
        data_path=data_path,
        num_slices=num_train + num_val,
        image_size=image_size,
        acceleration_factor=acceleration_factor,
        num_spikes=num_spikes,
    )
    # Trim: keep only slices [num_train : num_train+num_val]
    full_ds.slice_index = full_ds.slice_index[num_train : num_train + num_val]
    print(f"  Validation uses slices #{num_train}–#{num_train + num_val - 1}")

    return train_ds, full_ds



# SELF-TEST
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    print("=" * 58)
    print("  Testing FastMRI Dataset")
    print("=" * 58)

    # auto-detect data directory
    candidates = [
        'data/singlecoil_val',
        'data/singlecoil_train',
        'data/singlecoil_test',
        'data/raw/knee_singlecoil_val',
        '../data/singlecoil_val',
        '../../data/singlecoil_val',
    ]
    data_path = None
    for c in candidates:
        if os.path.isdir(c):
            data_path = c
            break

    if data_path is None:
        print("\n  ✗ Could not find data directory automatically.")
        print("    Edit 'candidates' list above or set data_path manually.")
        sys.exit(1)

    print(f"  Found data at:  {data_path}")

    # small test dataset
    ds = FastMRIArtifactDataset(
        data_path=data_path,
        num_slices=8,
        image_size=(256, 256),
    )

    # single sample
    s = ds[0]
    print(f"\n  Single sample:")
    print(f"    corrupted  : {s['corrupted'].shape}  "
          f"range [{s['corrupted'].min():.3f}, {s['corrupted'].max():.3f}]")
    print(f"    clean      : {s['clean'].shape}  "
          f"range [{s['clean'].min():.3f}, {s['clean'].max():.3f}]")

    # DataLoader batch
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    print(f"\n  Batch (size 4):")
    print(f"    corrupted  : {batch['corrupted'].shape}")
    print(f"    clean      : {batch['clean'].shape}")

    # visualise
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Dataset Self-Test — Corrupted vs Clean', fontsize=15)
    for i in range(4):
        axes[0, i].imshow(batch['corrupted'][i, 0].numpy(), cmap='gray')
        axes[0, i].set_title(f'Corrupted {i+1}'); axes[0, i].axis('off')
        axes[1, i].imshow(batch['clean'][i, 0].numpy(),     cmap='gray')
        axes[1, i].set_title(f'Clean {i+1}');     axes[1, i].axis('off')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(_THIS), '..', 'outputs', 'figures')
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, 'dataset_test.png'), dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved visualisation → outputs/figures/dataset_test.png")
    print("  ✓ All dataset tests passed!")
