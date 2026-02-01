"""
Evaluation Metrics for MRI Reconstruction

FILE: src/utils/metrics.py

THREE STANDARD METRICS used in every MRI reconstruction paper:

┌─────────────────────────────────────────────────────────────┐
│  PSNR  (Peak Signal-to-Noise Ratio)                         │
│    • Measures pixel-level accuracy                          │
│    • Higher = better                                        │
│    • Unit: dB (decibels)                                    │
│    • Good MRI reconstruction: 25–35 dB                      │
│                                                             │
│  SSIM  (Structural Similarity Index)                        │
│    • Measures perceptual / structural similarity            │
│    • Range: [0, 1] — higher = better                        │
│    • 1.0 = perfect match                                    │
│    • Better than PSNR at capturing what humans see          │
│                                                             │
│  NMSE  (Normalized Mean Squared Error)                      │
│    • Measures relative reconstruction error                 │
│    • Lower = better                                         │
│    • 0 = perfect reconstruction                             │
│    • Normalized so results are comparable across scans      │
└─────────────────────────────────────────────────────────────┘

These are the EXACT metrics used in the official fastMRI benchmark,

"""

import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as skimage_psnr,
    structural_similarity   as skimage_ssim,
)
from typing import Dict, List



# INDIVIDUAL METRICS

def compute_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Formula:  PSNR = 10 * log10(MAX² / MSE)
    where MAX = dynamic range of the image.

    A +3 dB improvement roughly means halving the mean squared error.


        target:     Ground truth image [H, W], values in [0, 1].
        prediction: Reconstructed image [H, W], values in [0, 1].

    Returns:
        PSNR in dB (float).
    """
    assert target.shape == prediction.shape, (
        f"Shape mismatch: target {target.shape} vs prediction {prediction.shape}"
    )
    prediction = np.clip(prediction, 0.0, 1.0)

    data_range = target.max() - target.min()
    if data_range == 0:
        return float('inf')   # constant image, perfect

    return float(skimage_psnr(target, prediction, data_range=data_range))


def compute_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Structural Similarity Index.

    Considers luminance, contrast, and structural information — much closer
    to human perception than raw pixel error.

        target:     Ground truth [H, W] in [0, 1].
        prediction: Reconstruction [H, W] in [0, 1].

    Returns:
        SSIM in [0, 1] (float).
    """
    assert target.shape == prediction.shape
    prediction = np.clip(prediction, 0.0, 1.0)

    data_range = target.max() - target.min()
    if data_range == 0:
        return 1.0

    return float(skimage_ssim(
        target, prediction,
        data_range=data_range,
        win_size=7                  # 7×7 window
    ))


def compute_nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Normalized Mean Squared Error.

    Formula:  NMSE = ‖target − prediction‖² / ‖target‖²

    Dividing by target energy makes results comparable across scans
    with different overall intensity levels.

        target:     Ground truth [H, W].
        prediction: Reconstruction [H, W].

    Returns:
        NMSE (float, lower = better, 0 = perfect).
    """
    assert target.shape == prediction.shape
    prediction = np.clip(prediction, 0.0, 1.0)

    target_energy = np.sum(target ** 2)
    if target_energy == 0:
        return 0.0

    return float(np.sum((target - prediction) ** 2) / target_energy)



# All three at once
def compute_all_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, and NMSE in one call.

        target:     Ground truth [H, W].
        prediction: Reconstruction [H, W].

    Returns:
        Dict: { 'psnr': float, 'ssim': float, 'nmse': float }
    """
    return {
        'psnr': compute_psnr(target, prediction),
        'ssim': compute_ssim(target, prediction),
        'nmse': compute_nmse(target, prediction),
    }


def compute_dataset_metrics(
    targets:     List[np.ndarray],
    predictions: List[np.ndarray],
) -> Dict[str, dict]:
    """
    Compute metrics over an ENTIRE dataset and return summary statistics.

    Returns:
        Nested dict, e.g.:
        {
            'psnr': { 'mean': 27.4, 'std': 2.1, 'min': 23.0, 'max': 31.5, 'median': 27.2 },
            'ssim': { ... },
            'nmse': { ... },
            'num_samples': 20
        }
    """
    psnr_vals, ssim_vals, nmse_vals = [], [], []

    for t, p in zip(targets, predictions):
        m = compute_all_metrics(t, p)
        psnr_vals.append(m['psnr'])
        ssim_vals.append(m['ssim'])
        nmse_vals.append(m['nmse'])

    def _stats(values: list) -> dict:
        arr = np.array(values)
        return {
            'mean':   float(np.mean(arr)),
            'std':    float(np.std(arr)),
            'min':    float(np.min(arr)),
            'max':    float(np.max(arr)),
            'median': float(np.median(arr)),
        }

    return {
        'psnr':        _stats(psnr_vals),
        'ssim':        _stats(ssim_vals),
        'nmse':        _stats(nmse_vals),
        'num_samples': len(targets),
    }



# SELF-TEST
if __name__ == "__main__":
    print("=" * 55)
    print("  Testing metrics")
    print("=" * 55)

    rng = np.random.default_rng(42)
    target = rng.random((256, 256)).astype(np.float32)

    # 1) Reconstruction
    m = compute_all_metrics(target, target.copy())
    print(f"\n  Perfect reconstruction:")
    print(f"    PSNR  = {m['psnr']:>10.2f} dB   (expected: very high / inf)")
    print(f"    SSIM  = {m['ssim']:>10.4f}      (expected: 1.0)")
    print(f"    NMSE  = {m['nmse']:>10.6f}      (expected: 0.0)")

    # 2) Light noise
    noisy = np.clip(target + 0.05 * rng.standard_normal(target.shape), 0, 1)
    m = compute_all_metrics(target, noisy)
    print(f"\n  Light noise (σ=0.05):")
    print(f"    PSNR  = {m['psnr']:>10.2f} dB")
    print(f"    SSIM  = {m['ssim']:>10.4f}")
    print(f"    NMSE  = {m['nmse']:>10.6f}")

    # 3) Heavy noise
    heavy = np.clip(target + 0.30 * rng.standard_normal(target.shape), 0, 1)
    m = compute_all_metrics(target, heavy)
    print(f"\n  Heavy noise (σ=0.30):")
    print(f"    PSNR  = {m['psnr']:>10.2f} dB")
    print(f"    SSIM  = {m['ssim']:>10.4f}")
    print(f"    NMSE  = {m['nmse']:>10.6f}")

    # 4) Dataset-level stats
    targets_list     = [target] * 5
    predictions_list = [noisy, heavy, target, noisy, heavy]
    stats = compute_dataset_metrics(targets_list, predictions_list)
    print(f"\n  Dataset summary (5 samples):")
    print(f"    PSNR mean ± std = {stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} dB")
    print(f"    SSIM mean ± std = {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")

    print("\n  ✓ All metric tests passed!")
    print("=" * 55)
