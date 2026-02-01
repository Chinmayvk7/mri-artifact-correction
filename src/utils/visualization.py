"""
Visualization Utilities for MRI Reconstruction

FILE: src/utils/visualization.py

    fig01_data_overview.png          — grid of raw MRI slices
    fig02_artifact_examples.png      — what each artifact type looks like
    fig03_training_curves.png        — loss / PSNR / SSIM vs epoch
    fig04_comparison_grid.png        — THE main figure (corrupted → predicted → GT → error)
    fig05_kspace_analysis.png        — k-space domain before/after
    fig06_metrics_distribution.png   — box-plots of metrics vs baseline
    fig07_failure_cases.png          — honest analysis of worst results


"""

import numpy as np
import matplotlib
matplotlib.use('Agg')                        
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os



# Makes every figure look consistent & professional

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'figure.dpi':      150,
    'savefig.dpi':     150,
    'savefig.bbox':    'tight',
    'figure.facecolor':'white',
    'axes.facecolor':  'white',
    'axes.grid':       True,
    'grid.alpha':      0.25,
})

# Determine output directory
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))        # utils/
_SRC_DIR    = os.path.dirname(_THIS_DIR)                        # src/
_PROJECT_DIR= os.path.dirname(_SRC_DIR)                         # project root
OUTPUT_DIR  = os.path.join(_PROJECT_DIR, 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    """Save figure to outputs/figures/ and close it to free memory."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      ✓ saved  {path}")
    return path



# FIG 01 — Data Overview
def plot_data_overview(
    images: List[np.ndarray],
    title:    str = "FastMRI Knee Dataset — Sample Slices",
    filename: str = "fig01_data_overview.png",
):
    """
    A neat grid of raw MRI slices showing dataset diversity.


        images:   List of 2D numpy arrays (magnitude images, any range).
        title:    Figure title.
        filename: Output filename.
    """
    n    = min(len(images), 12)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows),
                             constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    axes_flat = axes.flatten() if rows > 1 else axes.reshape(1, -1).flatten()

    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n:
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'Slice {i+1}', fontsize=10)
            ax.axis('off')
        else:
            ax.set_visible(False)

    _save(fig, filename)



# FIG 02 — Artifact Examples
def plot_artifact_examples(
    clean:        np.ndarray,
    undersampled: np.ndarray,
    spike_noise:  np.ndarray,
    combined:     np.ndarray,
    filename:     str = "fig02_artifact_examples.png",
):
    """
    Side-by-side: Clean | Undersampled | Spike Noise | Combined.

        clean:        Clean ground-truth image [H, W].
        undersampled: Image with undersampling only [H, W].
        spike_noise:  Image with spike noise only [H, W].
        combined:     Image with both artifacts [H, W].
        filename:     Output filename.
    """
    images = [clean, undersampled, spike_noise, combined]
    titles = [
        'Clean\n(Ground Truth)',
        'Undersampled\n(4× acceleration)',
        'Spike Noise\nOnly',
        'Combined\n(Both Artifacts)',
    ]

    # shared intensity range for comparison
    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    fig.suptitle('MRI Artifact Types', fontsize=16, fontweight='bold')

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    fig.colorbar(im, ax=axes, orientation='vertical',
                 fraction=0.018, pad=0.04, label='Signal Intensity')

    _save(fig, filename)



# FIG 03 — Training Curves

def plot_training_curves(
    history:  Dict[str, list],
    filename: str = "fig03_training_curves.png",
):
    """
    Loss, PSNR, and SSIM over training epochs.

    Verifies:
      • Loss is decreasing  →  model is learning
      • Train–val gap is small  →  not overfitting
      • PSNR / SSIM are rising  →  reconstruction quality improves

        history: Dict with keys:
                   'train_loss', 'val_loss'    
                   'val_psnr', 'val_ssim'        

    """
    has_psnr = 'val_psnr' in history and len(history.get('val_psnr', [])) > 0
    has_ssim = 'val_ssim' in history and len(history.get('val_ssim', [])) > 0
    n_plots  = 1 + int(has_psnr) + int(has_ssim)

    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.5),
                             constrained_layout=True)
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-',  linewidth=2, label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'],  'r-',  linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend(frameon=True, loc='best')
    ax.set_ylim(bottom=0)

    idx = 1

    # PSNR
    if has_psnr:
        ax = axes[idx]; idx += 1
        ep_p = range(1, len(history['val_psnr']) + 1)
        ax.plot(ep_p, history['val_psnr'], 'g-o', linewidth=2, markersize=3, color='#27AE60')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Validation PSNR')
        # highlight best
        best_idx = int(np.argmax(history['val_psnr']))
        ax.axhline(history['val_psnr'][best_idx], color='gray', ls='--', alpha=0.5)
        ax.annotate(f"best: {history['val_psnr'][best_idx]:.2f} dB",
                    xy=(best_idx + 1, history['val_psnr'][best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='#27AE60', fontweight='bold')

    # SSIM
    if has_ssim:
        ax = axes[idx]
        ep_s = range(1, len(history['val_ssim']) + 1)
        ax.plot(ep_s, history['val_ssim'], 'o-', linewidth=2, markersize=3, color='#8E44AD')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('Validation SSIM')
        ax.set_ylim(0, 1.05)
        best_idx = int(np.argmax(history['val_ssim']))
        ax.axhline(history['val_ssim'][best_idx], color='gray', ls='--', alpha=0.5)
        ax.annotate(f"best: {history['val_ssim'][best_idx]:.3f}",
                    xy=(best_idx + 1, history['val_ssim'][best_idx]),
                    xytext=(10, -18), textcoords='offset points',
                    fontsize=9, color='#8E44AD', fontweight='bold')

    _save(fig, filename)



# FIG 04 — Main Comparison Grid
def plot_comparison_grid(
    corrupted_images:  List[np.ndarray],
    predicted_images:  List[np.ndarray],
    target_images:     List[np.ndarray],
    metrics_list:      List[Dict[str, float]],
    num_examples:      int = 5,
    filename:          str = "fig04_comparison_grid.png",
):
    """
    Each row = one test example.  Four columns:
        Corrupted Input  |  Model Output  |  Ground Truth  |  Error Map

    The error map (column 4) uses a 'hot' colormap so bright spots = big errors.

        corrupted_images:  List of [H, W] corrupted inputs.
        predicted_images:  List of [H, W] model outputs.
        target_images:     List of [H, W] ground truths.
        metrics_list:      List of dicts with 'psnr' and 'ssim' per example.
        num_examples:      Number of rows to show.
    """
    n = min(num_examples, len(corrupted_images))

    fig, axes = plt.subplots(n, 4, figsize=(17, 4.2 * n))
    fig.suptitle('MRI Artifact Correction — Results',
                 fontsize=18, fontweight='bold', y=1.02)

    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Corrupted Input', 'Model Output', 'Ground Truth', 'Error Map']

    for i in range(n):
        corr = corrupted_images[i]
        pred = predicted_images[i]
        tgt  = target_images[i]
        err  = np.abs(pred - tgt)

        # shared range for columns 0-2
        vmin = min(corr.min(), pred.min(), tgt.min())
        vmax = max(corr.max(), pred.max(), tgt.max())

        axes[i, 0].imshow(corr, cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 1].imshow(pred, cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(tgt,  cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 3].imshow(err,  cmap='hot',  vmin=0)

        for ax in axes[i]:
            ax.axis('off')

        # column headers (top row only)
        if i == 0:
            for ax, title in zip(axes[i], col_titles):
                ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

        # row label with metrics
        if i < len(metrics_list):
            m = metrics_list[i]
            axes[i, 0].set_ylabel(
                f"PSNR {m['psnr']:.1f} dB\nSSIM {m['ssim']:.3f}",
                fontsize=9, rotation=0, labelpad=95, va='center',
                color='#2C3E50', fontweight='bold'
            )

    plt.tight_layout()
    _save(fig, filename)



# FIG 05 — K-Space Analysis
def plot_kspace_analysis(
    clean_kspace:        np.ndarray,
    corrupted_kspace:    np.ndarray,
    clean_image:         np.ndarray,
    corrupted_image:     np.ndarray,
    reconstructed_image: np.ndarray,
    filename:            str = "fig05_kspace_analysis.png",
):
    """
    Two-row figure showing the problem in BOTH domains.

    Row 1 (K-Space):  Clean  |  Corrupted  |  Difference
    Row 2 (Image):    Clean  |  Corrupted  |  Model Output

    NOTE: K-space is shown on a LOG scale because its dynamic range is huge
    Without log, we would only see the bright center dot.


        clean_kspace:        Complex [H, W] — clean k-space.
        corrupted_kspace:    Complex [H, W] — corrupted k-space.
        clean_image:         Real [H, W] — clean magnitude image.
        corrupted_image:     Real [H, W] — corrupted magnitude image.
        reconstructed_image: Real [H, W] — model output.

    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle('K-Space and Image Domain Analysis',
                 fontsize=16, fontweight='bold')

    # log-magnitude of k-space
    ks_clean = np.log1p(np.abs(clean_kspace))
    ks_corr  = np.log1p(np.abs(corrupted_kspace))
    ks_diff  = np.log1p(np.abs(clean_kspace - corrupted_kspace))

    ks_vmax = max(ks_clean.max(), ks_corr.max())

    # Row 0: K-space
    axes[0, 0].imshow(ks_clean, cmap='magma', vmin=0, vmax=ks_vmax)
    axes[0, 0].set_title('Clean K-Space', fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ks_corr, cmap='magma', vmin=0, vmax=ks_vmax)
    axes[0, 1].set_title('Corrupted K-Space', fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(ks_diff, cmap='hot')
    axes[0, 2].set_title('K-Space Difference\n(artifact locations)', fontweight='bold')
    axes[0, 2].axis('off')

    # Row 1: Images
    img_vmax = max(clean_image.max(), corrupted_image.max(), reconstructed_image.max())

    axes[1, 0].imshow(clean_image,         cmap='gray', vmin=0, vmax=img_vmax)
    axes[1, 0].set_title('Clean Image',    fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(corrupted_image,     cmap='gray', vmin=0, vmax=img_vmax)
    axes[1, 1].set_title('Corrupted Image', fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(reconstructed_image, cmap='gray', vmin=0, vmax=img_vmax)
    axes[1, 2].set_title('Model Output',   fontweight='bold')
    axes[1, 2].axis('off')

    # row labels
    axes[0, 0].set_ylabel('K-Space Domain', fontsize=13, rotation=90,
                           labelpad=12, va='center', fontweight='bold')
    axes[1, 0].set_ylabel('Image Domain',  fontsize=13, rotation=90,
                           labelpad=12, va='center', fontweight='bold')

    _save(fig, filename)



# FIG 06 — Metrics Distribution (Box Plots)
def plot_metrics_distribution(
    psnr_values: List[float],
    ssim_values: List[float],
    nmse_values: List[float],
    baseline_psnr:  Optional[float] = None,
    baseline_ssim:  Optional[float] = None,
    filename: str = "fig06_metrics_distribution.png",
):
    """
    Box plots of each metric across the entire test set.

    Shows not just the mean but the SPREAD — tells us how consistent is the model
    A tight box = consistent performance across different slices.
    The dashed red line = baseline (zero-filled), so that we can see the gap and compare.

    
        psnr_values:   List of per-slice PSNR values.
        ssim_values:   List of per-slice SSIM values.
        nmse_values:   List of per-slice NMSE values.
        baseline_psnr: Mean PSNR of zero-filled baseline
        baseline_ssim: Mean SSIM of zero-filled baseline.
  
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle('Metric Distributions — Test Set',
                 fontsize=16, fontweight='bold')

    colors   = ['#4A90D9', '#2ECC71', '#9B59B6']
    labels   = ['PSNR (dB)', 'SSIM', 'NMSE']
    datasets = [psnr_values, ssim_values, nmse_values]
    baselines= [baseline_psnr, baseline_ssim, None]

    for ax, vals, color, ylabel, baseline in zip(axes, datasets, colors, labels, baselines):
        bp = ax.boxplot(
            vals,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6, color=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            medianprops=dict(color='black', linewidth=2.5),
            flierprops=dict(marker='o', markerfacecolor=color, markersize=5, alpha=0.5),
        )
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xticklabels(['U-Net'])

        if baseline is not None:
            ax.axhline(baseline, color='#E74C3C', ls='--', linewidth=2,
                       label=f'Baseline: {baseline:.2f}')
            ax.legend(frameon=True, fontsize=9)


        mean_val = np.mean(vals)
        ax.axhline(mean_val, color=color, ls='-', linewidth=1, alpha=0.4)
        ax.annotate(f'mean={mean_val:.2f}', xy=(1.28, mean_val),
                    fontsize=8.5, color=color, va='center')

        if ylabel == 'SSIM':
            ax.set_ylim(0, 1.08)

    _save(fig, filename)



# FIG 07 — Failure Cases
def plot_failure_cases(
    corrupted_images:  List[np.ndarray],
    predicted_images:  List[np.ndarray],
    target_images:     List[np.ndarray],
    metrics_list:      List[Dict[str, float]],
    num_cases:         int = 3,
    filename:          str = "fig07_failure_cases.png",
):
    """
    Show the WORST results (lowest PSNR).

    
        corrupted_images:  All corrupted images.
        predicted_images:  All model outputs.
        target_images:     All ground truths.
        metrics_list:      Per-example metrics dicts.
        num_cases:         How many failure cases to display.
    """
    # sort ascending by PSNR → worst first
    order = sorted(range(len(metrics_list)),
                   key=lambda i: metrics_list[i]['psnr'])
    worst = order[:min(num_cases, len(order))]

    n = len(worst)
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.2 * n))
    fig.suptitle('Failure Cases — Lowest PSNR',
                 fontsize=16, fontweight='bold', color='#E74C3C', y=1.02)

    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Corrupted', 'Model Output', 'Ground Truth', 'Error Map']

    for row, idx in enumerate(worst):
        corr = corrupted_images[idx]
        pred = predicted_images[idx]
        tgt  = target_images[idx]
        err  = np.abs(pred - tgt)

        vmin = min(corr.min(), tgt.min())
        vmax = max(corr.max(), tgt.max())

        axes[row, 0].imshow(corr, cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(pred, cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 2].imshow(tgt,  cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 3].imshow(err,  cmap='hot',  vmin=0)

        for ax in axes[row]:
            ax.axis('off')

        if row == 0:
            for ax, title in zip(axes[row], col_titles):
                ax.set_title(title, fontsize=12, fontweight='bold')

        m = metrics_list[idx]
        axes[row, 0].set_ylabel(
            f"PSNR {m['psnr']:.1f} dB\nSSIM {m['ssim']:.3f}",
            fontsize=9, rotation=0, labelpad=95, va='center',
            color='#E74C3C', fontweight='bold'
        )

    plt.tight_layout()
    _save(fig, filename)
