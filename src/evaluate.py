"""
Evaluation Script — MRI Artifact Correction
=============================================
FILE: src/evaluate.py

FIXED VERSION: Uses independent normalization for each image.
"""

import torch
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT)
os.chdir(_ROOT)

sys.path.insert(0, os.path.join(_SCRIPT, 'data'))
sys.path.insert(0, os.path.join(_SCRIPT, 'models'))
sys.path.insert(0, os.path.join(_SCRIPT, 'utils'))

from fastmri_loader import FastMRILoader
from artifacts import MultiArtifactSimulator
from Unet import UNet
from metrics import compute_all_metrics
from visualization import (
    plot_data_overview,
    plot_artifact_examples,
    plot_training_curves,
    plot_comparison_grid,
    plot_kspace_analysis,
    plot_metrics_distribution,
    plot_failure_cases,
)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = image.min(), image.max()
    if hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def center_crop(image: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = image.shape
    if h < th or w < tw:
        ph, pw = max(0, th - h), max(0, tw - w)
        image = np.pad(image, ((ph//2, ph - ph//2), (pw//2, pw - pw//2)), mode='constant')
        h, w = image.shape
    sh, sw = (h - th) // 2, (w - tw) // 2
    return image[sh:sh+th, sw:sw+tw].copy()


def find_data_path(hint: str) -> str:
    for c in [hint, 'data/singlecoil_val', 'data/singlecoil_train',
              'data/raw/knee_singlecoil_val', 'data/raw']:
        if os.path.isdir(c) and any(f.endswith('.h5') for f in os.listdir(c)):
            return c
    for root, _, files in os.walk('data'):
        if any(f.endswith('.h5') for f in files):
            return root
    raise FileNotFoundError("Cannot find fastMRI .h5 files.")


def main():
    print("\n" + "=" * 65)
    print("  MRI ARTIFACT CORRECTION - EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 65)

    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='outputs/models/best_model.pth')
    p.add_argument('--num_samples', type=int, default=20)
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--data_path', type=str, default='data/singlecoil_val')
    p.add_argument('--no_residual', action='store_true')
    args = p.parse_args()

    use_residual = True
    config_path = Path('outputs/training_config.json')
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
            use_residual = train_config.get('use_residual', True)

    if args.no_residual:
        use_residual = False

    print(f"\n  RESIDUAL LEARNING: {'ENABLED' if use_residual else 'DISABLED'}")

    img_size = (args.image_size, args.image_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device : {device}")

    print(f"\n  Loading model ...  {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"  ERROR: File not found: {ckpt_path}")
        return

    model = UNet(in_channels=1, out_channels=1, init_features=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("     Model loaded")

    data_path = find_data_path(args.data_path)
    print(f"\n  Data   : {data_path}")

    accel = 4
    spikes = 5
    if config_path.exists():
        accel = train_config.get('acceleration_factor', 4)
        spikes = train_config.get('num_spikes', 5)

    loader = FastMRILoader(data_path)
    artifact_sim = MultiArtifactSimulator(acceleration_factor=accel, num_spikes=spikes)
    print(f"  Artifacts: {accel}x acceleration, {spikes} spikes")

    eval_slices = []
    for fi in range(len(loader.file_list)):
        info = loader.get_file_info(fi)
        ns = info['num_slices']
        eval_slices.append((fi, ns // 2))
        if len(eval_slices) >= args.num_samples:
            break

    print(f"\n  Evaluating {len(eval_slices)} slices ...")

    all_corrupted = []
    all_predicted = []
    all_clean = []
    all_metrics = []
    all_baseline = []
    raw_images = []
    kspace_example = None

    for fi, si in tqdm(eval_slices, desc='  Eval', ncols=70):
        kspace_clean, image_clean = loader.load_slice(fi, si)
        raw_images.append(image_clean.copy())

        result = artifact_sim.apply(kspace_clean)
        kspace_corrupted = result['corrupted_kspace']

        image_corrupted = np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_corrupted)))
        )

        # FIXED: Independent normalization for each image
        clean_norm = normalize_image(image_clean)
        corrupted_norm = normalize_image(image_corrupted)

        clean_crop = center_crop(clean_norm, *img_size)
        corrupted_crop = center_crop(corrupted_norm, *img_size)

        with torch.no_grad():
            inp = torch.from_numpy(corrupted_crop)[None, None].to(device)

            if use_residual:
                correction = model(inp)
                out = inp + correction
                out = torch.clamp(out, 0.0, 1.0)
            else:
                out = model(inp)

            predicted = out[0, 0].cpu().numpy()

        predicted = np.clip(predicted, 0.0, 1.0)

        model_m = compute_all_metrics(clean_crop, predicted)
        baseline_m = compute_all_metrics(clean_crop, corrupted_crop)

        all_corrupted.append(corrupted_crop)
        all_predicted.append(predicted)
        all_clean.append(clean_crop)
        all_metrics.append(model_m)
        all_baseline.append(baseline_m)

        if kspace_example is None:
            kspace_example = dict(
                clean_kspace=kspace_clean,
                corrupted_kspace=kspace_corrupted,
                clean_image=clean_crop,
                corrupted_image=corrupted_crop,
                predicted_image=predicted,
            )

    m_psnr = [m['psnr'] for m in all_metrics]
    m_ssim = [m['ssim'] for m in all_metrics]
    m_nmse = [m['nmse'] for m in all_metrics]

    b_psnr = [m['psnr'] for m in all_baseline]
    b_ssim = [m['ssim'] for m in all_baseline]
    b_nmse = [m['nmse'] for m in all_baseline]

    psnr_improvement = np.mean(m_psnr) - np.mean(b_psnr)
    ssim_improvement = np.mean(m_ssim) - np.mean(b_ssim)

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  {'Metric':<8}  {'Baseline':<26}  {'U-Net':<26}  {'Improvement'}")
    print("-" * 72)
    print(f"  {'PSNR':<8}  {np.mean(b_psnr):.2f} +/- {np.std(b_psnr):.2f} dB              "
          f"{np.mean(m_psnr):.2f} +/- {np.std(m_psnr):.2f} dB              "
          f"{psnr_improvement:+.2f} dB")
    print(f"  {'SSIM':<8}  {np.mean(b_ssim):.4f} +/- {np.std(b_ssim):.4f}                "
          f"{np.mean(m_ssim):.4f} +/- {np.std(m_ssim):.4f}                "
          f"{ssim_improvement:+.4f}")
    print(f"  {'NMSE':<8}  {np.mean(b_nmse):.6f}                          "
          f"{np.mean(m_nmse):.6f}                          "
          f"{np.mean(m_nmse)-np.mean(b_nmse):+.6f}")
    print("=" * 72)

    if psnr_improvement > 0:
        print(f"\n  SUCCESS: Model improves PSNR by {psnr_improvement:.2f} dB!")
    else:
        print(f"\n  WARNING: Model performs worse than baseline.")

    results_dir = Path('outputs/results')
    results_dir.mkdir(parents=True, exist_ok=True)

    def _stats(vals):
        a = np.array(vals)
        return {k: float(v) for k, v in zip(
            ['mean', 'std', 'min', 'max', 'median'],
            [a.mean(), a.std(), a.min(), a.max(), np.median(a)]
        )}

    results = {
        'model': {'psnr': _stats(m_psnr), 'ssim': _stats(m_ssim), 'nmse': _stats(m_nmse)},
        'baseline': {'psnr': _stats(b_psnr), 'ssim': _stats(b_ssim), 'nmse': _stats(b_nmse)},
        'improvement': {
            'psnr_db': float(psnr_improvement),
            'ssim': float(ssim_improvement),
            'nmse': float(np.mean(m_nmse) - np.mean(b_nmse)),
        },
        'num_samples': len(eval_slices),
        'use_residual': use_residual,
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON -> {results_dir / 'evaluation_results.json'}")

    print("\n  Generating figures ...")
    figures_dir = Path('outputs/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    # fig01
    print("     fig01 ...", end=' ')
    try:
        overview_imgs = [normalize_image(img) for img in raw_images[:12]]
        plot_data_overview(overview_imgs, save_path=figures_dir / 'fig01_data_overview.png')
    except Exception as e:
        print(f"error: {e}")

    # fig02
    print("     fig02 ...", end=' ')
    try:
        fi0, si0 = eval_slices[0]
        ks_clean0, img_clean0 = loader.load_slice(fi0, si0)

        us_sim = MultiArtifactSimulator(acceleration_factor=accel, num_spikes=0)
        us_res = us_sim.apply(ks_clean0)
        img_us = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(us_res['corrupted_kspace']))))

        sp_sim = MultiArtifactSimulator(acceleration_factor=1, num_spikes=spikes)
        sp_res = sp_sim.apply(ks_clean0)
        img_sp = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sp_res['corrupted_kspace']))))

        plot_artifact_examples(
            clean=center_crop(normalize_image(img_clean0), *img_size),
            undersampled=center_crop(normalize_image(img_us), *img_size),
            spike_noise=center_crop(normalize_image(img_sp), *img_size),
            combined=all_corrupted[0],
            save_path=figures_dir / 'fig02_artifact_examples.png'
        )
    except Exception as e:
        print(f"error: {e}")

    # fig03
    print("     fig03 ...", end=' ')
    history_path = Path('outputs/training_history.json')
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(history, save_path=figures_dir / 'fig03_training_curves.png')
        except Exception as e:
            print(f"error: {e}")
    else:
        print("skipped")

    # fig04
    print("     fig04 ...", end=' ')
    try:
        best_idx = sorted(range(len(all_metrics)),
                          key=lambda i: all_metrics[i]['psnr'], reverse=True)[:5]
        plot_comparison_grid(
            corrupted_images=[all_corrupted[i] for i in best_idx],
            predicted_images=[all_predicted[i] for i in best_idx],
            target_images=[all_clean[i] for i in best_idx],
            metrics_list=[all_metrics[i] for i in best_idx],
            num_examples=5,
            save_path=figures_dir / 'fig04_comparison_grid.png'
        )
    except Exception as e:
        print(f"error: {e}")

    # fig05
    print("     fig05 ...", end=' ')
    try:
        if kspace_example:
            plot_kspace_analysis(
                clean_kspace=kspace_example['clean_kspace'],
                corrupted_kspace=kspace_example['corrupted_kspace'],
                clean_image=kspace_example['clean_image'],
                corrupted_image=kspace_example['corrupted_image'],
                reconstructed_image=kspace_example['predicted_image'],
                save_path=figures_dir / 'fig05_kspace_analysis.png'
            )
    except Exception as e:
        print(f"error: {e}")

    # fig06
    print("     fig06 ...", end=' ')
    try:
        plot_metrics_distribution(
            psnr_values=m_psnr,
            ssim_values=m_ssim,
            nmse_values=m_nmse,
            baseline_psnr=float(np.mean(b_psnr)),
            baseline_ssim=float(np.mean(b_ssim)),
            save_path=figures_dir / 'fig06_metrics_distribution.png'
        )
    except Exception as e:
        print(f"error: {e}")

    # fig07
    print("     fig07 ...", end=' ')
    try:
        plot_failure_cases(
            corrupted_images=all_corrupted,
            predicted_images=all_predicted,
            target_images=all_clean,
            metrics_list=all_metrics,
            num_cases=3,
            save_path=figures_dir / 'fig07_failure_cases.png'
        )
    except Exception as e:
        print(f"error: {e}")

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65)
    print(f"     Results  -> outputs/results/evaluation_results.json")
    print(f"     Figures  -> outputs/figures/")
    print(f"\n     Key result: {psnr_improvement:+.2f} dB PSNR improvement")
    print("=" * 65)


if __name__ == "__main__":
    main()