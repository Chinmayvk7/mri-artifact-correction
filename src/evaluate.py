"""
Evaluation Script — MRI Artifact Correction

FILE: src/evaluate.py

FIXED VERSION: Supports RESIDUAL LEARNING
    output = corrupted + model(corrupted)

USAGE:
    python src/evaluate.py
    python src/evaluate.py --num_samples 30
    python src/evaluate.py --checkpoint outputs/models/checkpoint_epoch50.pth
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

# path setup
_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_SCRIPT)
os.chdir(_ROOT)

sys.path.insert(0, os.path.join(_SCRIPT, 'data'))
sys.path.insert(0, os.path.join(_SCRIPT, 'models'))
sys.path.insert(0, os.path.join(_SCRIPT, 'utils'))

from fastmri_loader import FastMRILoader
from artifacts      import MultiArtifactSimulator
from Unet           import UNet
from metrics        import compute_all_metrics
from visualization  import (
    plot_data_overview,
    plot_artifact_examples,
    plot_training_curves,
    plot_comparison_grid,
    plot_kspace_analysis,
    plot_metrics_distribution,
    plot_failure_cases,
)


# HELPERS
def center_crop(image: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = image.shape
    if h < th or w < tw:
        ph, pw = max(0, th - h), max(0, tw - w)
        image  = np.pad(image,
                        ((ph//2, ph - ph//2), (pw//2, pw - pw//2)),
                        mode='constant')
        h, w   = image.shape
    sh, sw = (h - th) // 2, (w - tw) // 2
    return image[sh:sh+th, sw:sw+tw].copy()


def find_data_path(hint: str) -> str:
    for c in [hint,
              'data/singlecoil_val',
              'data/singlecoil_train',
              'data/raw/knee_singlecoil_val',
              'data/raw/knee_singlecoil_train',
              'data/raw']:
        if os.path.isdir(c) and any(f.endswith('.h5') for f in os.listdir(c)):
            return c
    for root, _, files in os.walk('data'):
        if any(f.endswith('.h5') for f in files):
            return root
    raise FileNotFoundError("Cannot find fastMRI .h5 files. Pass --data_path.")


# MAIN
def main():
    print("\n" + "=" * 65)
    print("  MRI ARTIFACT CORRECTION - EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 65)

    # args 
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   type=str, default='outputs/models/best_model.pth')
    p.add_argument('--num_samples',  type=int, default=20)
    p.add_argument('--image_size',   type=int, default=256)
    p.add_argument('--data_path',    type=str, default='data/singlecoil_val')
    p.add_argument('--no_residual',  action='store_true', help='Disable residual learning')
    args = p.parse_args()

    # Check if residual learning was used in training
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
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device : {device}")

    # load model
    print(f"\n  Loading model  ...  {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"  ERROR: File not found: {ckpt_path}")
        print("    Did training finish?  Check outputs/models/")
        return

    model = UNet(in_channels=1, out_channels=1, init_features=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("     Model loaded successfully")

    # data
    data_path = find_data_path(args.data_path)
    print(f"\n  Data   : {data_path}")

    # Use same artifact settings as training
    accel = 4
    spikes = 5
    if config_path.exists():
        accel = train_config.get('acceleration_factor', 4)
        spikes = train_config.get('num_spikes', 5)
    
    loader       = FastMRILoader(data_path)
    artifact_sim = MultiArtifactSimulator(acceleration_factor=accel, num_spikes=spikes)
    print(f"  Artifacts: {accel}x acceleration, {spikes} spikes")

    # pick evaluation slices
    eval_slices = []
    for fi in range(len(loader.file_list)):
        info = loader.get_file_info(fi)
        ns   = info['num_slices']
        eval_slices.append((fi, ns // 2))
        if len(eval_slices) >= args.num_samples:
            break

    if len(eval_slices) < args.num_samples:
        eval_slices = []
        for fi in range(min(10, len(loader.file_list))):
            info = loader.get_file_info(fi)
            ns   = info['num_slices']
            idxs = np.linspace(int(ns*0.2), int(ns*0.8),
                               max(1, args.num_samples // 5 + 1)).astype(int)
            for si in idxs:
                eval_slices.append((fi, int(si)))
                if len(eval_slices) >= args.num_samples:
                    break
            if len(eval_slices) >= args.num_samples:
                break

    print(f"\n  Evaluating {len(eval_slices)} slices ...")

    # EVALUATE EVERY SLICE
    all_corrupted  = []
    all_predicted  = []
    all_clean      = []
    all_metrics    = []
    all_baseline   = []
    raw_images     = []

    kspace_example = None

    for fi, si in tqdm(eval_slices, desc='  Eval', ncols=70):
        kspace_clean, image_clean = loader.load_slice(fi, si)
        raw_images.append(image_clean.copy())

        result           = artifact_sim.apply(kspace_clean)
        kspace_corrupted = result['corrupted_kspace']

        image_corrupted = np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_corrupted)))
        )

        lo  = image_clean.min()
        hi  = image_clean.max()
        rng = (hi - lo) if (hi - lo) > 0 else 1.0

        clean_norm     = ((image_clean     - lo) / rng).astype(np.float32)
        corrupted_norm = np.clip(
            ((image_corrupted - lo) / rng).astype(np.float32), 0.0, 1.0
        )

        clean_crop     = center_crop(clean_norm,     *img_size)
        corrupted_crop = center_crop(corrupted_norm, *img_size)

        # model inference WITH RESIDUAL LEARNING
        with torch.no_grad():
            inp = torch.from_numpy(corrupted_crop)[None, None].to(device)
            
            if use_residual:
                # RESIDUAL: output = input + model(input)
                correction = model(inp)
                out = inp + correction
                out = torch.clamp(out, 0.0, 1.0)
            else:
                out = model(inp)
            
            predicted = out[0, 0].cpu().numpy()

        predicted = np.clip(predicted, 0.0, 1.0)

        model_m    = compute_all_metrics(clean_crop, predicted)
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

    # SUMMARY TABLE
    m_psnr = [m['psnr'] for m in all_metrics]
    m_ssim = [m['ssim'] for m in all_metrics]
    m_nmse = [m['nmse'] for m in all_metrics]

    b_psnr = [m['psnr'] for m in all_baseline]
    b_ssim = [m['ssim'] for m in all_baseline]
    b_nmse = [m['nmse'] for m in all_baseline]

    psnr_improvement = np.mean(m_psnr) - np.mean(b_psnr)
    ssim_improvement = np.mean(m_ssim) - np.mean(b_ssim)

    # The unusual spacing I have given is for better formatting and presentation.
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  {'Metric':<8}  {'Baseline (zero-filled)':<26}  "
          f"{'U-Net':<26}  {'Improvement'}")
    print("-" * 72)
    print(f"  {'PSNR':<8}  {np.mean(b_psnr):.2f} +/- {np.std(b_psnr):.2f} dB              "
          f"{np.mean(m_psnr):.2f} +/- {np.std(m_psnr):.2f} dB              "
          f"{psnr_improvement:+.2f} dB")
    print(f"  {'SSIM':<8}  {np.mean(b_ssim):.4f} +/- {np.std(b_ssim):.4f}                "
          f"{np.mean(m_ssim):.4f} +/- {np.std(m_ssim):.4f}                "
          f"{ssim_improvement:+.4f}")
    print(f"  {'NMSE':<8}  {np.mean(b_nmse):.6f} +/- {np.std(b_nmse):.6f}          "
          f"{np.mean(m_nmse):.6f} +/- {np.std(m_nmse):.6f}          "
          f"{np.mean(m_nmse)-np.mean(b_nmse):+.6f}")
    print("=" * 72)

    # Check if model is actually better
    if psnr_improvement > 0:
        print(f"\n  SUCCESS: Model improves PSNR by {psnr_improvement:.2f} dB!")
    else:
        print(f"\n  WARNING: Model performs WORSE than baseline by {-psnr_improvement:.2f} dB")
        print("           Consider retraining with different parameters.")

    # SAVE JSON RESULTS
    results_dir = Path('outputs/results')
    results_dir.mkdir(parents=True, exist_ok=True)

    def _stats(vals):
        a = np.array(vals)
        return {k: float(v) for k, v in zip(
            ['mean','std','min','max','median'],
            [a.mean(), a.std(), a.min(), a.max(), np.median(a)]
        )}

    results = {
        'model':       { 'psnr': _stats(m_psnr), 'ssim': _stats(m_ssim), 'nmse': _stats(m_nmse) },
        'baseline':    { 'psnr': _stats(b_psnr), 'ssim': _stats(b_ssim), 'nmse': _stats(b_nmse) },
        'improvement': {
            'psnr_db':  float(psnr_improvement),
            'ssim':     float(ssim_improvement),
            'nmse':     float(np.mean(m_nmse) - np.mean(b_nmse)),
        },
        'num_samples': len(eval_slices),
        'use_residual': use_residual,
        'timestamp':   datetime.now().isoformat(),
    }

    json_path = results_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON  ->  {json_path}")

    # GENERATE ALL 7 FIGURES
    print("\n  Generating figures ...")
    figures_dir = Path('outputs/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    # fig01: data overview
    print("     fig01 ...", end=' ')
    overview_imgs = []
    for img in raw_images[:12]:
        lo, hi = img.min(), img.max()
        overview_imgs.append((img - lo) / (hi - lo) if hi > lo else img * 0)
    try:
        plot_data_overview(overview_imgs, save_path=figures_dir / 'fig01_data_overview.png')
        print("done")
    except Exception as e:
        print(f"error: {e}")

    # fig02: artifact examples
    print("     fig02 ...", end=' ')
    fi0, si0 = eval_slices[0]
    ks_clean0, img_clean0 = loader.load_slice(fi0, si0)
    lo0 = img_clean0.min()
    hi0 = img_clean0.max()
    rng0 = (hi0 - lo0) if (hi0 - lo0) > 0 else 1.0

    try:
        us_sim  = MultiArtifactSimulator(acceleration_factor=accel, num_spikes=0)
        us_res  = us_sim.apply(ks_clean0)
        img_us  = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(us_res['corrupted_kspace']))))
        img_us_norm = np.clip(center_crop(((img_us - lo0)/rng0).astype(np.float32), *img_size), 0, 1)
    except Exception:
        img_us_norm = all_corrupted[0]

    try:
        sp_sim  = MultiArtifactSimulator(acceleration_factor=1, num_spikes=spikes)
        sp_res  = sp_sim.apply(ks_clean0)
        img_sp  = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sp_res['corrupted_kspace']))))
        img_sp_norm = np.clip(center_crop(((img_sp - lo0)/rng0).astype(np.float32), *img_size), 0, 1)
    except Exception:
        img_sp_norm = all_corrupted[0]

    clean_for_fig = center_crop(((img_clean0 - lo0)/rng0).astype(np.float32), *img_size)

    try:
        plot_artifact_examples(
            clean=clean_for_fig,
            undersampled=img_us_norm,
            spike_noise=img_sp_norm,
            combined=all_corrupted[0],
            save_path=figures_dir / 'fig02_artifact_examples.png'
        )
        print("done")
    except Exception as e:
        print(f"error: {e}")

    # fig03: training curves
    print("     fig03 ...", end=' ')
    history_path = Path('outputs/training_history.json')
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        try:
            plot_training_curves(history, save_path=figures_dir / 'fig03_training_curves.png')
            print("done")
        except Exception as e:
            print(f"error: {e}")
    else:
        print("skipped - no training_history.json")

    # fig04: comparison grid (by PSNR)
    print("     fig04 ...", end=' ')
    best_idx = sorted(range(len(all_metrics)),
                      key=lambda i: all_metrics[i]['psnr'], reverse=True)[:5]
    try:
        plot_comparison_grid(
            corrupted_images = [all_corrupted[i] for i in best_idx],
            predicted_images = [all_predicted[i] for i in best_idx],
            target_images    = [all_clean[i]     for i in best_idx],
            metrics_list     = [all_metrics[i]   for i in best_idx],
            num_examples     = 5,
            save_path        = figures_dir / 'fig04_comparison_grid.png'
        )
        print("done")
    except Exception as e:
        print(f"error: {e}")

    # fig05: k-space analysis
    print("     fig05 ...", end=' ')
    if kspace_example:
        try:
            plot_kspace_analysis(
                clean_kspace        = kspace_example['clean_kspace'],
                corrupted_kspace    = kspace_example['corrupted_kspace'],
                clean_image         = kspace_example['clean_image'],
                corrupted_image     = kspace_example['corrupted_image'],
                reconstructed_image = kspace_example['predicted_image'],
                save_path           = figures_dir / 'fig05_kspace_analysis.png'
            )
            print("done")
        except Exception as e:
            print(f"error: {e}")

    # fig06: metrics distribution
    print("     fig06 ...", end=' ')
    try:
        plot_metrics_distribution(
            psnr_values   = m_psnr,
            ssim_values   = m_ssim,
            nmse_values   = m_nmse,
            baseline_psnr = float(np.mean(b_psnr)),
            baseline_ssim = float(np.mean(b_ssim)),
            save_path     = figures_dir / 'fig06_metrics_distribution.png'
        )
        print("done")
    except Exception as e:
        print(f"error: {e}")

    # fig07: failure cases
    print("     fig07 ...", end=' ')
    try:
        plot_failure_cases(
            corrupted_images = all_corrupted,
            predicted_images = all_predicted,
            target_images    = all_clean,
            metrics_list     = all_metrics,
            num_cases        = 3,
            save_path        = figures_dir / 'fig07_failure_cases.png'
        )
        print("done")
    except Exception as e:
        print(f"error: {e}")

    # DONE
    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65)
    print(f"     Results  ->  outputs/results/evaluation_results.json")
    print(f"     Figures  ->  outputs/figures/  (7 figures)")
    print(f"\n     Key result  :  {psnr_improvement:+.2f} dB PSNR improvement")
    if psnr_improvement > 0:
        print(f"\n  -> Next step: Generate more figures and analysis retrain the model with more epochs and slices.")
    else:
        print(f"\n  -> Model needs retraining. Run: python src/train.py")
    print("=" * 65)


if __name__ == "__main__":
    main()