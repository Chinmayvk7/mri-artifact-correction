"""
Training Script — MRI Artifact Correction with U-Net

FILE: src/train.py

USAGE  (run from project root):
    python src/train.py                          # defaults
    python src/train.py --epochs 100             # more epochs
    python src/train.py --batch_size 2           # if GPU OOM
    python src/train.py --image_size 320         # larger images
    python src/train.py --resume                 # resume from last checkpoint
    python src/train.py --data_path path/to/data # override data location

WHAT HAPPENS:
    1.  Verify GPU is available
    2.  Auto-detect fastMRI data directory
    3.  Create train (80%) and val (20%) datasets
    4.  Build U-Net model  (7.8M params with init_features=32)
    5.  Train with Combined L1+SSIM loss + Adam + LR scheduling
    6.  Every epoch: log train loss, val loss, val PSNR, val SSIM
    7.  Save best_model.pth whenever val loss improves
    8.  Save periodic checkpoints for safety
    9.  Write training_history.json  (used by evaluate.py)
   10.  Plot training curves every 5 epochs
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_SCRIPT)
os.chdir(_ROOT)

sys.path.insert(0, os.path.join(_SCRIPT, 'data'))
sys.path.insert(0, os.path.join(_SCRIPT, 'models'))
sys.path.insert(0, os.path.join(_SCRIPT, 'utils'))

from dataset import FastMRIArtifactDataset, create_train_val_datasets
from unet import UNet
from losses import CombinedL1SSIMLoss
from metrics import compute_all_metrics
from visualization import plot_training_curves


DEFAULT_CONFIG = dict(
    data_path            = 'data/singlecoil_val',
    total_slices         = 120,
    val_fraction         = 0.2,
    image_size           = 256,

    acceleration_factor  = 4,
    num_spikes           = 10,

    in_channels          = 1,
    out_channels         = 1,
    init_features        = 32,

    batch_size           = 4,
    num_epochs           = 50,
    learning_rate        = 1e-3,
    weight_decay         = 1e-5,
    loss_alpha           = 0.5,

    scheduler_patience   = 8,
    scheduler_factor     = 0.5,

    save_every           = 10,
    output_dir           = 'outputs',

    num_workers          = 2,
)


def parse_args():
    p = argparse.ArgumentParser(description='Train MRI Artifact Correction U-Net')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--image_size', type=int)
    p.add_argument('--data_path', type=str)
    p.add_argument('--total_slices', type=int)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def apply_cli_overrides(config: dict, args) -> dict:
    mapping = {
        'epochs': 'num_epochs',
        'batch_size': 'batch_size',
        'lr': 'learning_rate',
        'image_size': 'image_size',
        'data_path': 'data_path',
        'total_slices': 'total_slices',
    }
    for cli_key, cfg_key in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            config[cfg_key] = val
    return config


def find_data_path(hint: str) -> str:
    candidates = [
        hint,
        'data/singlecoil_val',
        'data/singlecoil_train',
        'data',
    ]
    for c in candidates:
        if os.path.isdir(c) and any(f.endswith('.h5') for f in os.listdir(c)):
            return c
    for root, _, files in os.walk('data'):
        if any(f.endswith('.h5') for f in files):
            return root
    raise FileNotFoundError("Could not find fastMRI .h5 files.")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc='  Train', leave=False, ncols=90):
        corrupted = batch['corrupted'].to(device)
        clean = batch['clean'].to(device)

        predicted = model(corrupted)
        loss = criterion(predicted, clean)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    psnr_list = []
    ssim_list = []

    for batch in tqdm(loader, desc='  Val', leave=False, ncols=90):
        corrupted = batch['corrupted'].to(device)
        clean = batch['clean'].to(device)

        predicted = model(corrupted)
        loss = criterion(predicted, clean)

        total_loss += loss.item()
        n_batches += 1

        pred_np = predicted.cpu().numpy()
        clean_np = clean.cpu().numpy()

        for i in range(pred_np.shape[0]):
            m = compute_all_metrics(clean_np[i, 0], pred_np[i, 0])
            psnr_list.append(m['psnr'])
            ssim_list.append(m['ssim'])

    return (
        total_loss / n_batches,
        float(np.mean(psnr_list)),
        float(np.mean(ssim_list)),
    )


def main():
    print("\n" + "═" * 65)
    print("  MRI ARTIFACT CORRECTION — TRAINING")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("═" * 65)

    args = parse_args()
    config = apply_cli_overrides(dict(DEFAULT_CONFIG), args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")
    if device.type == 'cuda':
        print(f"     GPU    : {torch.cuda.get_device_name(0)}")
        print(f"     VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config['data_path'] = find_data_path(config['data_path'])
    print(f"\n  Data   : {config['data_path']}")

    out = Path(config['output_dir'])
    models = out / 'models'
    figures = out / 'figures'
    for d in (models, figures, out / 'results', out / 'logs'):
        d.mkdir(parents=True, exist_ok=True)

    img_size = (config['image_size'], config['image_size'])
    train_ds, val_ds = create_train_val_datasets(
        data_path=config['data_path'],
        total_slices=config['total_slices'],
        val_fraction=config['val_fraction'],
        image_size=img_size,
        acceleration_factor=config['acceleration_factor'],
        num_spikes=config['num_spikes'],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )

    model = UNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        init_features=config['init_features'],
    ).to(device)

    criterion = CombinedL1SSIMLoss(alpha=config['loss_alpha'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['scheduler_patience'],
        factor=config['scheduler_factor'],
    )

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': []}

    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}")

    with open(out / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history)


if __name__ == "__main__":
    main()
