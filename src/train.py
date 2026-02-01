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
import torch.nn as nn
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

#path setup so imports work from project root
_SCRIPT = os.path.dirname(os.path.abspath(__file__))          # src/
_ROOT   = os.path.dirname(_SCRIPT)                            # project root
os.chdir(_ROOT)                                               # all relative paths from project root

sys.path.insert(0, os.path.join(_SCRIPT, 'data'))
sys.path.insert(0, os.path.join(_SCRIPT, 'models'))
sys.path.insert(0, os.path.join(_SCRIPT, 'utils'))

from dataset       import FastMRIArtifactDataset, create_train_val_datasets
from Unet          import UNet
from losses        import CombinedL1SSIMLoss
from metrics       import compute_all_metrics
from visualization import plot_training_curves



# DEFAULT CONFIGURATION

DEFAULT_CONFIG = dict(
    # Data
    data_path            = 'data/raw/knee_singlecoil_val',  
    total_slices         = 120,      # train + val combined
    val_fraction         = 0.2,      # 20 % validation
    image_size           = 256,      # square crop side length

    # Artifacts
    acceleration_factor  = 4,        # 4× undersampling
    num_spikes           = 10,       # spike noise count

    # Model
    in_channels          = 1,        # grayscale
    out_channels         = 1,
    init_features        = 32,      

    # Training
    batch_size           = 4,        # reduce to 2
    num_epochs           = 50,
    learning_rate        = 1e-3,     # Adam starting LR
    weight_decay         = 1e-5,     # L2 regularisation
    loss_alpha           = 0.5,      # L1 weight in combined loss

    # LR Schedule 
    scheduler_patience   = 8,        # epochs without improvement before reducing LR
    scheduler_factor     = 0.5,      # LR multiplier when reducing

    # Checkpointing
    save_every           = 10,       # save numbered checkpoint every N epochs
    output_dir           = 'outputs',

    # DataLoader 
    num_workers          = 2,   
)


def parse_args():
    p = argparse.ArgumentParser(description='Train MRI Artifact Correction U-Net')
    p.add_argument('--epochs',      type=int,   help='Override num_epochs')
    p.add_argument('--batch_size',  type=int,   help='Override batch_size')
    p.add_argument('--lr',          type=float, help='Override learning_rate')
    p.add_argument('--image_size',  type=int,   help='Override image_size')
    p.add_argument('--data_path',   type=str,   help='Override data_path')
    p.add_argument('--total_slices',type=int,   help='Override total_slices')
    p.add_argument('--resume',      action='store_true', help='Resume from latest checkpoint')
    return p.parse_args()


def apply_cli_overrides(config: dict, args) -> dict:
    """Merge command-line flags into config dict."""
    mapping = {
        'epochs':       'num_epochs',
        'batch_size':   'batch_size',
        'lr':           'learning_rate',
        'image_size':   'image_size',
        'data_path':    'data_path',
        'total_slices': 'total_slices',
    }
    for cli_key, cfg_key in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            config[cfg_key] = val
    return config



# AUTO-DETECT DATA DIRECTORY
def find_data_path(hint: str) -> str:
    """
    Try several common locations for fastMRI .h5 files.
    Returns the first one that exists, or raises an error.
    """
    candidates = [
        hint,
        'data/raw/knee_singlecoil_val',
        'data/raw/knee_singlecoil_train',
        'data/raw',
    ]
    for c in candidates:
        if os.path.isdir(c) and any(f.endswith('.h5') for f in os.listdir(c)):
            return c
    # Worst case: data/raw looking for .h5
    for root, dirs, files in os.walk('data'):
        if any(f.endswith('.h5') for f in files):
            return root
    raise FileNotFoundError(
        "Could not find fastMRI .h5 files.\n"
        "Please pass --data_path <directory> or put .h5 files in data/raw/"
    )



# TRAIN / VALIDATE FUNCTIONS

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """
    One full pass over the training set.

    Returns average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc='  Train', leave=False, ncols=90):
        corrupted = batch['corrupted'].to(device)
        clean     = batch['clean'].to(device)

        predicted = model(corrupted)
        loss      = criterion(predicted, clean)

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    One full pass over the validation set.

    Returns (avg_loss, avg_psnr, avg_ssim).
    We compute PSNR/SSIM here so we can plot them over training.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    psnr_list  = []
    ssim_list  = []

    for batch in tqdm(loader, desc='  Val  ', leave=False, ncols=90):
        corrupted = batch['corrupted'].to(device)
        clean     = batch['clean'].to(device)

        predicted = model(corrupted)
        loss      = criterion(predicted, clean)

        total_loss += loss.item()
        n_batches  += 1

        # metrics (move to CPU, convert to numpy)
        pred_np  = predicted.cpu().numpy()
        clean_np = clean.cpu().numpy()

        for i in range(pred_np.shape[0]):
            m = compute_all_metrics(clean_np[i, 0], pred_np[i, 0])
            psnr_list.append(m['psnr'])
            ssim_list.append(m['ssim'])

    return (
        total_loss / n_batches,
        float(np.mean(psnr_list)) if psnr_list else 0.0,
        float(np.mean(ssim_list)) if ssim_list else 0.0,
    )



# MAIN
def main():
    print("\n" + "═" * 65)
    print("  MRI ARTIFACT CORRECTION — TRAINING")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("═" * 65)

    # config
    args   = parse_args()
    config = apply_cli_overrides(dict(DEFAULT_CONFIG), args)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")
    if device.type == 'cuda':
        print(f"     GPU    : {torch.cuda.get_device_name(0)}")
        print(f"     VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # find data
    config['data_path'] = find_data_path(config['data_path'])
    print(f"\n  Data   : {config['data_path']}")

    # print full config
    print("\n  Config:")
    for k, v in config.items():
        print(f"     {k:<25} {v}")

    # output dirs
    out      = Path(config['output_dir'])
    models   = out / 'models'
    figures  = out / 'figures'
    for d in (models, figures, out / 'results', out / 'logs'):
        d.mkdir(parents=True, exist_ok=True)

    # save config snapshot
    with open(out / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # datasets
    print("\n  Loading datasets …")
    img_size = (config['image_size'], config['image_size'])

    train_ds, val_ds = create_train_val_datasets(
        data_path            = config['data_path'],
        total_slices         = config['total_slices'],
        val_fraction         = config['val_fraction'],
        image_size           = img_size,
        acceleration_factor  = config['acceleration_factor'],
        num_spikes           = config['num_spikes'],
    )

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,   batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )

    # model
    print("\n  Building model …")
    model = UNet(
        in_channels  = config['in_channels'],
        out_channels = config['out_channels'],
        init_features= config['init_features'],
    ).to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     Parameters : {total_p:,}  ({trainable_p:,} trainable)")

    # loss + optimiser
    print("\n   Optimiser setup …")
    criterion  = CombinedL1SSIMLoss(alpha=config['loss_alpha'])
    optimizer  = optim.Adam(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=config['scheduler_patience'],
        factor=config['scheduler_factor'],
        verbose=True,
    )
    print(f"     Adam lr={config['learning_rate']}  |  "
          f"weight_decay={config['weight_decay']}")


    start_epoch     = 0
    best_val_loss   = float('inf')

    if args.resume:
        ckpt_path = models / 'latest_checkpoint.pth'
        if ckpt_path.exists():
            print(f"\n  Resuming from {ckpt_path} …")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch   = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f"     Resuming from epoch {start_epoch}")
        else:
            print("     No checkpoint found — starting fresh.")

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': []}


    # TRAINING LOOP

    print("\n" + "═" * 65)
    print("  TRAINING STARTED")
    print("═" * 65)

    for epoch in range(start_epoch, config['num_epochs']):
        t0   = datetime.now()
        lr   = optimizer.param_groups[0]['lr']
        print(f"\n  ── Epoch {epoch+1:>3}/{config['num_epochs']}   lr={lr:.2e} ──")

        # train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # validate
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)

        # lr schedule
        scheduler.step(val_loss)

        # log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['lr'].append(lr)

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"     train_loss={train_loss:.4f}  |  "
              f"val_loss={val_loss:.4f}  |  "
              f"PSNR={val_psnr:.2f} dB  |  "
              f"SSIM={val_ssim:.4f}  |  "
              f"{elapsed:.0f}s")

        # checkpoints
        torch.save({
            'epoch':             epoch,
            'model_state_dict':  model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':        train_loss,
            'val_loss':          val_loss,
            'best_val_loss':     best_val_loss,
        }, models / 'latest_checkpoint.pth')


        if (epoch + 1) % config['save_every'] == 0:
            torch.save(model.state_dict(), models / f'checkpoint_epoch{epoch+1}.pth')
            print(f"      checkpoint_epoch{epoch+1}.pth")

        # best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), models / 'best_model.pth')
            print(f"      RECENT →  best_model.pth  (val_loss={val_loss:.4f})")

        # plot curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_curves(history)




    with open(out / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # final curves plot
    plot_training_curves(history)

    print("\n" + "═" * 65)
    print("  TRAINING COMPLETE")
    print("═" * 65)
    print(f"     Best val loss  : {best_val_loss:.4f}")
    print(f"     Best val PSNR  : {max(history['val_psnr']):.2f} dB")
    print(f"     Best val SSIM  : {max(history['val_ssim']):.4f}")
    print(f"     Models dir     : {models}")
    print(f"     History        : {out / 'training_history.json'}")
    print(f"\n  → Next step:  python src/evaluate.py")
    print("═" * 65)


if __name__ == "__main__":
    main()
