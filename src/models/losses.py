# -*- coding: utf-8 -*-
"""
Loss Functions for MRI Reconstruction

FILE: src/models/losses.py

WHY NOT JUST L1 OR MSE?
    • L1 alone  → good pixel accuracy, but can produce blurry outputs
    • MSE alone → over-penalises large errors, also produces blur
    • SSIM loss → captures structural / perceptual similarity
    • L1 + SSIM → best of both: pixel accuracy AND perceptual quality

    alpha controls the blend:
        loss = alpha × L1  +  (1 − alpha) × (1 − SSIM)
        alpha = 0.5 → equal weight (safe default)

SSIM LOSS IMPLEMENTATION:
    scikit-image's SSIM is NOT differentiable (can't backprop through it).
    So we re-implement SSIM using PyTorch ops — where everything is differentiable,
    and gradients flow back to the network weights during training.

    The math is the same as the original SSIM paper (Wang et al. 2004):
        SSIM(x, y) = (2μx μy + C1)(2σxy + C2)
                     ─────────────────────────────
                     (μx² + μy² + C1)(σx² + σy² + C2)

    We approximate the local means/variances with a Gaussian-weighted
    convolution (window_size = 11, σ = 1.5) — standard practice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss = 1 − SSIM.

    Minimising this loss → maximising structural similarity.


        window_size: Gaussian window size
        channel: Number of image channels (1 for grayscale MRI).
    """

    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel     = channel
        # pre-compute the 2D Gaussian window
        self.register_buffer('window', self._make_window(window_size, channel))

    @staticmethod
    def _make_window(size: int, channel: int) -> torch.Tensor:
        """
        Build a 2D Gaussian kernel shaped [channel, 1, size, size].

        The Gaussian gives more weight to the centre of each local patch,
        which is perceptually more meaningful than uniform weighting.
        """
        # 1-D Gaussian
        x     = torch.arange(size, dtype=torch.float32) - size // 2
        gauss = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
        gauss = gauss / gauss.sum()

        # outer product → 2-D Gaussian
        g2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)          # [size, size]

        # repeat for each channel, shape for grouped conv2d
        window = g2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)  # [C, 1, size, size]
        return window

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute 1 − SSIM (scalar loss).

        Args:
            prediction: [B, C, H, W]
            target:     [B, C, H, W]

        Returns:
            Scalar loss tensor.
        """
        pad = self.window_size // 2
        w   = self.window       

        # local means (Gaussian-weighted)
        mu_x  = F.conv2d(prediction, w, padding=pad, groups=self.channel)
        mu_y  = F.conv2d(target,     w, padding=pad, groups=self.channel)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        # local variances & covariance
        sigma_x2  = F.conv2d(prediction * prediction, w, padding=pad, groups=self.channel) - mu_x2
        sigma_y2  = F.conv2d(target     * target,     w, padding=pad, groups=self.channel) - mu_y2
        sigma_xy  = F.conv2d(prediction * target,     w, padding=pad, groups=self.channel) - mu_xy

        # stability constants  (L = 1 because images are in [0, 1])
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # SSIM map
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

        # loss = 1 - mean SSIM
        return 1.0 - ssim_map.mean()


class CombinedL1SSIMLoss(nn.Module):
    """
    Combined loss:  alpha × L1  +  (1 − alpha) × SSIMLoss

    We will use this for training.


        alpha: Weight for L1 component.  0.5 = equal blend.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha   = alpha
        self.l1      = nn.L1Loss()
        self.ssim    = SSIMLoss()
        print(f"  Loss  -  Combined  [alpha={alpha:.2f} * L1 + {1-alpha:.2f} * SSIM]")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

            prediction: [B, C, H, W]
            target:     [B, C, H, W]
        Returns:
            Scalar loss.
        """
        l1_val   = self.l1(prediction, target)
        ssim_val = self.ssim(prediction, target)
        return self.alpha * l1_val + (1.0 - self.alpha) * ssim_val



# SELF-TEST
if __name__ == "__main__":
    print("=" * 55)
    print("  Testing loss functions")
    print("=" * 55)

    torch.manual_seed(42)
    pred   = torch.rand(2, 1, 256, 256).requires_grad_(True)
    target = torch.rand(2, 1, 256, 256)

    # SSIM loss
    ssim_loss = SSIMLoss()

    loss_random  = ssim_loss(pred, target)
    loss_perfect = ssim_loss(target, target)
    print(f"\n  SSIM Loss (random inputs) : {loss_random.item():.4f}")
    print(f"  SSIM Loss (perfect match) : {loss_perfect.item():.6f}  (should be ≈ 0)")

    # Combined loss 
    print()
    combined = CombinedL1SSIMLoss(alpha=0.5)
    loss = combined(pred, target)
    print(f"  Combined Loss (random)    : {loss.item():.4f}")

    # Backward pass (proves gradients flow)
    loss.backward()
    print("  Backward pass             : PASSED ✓")

    # Verify perfect match gives ~0
    pred2   = target.clone().requires_grad_(True)
    loss2   = combined(pred2, target)
    print(f"  Combined Loss (perfect)   : {loss2.item():.6f}  (should be ≈ 0)")

    print("\n  ✓ All loss tests passed!")
    print("=" * 55)
