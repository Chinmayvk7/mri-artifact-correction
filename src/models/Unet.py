"""
U-Net Model for MRI Reconstruction

implementation of U-Net architecture for medical image artifact correction.

Architecture Overview:
---------------------
U-Net is an encoder-decoder network with skip connections, originally designed
for biomedical image segmentation.

We use U-Net for MRI:
- Preserves fine details via skip connections.
- Works well with limited training data.
- Maintains spatial resolution.

"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    """
    Basic convolutional building block: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class UpBlock(nn.Module):
    """
    Upsampling block: TransposeConv → Concatenate → ConvBlock
    
        -We have added padding/cropping to handle size mismatches
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        
        # Transpose convolution for upsampling
        self.upconv = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2
        )
        
        # Conv block processes concatenated features
        self.conv_block = ConvBlock(out_channels * 2, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with size matching for skip connections.
        """
        # Upsample
        x = self.upconv(x)
        
        # Handle size mismatch (if any) by center cropping the skip connection
        if x.shape != skip.shape:
            # Crop skip to match x
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            
            # Center crop
            skip = skip[:, :, 
                       diff_h // 2 : skip.size(2) - (diff_h - diff_h // 2),
                       diff_w // 2 : skip.size(3) - (diff_w - diff_w // 2)]
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process concatenated features
        x = self.conv_block(x)
        
        return x


class UNet(nn.Module):
    """
    Complete U-Net architecture for MRI reconstruction.
    
    FIXED Architecture:
    ------------------
    Input (1 channel)
        ↓
    Encoder Path:
        enc1: ConvBlock [1→32]     at H×W        (skip1)
        pool1: MaxPool             at H/2×W/2
        enc2: ConvBlock [32→64]    at H/2×W/2    (skip2)
        pool2: MaxPool             at H/4×W/4
        enc3: ConvBlock [64→128]   at H/4×W/4    (skip3)
        pool3: MaxPool             at H/8×W/8
        enc4: ConvBlock [128→256]  at H/8×W/8    (skip4)
        pool4: MaxPool             at H/16×W/16
        
    Bottleneck:
        ConvBlock [256→512]        at H/16×W/16
        
    Decoder Path:
        up4 + skip4 → [512→256]    at H/8×W/8
        up3 + skip3 → [256→128]    at H/4×W/4
        up2 + skip2 → [128→64]     at H/2×W/2
        up1 + skip1 → [64→32]      at H×W
        
    Output:
        1×1 Conv [32→1]            at H×W
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 1, 
                 init_features: int = 32):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Input channels (1 for magnitude MRI images)
            out_channels: Output channels (1 for magnitude images)
            init_features: Number of features in first layer
        """
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        f = init_features  # shorthand
        
        # ENCODER
        # Each encoder block: ConvBlock (no downsampling in conv)
        # Downsampling done separately with MaxPool
        
        self.enc1 = ConvBlock(in_channels, f)       # 1 → 32
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ConvBlock(f, f * 2)             # 32 → 64
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ConvBlock(f * 2, f * 4)         # 64 → 128
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ConvBlock(f * 4, f * 8)         # 128 → 256
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # BOTTLENECK
        self.bottleneck = ConvBlock(f * 8, f * 16)  # 256 → 512
        
        # DECODER
        self.up4 = UpBlock(f * 16, f * 8)           # 512 → 256
        self.up3 = UpBlock(f * 8, f * 4)            # 256 → 128
        self.up2 = UpBlock(f * 4, f * 2)            # 128 → 64
        self.up1 = UpBlock(f * 2, f)                # 64 → 32
        
        # OUTPUT
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Where
            x: Input tensor [batch, 1, height, width]
        
        Returns:
            Output tensor [batch, 1, height, width]
        """
        # ENCODER PATH
        # Save outputs BEFORE pooling for skip connections
        
        enc1 = self.enc1(x)           # [B, 32, H, W]
        p1 = self.pool1(enc1)         # [B, 32, H/2, W/2]
        
        enc2 = self.enc2(p1)          # [B, 64, H/2, W/2]
        p2 = self.pool2(enc2)         # [B, 64, H/4, W/4]
        
        enc3 = self.enc3(p2)          # [B, 128, H/4, W/4]
        p3 = self.pool3(enc3)         # [B, 128, H/8, W/8]
        
        enc4 = self.enc4(p3)          # [B, 256, H/8, W/8]
        p4 = self.pool4(enc4)         # [B, 256, H/16, W/16]
        
        # BOTTLENECK
        bottleneck = self.bottleneck(p4)  # [B, 512, H/16, W/16]
        
        # DECODER PATH
        # Each up block: upsample → concat with skip → conv
        
        dec4 = self.up4(bottleneck, enc4)  # [B, 256, H/8, W/8]
        dec3 = self.up3(dec4, enc3)        # [B, 128, H/4, W/4]
        dec2 = self.up2(dec3, enc2)        # [B, 64, H/2, W/2]
        dec1 = self.up1(dec2, enc1)        # [B, 32, H, W]
        
        # OUTPUT
        output = self.output_conv(dec1)    # [B, 1, H, W]
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def print_model_summary(model: UNet, input_size: tuple = (1, 1, 320, 320)):
    """Print a summary of the model architecture."""
    print("=" * 70)
    print("U-Net Model Summary")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Input channels: {model.in_channels}")
    print(f"  Output channels: {model.out_channels}")
    
    total_params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Total trainable: {total_params:,} ({total_params/1e6:.2f}M)")
    
    param_memory = total_params * 4 / (1024**2)
    print(f"  Memory (params): ~{param_memory:.1f} MB")
    
    print(f"\nTesting forward pass with input size {input_size}...")
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✓ Success!")
        print(f"  Input shape:  {tuple(dummy_input.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    """Test U-Net model."""
    print("\n" + "=" * 70)
    print("U-Net Model Test")
    print("=" * 70)
    
    # Create model
    print("\nCreating U-Net...")
    model = UNet(in_channels=1, out_channels=1, init_features=32)
    
    # Print summary
    print_model_summary(model)
    
    # Test on different image sizes
    print("\n" + "-" * 70)
    print("Testing on various input sizes...")
    print("-" * 70)
    
    test_sizes = [
        (1, 1, 256, 256),
        (4, 1, 320, 320),
        (2, 1, 640, 368),
    ]
    
    for size in test_sizes:
        try:
            dummy_input = torch.randn(size)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"  ✓ {size} → {tuple(output.shape)}")
        except Exception as e:
            print(f"  ✗ {size} failed: {e}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\n" + "-" * 70)
        print("Testing on GPU...")
        print("-" * 70)
        
        model_cuda = model.cuda()
        dummy_input = torch.randn(2, 1, 320, 320).cuda()
        
        try:
            with torch.no_grad():
                output = model_cuda(dummy_input)
            print(f"  ✓ GPU forward pass successful")
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        except Exception as e:
            print(f"  ✗ GPU test failed: {e}")
    else:
        print("\n  ℹ CUDA not available, skipping GPU test")
    
    print("\n" + "=" * 70)
    print("All tests completed! ✓")
    print("=" * 70 + "\n")