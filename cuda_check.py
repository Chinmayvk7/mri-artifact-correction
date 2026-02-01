"""
CUDA & System Verification

It checks that your GPU, PyTorch, and all dependencies are ready.

USAGE:
    cd mri-artifact-correction
    python cuda_check.py
"""

import sys
import importlib

print("=" * 65)
print("  SYSTEM VERIFICATION")
print("=" * 65)

# Python
print(f"\n✓ Python: {sys.version.split()[0]}")

# PyTorch & CUDA
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"✓ CUDA: Available")
        print(f"✓ GPU:  {gpu_name}")
        print(f"✓ VRAM: {gpu_mem_gb:.1f} GB")

        # Quick compute test
        x = torch.randn(2000, 2000, device='cuda')
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        del x, y
        torch.cuda.empty_cache()
        print("✓ GPU compute test: PASSED")
    else:
        print("✗ CUDA: NOT available — training will run on CPU (much slower)")
        print("  → Make sure NVIDIA drivers are installed")
        print("  → Make sure PyTorch was installed with CUDA support:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

except ImportError:
    print("✗ PyTorch: NOT installed")
    print("  → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

# Required Packages
print("\n--- Required Packages ---")

required = {
    'numpy': 'numpy',
    'h5py': 'h5py',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'skimage': 'scikit-image',  
    'tqdm': 'tqdm',
    'PIL': 'Pillow',
}

missing = []
for import_name, pip_name in required.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'installed')
        print(f"  ✓ {pip_name:<20} {version}")
    except ImportError:
        print(f"  ✗ {pip_name:<20} NOT INSTALLED")
        missing.append(pip_name)

# Recommended Packages
print("\n--- Optional Packages ---")
optional = {'tensorboard': 'tensorboard', 'yaml': 'pyyaml'}
for import_name, pip_name in optional.items():
    try:
        importlib.import_module(import_name)
        print(f"  ✓ {pip_name}")
    except ImportError:
        print(f"  ○ {pip_name} (not installed — optional)")

# Install missing packages
if missing:
    print("\n" + "=" * 65)
    print("  MISSING PACKAGES — run this command:")
    print("=" * 65)
    print(f"  pip install {' '.join(missing)}")
    print("=" * 65)
else:
    print("\n" + "=" * 65)
    print("  ✓ ALL CHECKS PASSED — you're ready to go!")
    print("=" * 65)
