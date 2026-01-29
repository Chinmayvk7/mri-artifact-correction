
"""
Fix fastmri_loader.py to handle both real FastMRI data (direct complex64)
and synthetic data (structured array with 'real' and 'imag' fields)
"""

import sys
from pathlib import Path

def fix_loader():
    """Fix the load_slice and load_volume methods"""
    
    # Read the current file
    loader_path = Path('src/data/fastmri_loader.py')
    
    if not loader_path.exists():
        print(f"❌ Error: {loader_path} not found!")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Make sure you're in the project root directory")
        return False
    
    with open(loader_path, 'r') as f:
        content = f.read()
    
    # Backup original
    backup_path = loader_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✓ Created backup: {backup_path}")
    
    # Fix 1: load_slice method
    old_load_slice = """            # Convert to complex numpy array
            # HDF5 stores complex as structured array with 'real' and 'imag' fields
            kspace = kspace_raw['real'] + 1j * kspace_raw['imag']"""
    
    new_load_slice = """            # Convert to complex numpy array
            # Real FastMRI data stores as complex64 directly
            # Synthetic data may use structured array with 'real' and 'imag' fields
            if hasattr(kspace_raw, 'dtype') and kspace_raw.dtype.names:
                # Structured array format (synthetic data)
                kspace = kspace_raw['real'] + 1j * kspace_raw['imag']
            else:
                # Direct complex format (real FastMRI data)
                kspace = np.array(kspace_raw)"""
    
    if old_load_slice in content:
        content = content.replace(old_load_slice, new_load_slice)
        print("✓ Fixed load_slice method")
    else:
        print("⚠ Warning: load_slice pattern not found (might be already fixed)")
    
    # Fix 2: load_volume method  
    old_load_volume = """            kspace_volume = kspace_raw['real'] + 1j * kspace_raw['imag']"""
    
    new_load_volume = """            # Handle both structured array and direct complex formats
            if hasattr(kspace_raw, 'dtype') and kspace_raw.dtype.names:
                kspace_volume = kspace_raw['real'] + 1j * kspace_raw['imag']
            else:
                kspace_volume = np.array(kspace_raw)"""
    
    if old_load_volume in content:
        content = content.replace(old_load_volume, new_load_volume)
        print("✓ Fixed load_volume method")
    else:
        print("⚠ Warning: load_volume pattern not found (might be already fixed)")
    
    # Write the fixed content
    with open(loader_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ SUCCESS! Fixed {loader_path}")
    print(f"   Backup saved to: {backup_path}")
    return True

if __name__ == "__main__":
    print("="*60)
    print("Fixing FastMRI Loader for Real Data Compatibility")
    print("="*60)
    print()
    
    success = fix_loader()
    
    if success:
        print("\n" + "="*60)
        print("Next step: Test the loader with:")
        print("  python -c 'from src.data.fastmri_loader import FastMRILoader; ...")
        print("="*60)
        sys.exit(0)
    else:
        sys.exit(1)
