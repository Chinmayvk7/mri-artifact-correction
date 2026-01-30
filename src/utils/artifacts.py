'''
MRI Artifact Simulator

    We will be simulating MRI artifacts for training deep learning models.
    VIA :  
            1. Undersampling (accelerated acquisition ) 
            2. Spike noise ( hardware glitches )

     MultiArtifactSimulator class combines multiple artifact types for realistic corruption

'''

import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



import numpy as np


# NAIVE DFT (FALLBACK FOR ODD SIZES)

def dft_naive(x: np.ndarray) -> np.ndarray:
    """
    Naive Discrete Fourier Transform (O(N^2)).

    Used only when input length is odd and cannot be split evenly
    by the Cooley–Tukey FFT algorithm.

    Formula:
        X[k] = sum(x[n] * exp(-2πi * k * n / N)) for n = 0 to N-1
    """
    N = len(x)

    # Create index vectors
    n = np.arange(N)
    k = n.reshape((N, 1))

    # Construct DFT matrix: W[k,n] = exp(-2πi * k * n / N)
    W = np.exp(-2j * np.pi * k * n / N)

    # Matrix-vector multiplication gives the DFT
    return W @ x



# FORWARD FFT (COOLEY–TUKEY)

def fft_1d_scratch(x: np.ndarray) -> np.ndarray:
    """
    Forward 1D Fast Fourier Transform implemented from scratch
    using the recursive Cooley–Tukey algorithm.

    This is the FORWARD FFT required for inverse FFT construction.

    Algorithm:
    - Divide input into even and odd indexed samples
    - Recursively compute FFTs
    - Combine using twiddle factors

    Time complexity:
        O(N log N)
    """
    N = len(x)

    # Base case: FFT of length 1 is the signal itself
    if N <= 1:
        return x

    # If length is odd, fall back to naive DFT
    if N % 2 != 0:
        return dft_naive(x)

    # Divide: split signal into even and odd indexed elements
    even = fft_1d_scratch(x[0::2])  # x[0], x[2], x[4], ...
    odd  = fft_1d_scratch(x[1::2])  # x[1], x[3], x[5], ...

  
    # W_N^k = exp(-2πi * k / N)
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)

    # Combine:
    # X[k]       = even[k] + W[k] * odd[k]
    # X[k+N/2]   = even[k] - W[k] * odd[k]
    return np.concatenate([
        even + W * odd,
        even - W * odd
    ])



# INVERSE FFT (IMPLEMENTATION FROM SCRATCH) 
# We won't be using  # IFFT(X) = conj(FFT(conj(X))) / N and will be implementing IFFT from scratch.

def ifft_1d_scratch(X: np.ndarray) -> np.ndarray:
    """
    Explicit recursive 1D Inverse Fast Fourier Transform.

    Key differences from FFT:
    - Uses POSITIVE exponent: exp(+2πi * k / N)
    - Normalizes by dividing by N (done gradually as /2 per stage)

    Mathematical definition:
        x[n] = (1/N) * sum(X[k] * exp(2πi * k * n / N))
    """
    N = len(X)

    # Base case
    if N <= 1:
        return X

    # Handle odd-sized inputs using naive DFT identity
    if N % 2 != 0:
        return np.conj(dft_naive(np.conj(X))) / N

    # Divide: split frequency components
    even = ifft_1d_scratch(X[0::2])
    odd  = ifft_1d_scratch(X[1::2])

  
    W = np.exp(+2j * np.pi * np.arange(N // 2) / N)

    # Combine inverse FFT results
    result = np.concatenate([
        even + W * odd,
        even - W * odd
    ])

    # Normalize (each recursion level contributes a factor of 1/2)
    return result / 2



# 2D INVERSE FFT 

def ifft_2d_scratch(kspace: np.ndarray) -> np.ndarray:
    """
    2D Inverse FFT implemented using separability and again we didn't use the conjugate mtd.

    Property:
        2D IFFT = IFFT_columns( IFFT_rows(kspace) )

    Steps:
    1. Apply 1D IFFT along rows
    2. Apply 1D IFFT along columns
    """
    H, W = kspace.shape

    # Step 1: Apply IFFT to each row
    temp = np.zeros_like(kspace, dtype=complex)
    for i in range(H):
        temp[i, :] = ifft_1d_scratch(kspace[i, :])

    # Step 2: Apply IFFT to each column
    image = np.zeros_like(kspace, dtype=complex)
    for j in range(W):
        image[:, j] = ifft_1d_scratch(temp[:, j])

    return image



# MRI-CORRECT CENTERED IFFT

def ifft2c_scratch(kspace: np.ndarray) -> np.ndarray:
    """
    Centered 2D Inverse FFT used in MRI reconstruction.

    MRI convention:
    - k-space is stored with zero-frequency at the center

    Steps:
    1. ifftshift → move DC component to corner
    2. Apply 2D inverse FFT
    3. fftshift → re-center the reconstructed image
    """
    # Undo k-space centering
    kspace_unshifted = np.fft.ifftshift(kspace)

    # Apply custom 2D IFFT
    image_unshifted = ifft_2d_scratch(kspace_unshifted)

    # Center the reconstructed image
    image = np.fft.fftshift(image_unshifted)

    return image



class UndersamplingSimulator:          
    
    def __init__(self, acceleration_factor: int = 4, center_fraction: float = 0.08):   
        self.acceleration_factor = acceleration_factor       # How much to accelerate (R).
        self.center_fraction = center_fraction               # Fraction of center k-space to fully sample
        logger.info(f"Undersampling simulator initialized: "
                   f"R={acceleration_factor}, center={center_fraction*100}%")

    
    def create_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.float32)       # create a empty k-space mask.
        mask[::self.acceleration_factor, :] = 1.0                # Each selected row is fully sampled across width in multiples of 4 (4 times faster scan).
        num_center_lines = int(height * self.center_fraction)    # Compute how many center k-space lines to always keep.
        center_start = height // 2 - num_center_lines // 2       # Find where the center region starts.
        center_end = center_start + num_center_lines             # Find where the center region ends.
        mask[center_start:center_end, :] = 1.0
        sampling_percentage = mask.mean() * 100         
        
        logger.debug(f"Created mask: {height}×{width}, "
                    f"{sampling_percentage:.1f}% sampled")

        return mask


    def apply(self, kspace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:        # Function to apply undersampling
        
        height, width = kspace.shape 
        mask = self.create_mask(height, width)
        undersampled_kspace = kspace * mask            # element wise multiplication ( 0s and 1s ) 
        return undersampled_kspace, mask


    
class SpikeNoiseSimulator:

    def __init__(self, num_spikes: int = 10, amplitude_range: Tuple[float, float] = (5, 20)):
        
        self.num_spikes = num_spikes
        self.amplitude_range = amplitude_range

        logger.info(f"Spike noise simulator initialized: "
                   f"{num_spikes} spikes, amplitude {amplitude_range[0]}-{amplitude_range[1]}×")

        
    def apply(self, kspace, seed=None):
        
        if seed is not None:
            np.random.seed(seed)              # reproduciblity

        height, width = kspace.shape
        corrupted_kspace = kspace.copy()              # Create a copy and not destroy the original data
        median_magnitude = np.median(np.abs(kspace))  # Gets magnitude of complex numbers and find a typical k-space value

        spike_locations = []

        for spike_idx in range(self.num_spikes):

            # Random location in k-space
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            
            # Random spike amplitude (relative to median)
            amplitude_factor = np.random.uniform(*self.amplitude_range)
            amplitude = median_magnitude * amplitude_factor              # spike magnitude scales with scan intensity

            phase = np.random.uniform(0, 2 * np.pi)                      # Creating complex phase.
            
            spike_value = amplitude * np.exp(1j * phase)                 # Creating a complex spike => z = Ae^iϕ
            corrupted_kspace[y, x] += spike_value                        # Add spike

            spike_locations.append({
                'position': (y, x),
                'amplitude': amplitude,
                'phase': phase,
                'relative_amplitude': amplitude_factor
            })
 
        logger.debug(f"Added {self.num_spikes} spikes to k-space")
        return corrupted_kspace, spike_locations



class MultiArtifactSimulator:
            

    def __init__(self, 
                 acceleration_factor: int = 4,
                 num_spikes: int = 10,
                 apply_undersampling: bool = True,
                 apply_spikes: bool = True):
        
            
        self.apply_undersampling = apply_undersampling
        self.apply_spikes = apply_spikes

        # Initialize individual simulators

        if apply_undersampling:
            self.undersampling_sim = UndersamplingSimulator(acceleration_factor)

        if apply_spikes:
            self.spikes_sim = SpikeNoiseSimulator(num_spikes)

        logger.info(f"Multi-artifact simulator initialized: "
                   f"undersampling={apply_undersampling}, spikes={apply_spikes}")

        
        
    def apply(self, kspace: np.ndarray, seed: Optional[int] = None) -> dict:

        corrupted = kspace.copy()
        metadata = {
            'undersampling_applied': self.apply_undersampling,
            'spikes_applied': self.apply_spikes
        }

         # We will first apply undersampling first
        mask = None
        if self.apply_undersampling:
            corrupted, mask = self.undersampling_sim.apply(corrupted)
            metadata['acceleration_factor'] = self.undersampling_sim.acceleration_factor
            metadata['sampling_percentage'] = mask.mean() * 100

        
        # We will apply spike noise
        spike_locations = None
        if self.apply_spikes:            
            corrupted, spike_locations = self.spikes_sim.apply(corrupted, seed=seed)
            metadata['num_spikes'] = len(spike_locations)      # Record how many spikes were performed.

        return {
            'corrupted_kspace': corrupted,
            'clean_kspace': kspace,
            'mask': mask,
            'spike_locations': spike_locations,
            'metadata': metadata
        }


# Testing and we demonstrate one example here
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, 'src/data')
    from fastmri_loader import FastMRILoader, normalize_image
    
    print("="*60)
    print("Artifact Simulator Test")
    print("="*60)
    
    # Load sample data
    try:
      
        loader = FastMRILoader("data/singlecoil_val")
        kspace_clean, image_clean = loader.load_slice(0, 15)
        
        print(f"\n✓ Loaded test slice")
        print(f"  K-space shape: {kspace_clean.shape}")
        
        # Test undersampling
        print("\n" + "-"*60)
        print("Testing Undersampling...")
        print("-"*60)
        
        undersample_sim = UndersamplingSimulator(acceleration_factor=4)
        k_under, mask = undersample_sim.apply(kspace_clean)
        
        # Reconstruct image (inverse FFT)
        image_under = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_under))))
        
        print(f"✓ Undersampling applied")
        print(f"  Sampling: {mask.mean()*100:.1f}%")
        print(f"  Expected: ~{100/undersample_sim.acceleration_factor + 8:.1f}%")
        
        # Test spike noise
        print("\n" + "-"*60)
        print("Testing Spike Noise...")
        print("-"*60)
        
        spike_sim = SpikeNoiseSimulator(num_spikes=15)
        k_spikes, locations = spike_sim.apply(kspace_clean, seed=42)
        
        # Reconstruct
        image_spikes = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_spikes))))
        
        print(f"✓ Spike noise applied")
        print(f"  Number of spikes: {len(locations)}")
        print(f"  Spike locations: {[loc['position'] for loc in locations[:3]]}...")
        
        # Test combined
        print("\n" + "-"*60)
        print("Testing Combined Artifacts...")
        print("-"*60)
        
        multi_sim = MultiArtifactSimulator(acceleration_factor=4, num_spikes=10)
        result = multi_sim.apply(kspace_clean, seed=42)
        
        k_combined = result['corrupted_kspace']
        image_combined = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_combined))))
        
        print(f"✓ Combined artifacts applied")
        print(f"  Metadata: {result['metadata']}")
        
        # Visualize all
        print("\n" + "-"*60)
        print("Creating Visualizations...")
        print("-"*60)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Row 1: Images
        axes[0, 0].imshow(normalize_image(image_clean), cmap='gray')
        axes[0, 0].set_title('Clean Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(normalize_image(image_under), cmap='gray')
        axes[0, 1].set_title(f'Undersampled (R={undersample_sim.acceleration_factor})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(normalize_image(image_spikes), cmap='gray')
        axes[0, 2].set_title(f'Spike Noise ({spike_sim.num_spikes} spikes)')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(normalize_image(image_combined), cmap='gray')
        axes[0, 3].set_title('Combined Artifacts')
        axes[0, 3].axis('off')
        
        # Row 2: K-space (log magnitude)
        axes[1, 0].imshow(np.log(np.abs(kspace_clean) + 1), cmap='gray')
        axes[1, 0].set_title('Clean K-Space')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.log(np.abs(k_under) + 1), cmap='gray')
        axes[1, 1].set_title('Undersampled K-Space')
        axes[1, 1].axis('off')
        
        # Mark spike locations in k-space
        k_spikes_vis = np.log(np.abs(k_spikes) + 1)
        axes[1, 2].imshow(k_spikes_vis, cmap='gray')
        for loc in locations[:10]:  # Show first 10
            y, x = loc['position']
            axes[1, 2].plot(x, y, 'r*', markersize=8)
        axes[1, 2].set_title('K-Space with Spikes (marked)')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(np.log(np.abs(k_combined) + 1), cmap='gray')
        axes[1, 3].set_title('Combined K-Space')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/artifact_simulation_test.png', dpi=150, bbox_inches='tight')
        
        print(f"✓ Visualization saved: outputs/artifact_simulation_test.png")
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nMake sure fastMRI data is loaded first!")
        import traceback
        traceback.print_exc()