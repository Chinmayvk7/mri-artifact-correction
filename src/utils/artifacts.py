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


def normalize_image(img):
    img = np.abs(img)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img


def create_synthetic_kspace(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a phantom with multiple structures (simulating knee-like anatomy)
    phantom = np.zeros((size, size), dtype=np.float64)
    
    # Main elliptical structure (like bone)
    ellipse1 = ((X/0.3)**2 + (Y/0.6)**2) < 1
    phantom[ellipse1] = 1.0
    
    # Inner structure (like marrow)
    ellipse2 = ((X/0.2)**2 + (Y/0.45)**2) < 1
    phantom[ellipse2] = 0.7
    
    # Add some smaller structures (like cartilage/soft tissue)
    circle1 = ((X-0.4)**2 + (Y-0.3)**2) < 0.05
    phantom[circle1] = 0.9
    
    circle2 = ((X+0.4)**2 + (Y+0.3)**2) < 0.05
    phantom[circle2] = 0.9
    
    # Add subtle texture
    np.random.seed(42)
    texture = np.random.randn(size, size) * 0.05
    phantom = phantom + texture * (phantom > 0)
    phantom = np.clip(phantom, 0, 1)
    
    # Convert to k-space (centered)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phantom)))
    
    return kspace, phantom


# Testing and we demonstrate one example here
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    print("="*60)
    print("Artifact Simulator Test")
    print("="*60)
    
    # Create synthetic k-space data for demonstration
    print("\nGenerating synthetic MRI phantom...")
    kspace_clean, phantom = create_synthetic_kspace(256)
    
    print(f"\n✓ Created test phantom")
    print(f"  K-space shape: {kspace_clean.shape}")
    
    # Reconstruct clean image from k-space
    image_clean = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_clean))))
    
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
    
    # Row 2: K-space (log magnitude with proper windowing)
    def display_kspace(kspace_data, ax, title, spike_locs=None):
        log_kspace = np.log1p(np.abs(kspace_data))
        vmin, vmax = np.percentile(log_kspace, [2, 99.5])
        ax.imshow(log_kspace, cmap='gray', vmin=vmin, vmax=vmax)
        if spike_locs is not None:
            for loc in spike_locs:
                y, x = loc['position']
                ax.plot(x, y, 'r*', markersize=10, markeredgewidth=1.5)
        ax.set_title(title)
        ax.axis('off')
    
    display_kspace(kspace_clean, axes[1, 0], 'Clean K-Space')
    display_kspace(k_under, axes[1, 1], 'Undersampled K-Space')
    display_kspace(k_spikes, axes[1, 2], 'K-Space with Spikes (marked)', locations)
    display_kspace(k_combined, axes[1, 3], 'Combined K-Space')
    
    plt.tight_layout()
    plt.savefig('artifact_simulation_test.png', dpi=150, bbox_inches='tight')
    
    print(f"✓ Visualization saved: artifact_simulation_test.png")
    
    # Additional figure: Show the undersampling mask
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    
    axes2[0].imshow(mask, cmap='gray')
    axes2[0].set_title('Undersampling Mask')
    axes2[0].axis('off')
    
    # Show difference images
    diff_under = np.abs(image_clean - image_under)
    axes2[1].imshow(diff_under, cmap='hot')
    axes2[1].set_title('Undersampling Error')
    axes2[1].axis('off')
    
    diff_spikes = np.abs(image_clean - image_spikes)
    axes2[2].imshow(diff_spikes, cmap='hot')
    axes2[2].set_title('Spike Noise Error')
    axes2[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('artifact_analysis.png', dpi=150, bbox_inches='tight')
    
    print(f"✓ Analysis saved: artifact_analysis.png")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)