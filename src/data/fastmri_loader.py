'''
This below code is meant for data loading interface for the NYU fastMRI dataset, which is a large-scale medical imaging dataset used for MRI reconstruction research. The goal is to make it easy to:

    1) Load MRI scan data from disk.
    2) Access individual 2D slices or entire 3D volumes.
    3) Preprocess images (normalize, crop).
    4) Visualize the data.

- scanners capture data in "k-space" (frequency domain), not as direct images.

    We will be performing the follwing operations:- 
    
    Compressed sensing: Reconstructing images from incomplete k-space data
    Deep learning: Training neural networks to improve reconstruction quality
    Image quality assessment: Comparing different reconstruction methods


'''

import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)          # INFO level: shows important details like file counts, loaded volumes,etc.
logger = logging.getLogger(__name__)    

# fastMRILoader Class

class FastMRILoader:

    def __init__(self, data_path: str):                              # self is FastMRILoader object
        self.data_path = Path(data_path)                             # Converts string to Path object.
        self.file_list = sorted(list(self.data_path.glob('*.h5')))   # fastMRI stores each volume in a separate HDF5 file

        if len(self.file_list) == 0:
            raise ValueError(f"No .h5 files found in {data_path}")

        logger.info(f"Found {len(self.file_list)} MRI volumes in {data_path}")   


    ''' 
    The below load_slice function indexes into list of .h5 files, selects which 2D slice from 3D volume and converts it into
        tuple of 
            First : k - space (for complex values)
            Second : image ( real magnitudes) 
    AND to store amplitude(strength) and phase(timing/angle)  we need to convert the image to complex numbers and inorder to perform fourier transform.
     =>   kspace(x, y) = real(x, y) + i · imag(x, y)
    '''



    def load_slice(self, file_index: int, slice_index: int) -> Tuple[np.ndarray, np.ndarray]: 
        h5_file_path = self.file_list[file_index]

        with h5py.File(h5_file_path, 'r') as h5_file:
            kspace_raw = h5_file['kspace'][slice_index]
        
            if hasattr(kspace_raw, 'dtype') and kspace_raw.dtype.names:
                kspace = kspace_raw['real'] + 1j * kspace_raw['imag']  # Synthetic data
            else:
                kspace = np.array(kspace_raw)  # Real FastMRI data

            if 'reconstruction_rss' in h5_file:
                image = np.array(h5_file['reconstruction_rss'][slice_index])
            elif 'reconstruction_esc' in h5_file:
                image = np.array(h5_file['reconstruction_esc'][slice_index])
            else:
                image = np.abs(
                    np.fft.fftshift(
                        np.fft.ifft2(
                            np.fft.ifftshift(kspace)
                        )
                    )
                )

        return kspace.astype(np.complex64), image.astype(np.float32)   # for memomry efficiency we keep it 8bytes total per complex number and upto 7 decimal digits


    def get_file_info(self, file_index: int) -> dict:         # data inspection without loading data( a dictionary of metadata).
        h5_file_path = self.file_list[file_index]
        
            
        with h5py.File(h5_file_path, 'r') as h5_file:
            num_slices = h5_file['kspace'].shape[0]                # How many 2D images are stacked in this MRI scan ( number of slices)
            kspace_shape = h5_file['kspace'].shape[1:]             # k-space spatial shape dimension.
            if 'reconstruction_rss' in h5_file:
                image_shape = h5_file['reconstruction_rss'].shape[1:]  # image spatial shape dimension.
            elif 'reconstruction_esc' in h5_file:
                image_shape = h5_file['reconstruction_esc'].shape[1:]
            else:
                image_shape = (h5_file['kspace'].shape[1], h5_file['kspace'].shape[2])
            
        return {
            'filename': h5_file_path.name,
            'num_slices': num_slices,
            'kspace_shape': kspace_shape,
            'image_shape': image_shape
        }


    def load_volume(self, file_index: int) -> Tuple[np.ndarray, np.ndarray]:  
        #Loads the full MRI volume from a file by reading all slices of complex k-space and the corresponding reconstructed images.
        
        h5_file_path = self.file_list[file_index]

        with h5py.File(h5_file_path, 'r') as h5_file:
            kspace_raw = h5_file['kspace'][:]                      # Load the entire dataset 
            
            if hasattr(kspace_raw, 'dtype') and kspace_raw.dtype.names:
                kspace_volume = kspace_raw['real'] + 1j * kspace_raw['imag']
            else:
                kspace_volume = np.array(kspace_raw)

            if 'reconstruction_rss' in h5_file:
                image_volume = np.array(h5_file['reconstruction_rss'][:])
            elif 'reconstruction_esc' in h5_file:
                image_volume = np.array(h5_file['reconstruction_esc'][:])
            else:
                image_volume = np.abs(
                    np.fft.fftshift(
                        np.fft.ifft2(
                            np.fft.ifftshift(kspace_volume, axes=(-2, -1)),
                            axes=(-2, -1)
                        ),
                        axes=(-2, -1)
                    )
                )
            
        logger.info(f"Loaded volume {h5_file_path.name}: "
               f"{kspace_volume.shape[0]} slices")                 # Confirms volume size and missing slices 


        return kspace_volume.astype(np.complex64), image_volume.astype(np.float32)


def normalize_image(image: np.ndarray, percentile: float = 99) -> np.ndarray: # percentile based normalization
    
    max_val = np.percentile(image, 99)           # Find the percentile value
    image_clipped = np.clip(image, 0, max_val)   # clip the outliers
    
    if max_val > 0:                              # edge case hanlding for ex- A complete balck image ( already 0 )
        image_normalized = image_clipped / max_val    # Normalize to [0, 1]
    else:
        image_normalized = image_clipped

    return image_normalized.astype(np.float32)



def center_crop(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:    # To make all teh imges the same size for batch processing

    height, width = image.shape[-2:]
    
    if target_height > height or target_width > width:
        raise ValueError(f"Target size ({target_height}, {target_width}) "
                        f"larger than image ({height}, {width})")            # if croping size larger than the image size, we raise an error.
        

    h_start = (height - target_height) // 2      
    w_start = (width - target_width) // 2          # This computes how much extra space exists, then splits it equally on both sides. ( crop start indices )

    
    h_end = h_start + target_height
    w_end = w_start + target_width                 # compute crop end indices


    if image.ndim == 2:                             # Apply crop for 2D image (H, W)
        return image[h_start:h_end, w_start:w_end] 

    elif image.ndim == 3:                           # Apply crop for 3D image (slice, H, W)
        return image[:, h_start:h_end, w_start:w_end]

    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


if __name__ == "__main__":                        # Test the FASTMRI loader.

    import matplotlib.pyplot as plt
    
    print("="*60)
    print("FastMRI Data Loader Test")
    print("="*60)
    
    
    data_path = "data/singlecoil_val"    # Initialize loader
    
    try:
        loader = FastMRILoader(data_path)
        print(f"\n✓ Successfully initialized loader")
        print(f"  Found {len(loader.file_list)} volumes")
        
        # Get info about first file
        info = loader.get_file_info(0)
        print(f"\n✓ First volume info:")
        print(f"  Filename: {info['filename']}")
        print(f"  Number of slices: {info['num_slices']}")
        print(f"  K-space shape: {info['kspace_shape']}")
        print(f"  Image shape: {info['image_shape']}")
        
        # Load a single slice (middle slice of first volume)
        middle_slice = info['num_slices'] // 2
        kspace, image = loader.load_slice(file_index=0, slice_index=middle_slice)
        
        print(f"\n✓ Successfully loaded slice {middle_slice}")
        print(f"  K-space dtype: {kspace.dtype}")
        print(f"  K-space shape: {kspace.shape}")
        print(f"  K-space range: [{np.abs(kspace).min():.2e}, {np.abs(kspace).max():.2e}]")
        print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
        
        # Normalize and visualize
        image_norm = normalize_image(image)
        
        # Create visualization - 3 sections = k-space magnitude(log), Original image, Normalized image
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # K-space magnitude (log scale for visibility)
        axes[0].imshow(np.log(np.abs(kspace) + 1), cmap='gray')
        axes[0].set_title('K-Space (log magnitude)')
        axes[0].axis('off')
        
        # Original image
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title('Original Image')
        axes[1].axis('off')
        
        # Normalized image
        axes[2].imshow(image_norm, cmap='gray')
        axes[2].set_title('Normalized Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/fastmri_loader_test.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: outputs/fastmri_loader_test.png")
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check that fastMRI data is in: data/singlecoil_val")
        print("2. Verify .h5 files exist in that directory")
        print("3. Ensure you have h5py installed: pip install h5py")
