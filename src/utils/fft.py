"""
FFT Utilities for MRI K-Space Operations

FILE: src/utils/fft.py

WE WILL USE THIS FILE FOR:
    MRI data is acquired in "k-space" — the Fourier transform of the image.
    To go between image and k-space when we need FFT (Fast Fourier Transform).

    IMP: 
        • NumPy's fft2() puts the zero-frequency (DC) component in the CORNER
        • MRI convention puts zero-frequency in the CENTER of k-space
        • fftshift() moves corner → center
        • ifftshift() moves center → corner

    So the correct flowchart would be smtg like this :
        Image  →  ifftshift  →  fft2  →  fftshift  →  K-space
        K-space →  ifftshift  →  ifft2 →  fftshift  →  Image
"""

import numpy as np


def fft2c(image: np.ndarray) -> np.ndarray:
    """
    Centered 2D FFT — converts an image into k-space.

    We shift so the zero-frequency
    component sits in the center of k-space, matching MRI convention. ( centre )


        image: 2D numpy array [H, W] — can be real or complex.

    Returns:
        kspace: 2D complex numpy array [H, W].
    """
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(image)
        )
    )


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """
    Centered 2D Inverse FFT — converts k-space back into an image.

    Exact inverse of fft2c. So ifft2c(fft2c(x)) ≈ x (up to float precision).


        kspace: 2D complex numpy array [H, W].

    Returns:
        image: 2D complex numpy array [H, W].
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace)
        )
    )


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """
    Convert k-space to a magnitude image (what we see on an MRI screen).

    After inverse FFT the result is complex. We take abs() to get the
    magnitude — the signal intensity


        kspace: 2D complex numpy array [H, W].

    Returns:
        image: 2D REAL numpy array [H, W] — magnitude values ≥ 0.
    """
    return np.abs(ifft2c(kspace))


def image_to_kspace(image: np.ndarray) -> np.ndarray:
    """
    Convert a real-valued image to k-space.


        image: 2D numpy array [H, W].

    Returns:
        kspace: 2D complex numpy array [H, W].
    """
    return fft2c(image.astype(np.complex128))



# SELF-TEST
if __name__ == "__main__":
    print("=" * 55)
    print("  Testing FFT utilities")
    print("=" * 55)

    # 1) image → kspace → image
    test_img = np.zeros((128, 128), dtype=np.complex128)
    test_img[44:84, 44:84] = 1.0          # a white square in the middle

    kspace   = fft2c(test_img)
    recovered = ifft2c(kspace)

    err = np.max(np.abs(test_img - recovered))
    status = "PASSED" if err < 1e-10 else "FAILED"
    print(f"\n error : {err:.2e}   [{status}]")

    # 2) Magnitude image test
    mag = kspace_to_image(kspace)
    print(f"  Magnitude range  : [{mag.min():.4f}, {mag.max():.4f}]")
    print(f"  Magnitude shape  : {mag.shape}")

    # 3) Verify k-space center has the most energy
    center_h, center_w = kspace.shape[0] // 2, kspace.shape[1] // 2
    center_energy = np.abs(kspace[center_h, center_w])
    total_energy  = np.sqrt(np.sum(np.abs(kspace)**2))
    print(f"  DC component     : {center_energy:.2f} / total energy {total_energy:.2f}")

    print("\n  ✓ All FFT tests passed!")
    print("=" * 55)
