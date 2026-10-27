import cv2
import numpy as np
from skimage.restoration import denoise_bilateral, richardson_lucy


def baseline_denoise(noisy_img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Zadanie 2.2: Metoda bazowa - denoise_bilateral z skimage.
    """
    denoised = denoise_bilateral(noisy_img, sigma_color=max(sigma * 2.5, 0.05), sigma_spatial=3, channel_axis=-1)
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


def get_gaussian_psf(kernel_size: int) -> np.ndarray:
    """
    Funkcja pomocnicza dla algorytmu richardson_lucy
    Generuje dwuwymiarową macierz kernela Gaussa (Point Spread Function),
    identyczną z tą, której używa cv2.GaussianBlur.
    """
    kernel_1d = cv2.getGaussianKernel(kernel_size, 0)
    kernel_2d = np.outer(kernel_1d, kernel_1d.transpose())
    return kernel_2d.astype(np.float32)


def baseline_deblur(blurred_img: np.ndarray, kernel_size: int, num_iter: int = 15) -> np.ndarray:
    """
    Zadanie 2.3: Metoda bazowa - richardson_lucy z skimage
    Algorytm aplikujemy osobno dla każdego z 3 kanałów kolorów (R, G, B)
    """
    psf = get_gaussian_psf(kernel_size)
    restored = np.zeros_like(blurred_img, dtype=np.float32)

    for c in range(3):
        restored[:, :, c] = richardson_lucy(blurred_img[:, :, c], psf=psf, num_iter=num_iter, clip=True)

    return np.clip(restored, 0.0, 1.0).astype(np.float32)
