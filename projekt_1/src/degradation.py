import cv2
import numpy as np
from skimage.util import random_noise


def load_and_preprocess_image(image_path: str, size: int = 256) -> np.ndarray:
    """
    Wczytuje obraz z dysku, zmienia rozmiar na 256x256 i normalizuje do przedziału <0.0, 1.0>.
    Zwraca tablicę numpy o wymiarach (256, 256, 3) typu float32.
    """
    # OpenCV domyślnie wczytuje kolory w kolejności BGR, zamieniamy na standardowe RGB
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu pod ścieżką: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Skalowanie obrazu do wymaganej w projekcie rozdzielczości 256x256
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

    # Normalizacja z wartości całkowitych [0, 255] do zmiennoprzecinkowych <0.0, 1.0>
    img_norm = img_resized.astype(np.float32) / 255.0
    return img_norm


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Zadanie 2.2: Odszumianie - przygotowanie danych.
    Dodaje szum gaussowski o odchyleniu standardowym 'sigma' (0.01 lub 0.03).
    """
    variance = sigma**2
    noisy_img = random_noise(img, mode="gaussian", var=variance, clip=True)
    return noisy_img.astype(np.float32)


def apply_gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Zadanie 2.3: Deblurowanie - przygotowanie danych.
    Aplikuje rozmycie gaussowskie z kernelem o wymiarach (kernel_size, kernel_size), np. 3x3 lub 5x5.
    """
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)
    return blurred_img.astype(np.float32)
