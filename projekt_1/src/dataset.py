import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.degradation import add_gaussian_noise, apply_gaussian_blur


class DIV2KRestorationDataset(Dataset):
    """
    Dataset ładujący obrazy DIV2K w rozdzielczości 256x256 i generujący w locie
    pary (obraz_zdegradowany, obraz_oryginalny) dla wybranego zadania.
    """

    def __init__(self, data_dir: str, task: str, param: float, is_train: bool = True):
        """
        data_dir: ścieżka do folderu ze zdjęciami (np. 'data/train' lub 'data/valid')
        task: 'denoising' (Zadanie 2.2) lub 'deblurring' (Zadanie 2.3)
        param: dla 'denoising' -> 0.01 lub 0.03; dla 'deblurring' -> 3 lub 5
        is_train: jeśli True, losuje wycinki 256x256 i obraca je (augmentacja danych)
        """
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True))
        if not self.image_paths:
            raise FileNotFoundError(f"Brak plików .png w katalogu: {data_dir}")

        self.task = task
        self.param = param
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_bgr = cv2.imread(self.image_paths[idx])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Przygotowanie obrazu 256x256
        if self.is_train:
            # Podczas treningu wycinamy losowy fragment 256x256 z dużego zdjęcia HD,
            # dzięki czemu sieć w każdej epoce widzi nieco inne detale i lepiej się uczy
            h, w, _ = img_rgb.shape
            if h >= 256 and w >= 256:
                top = random.randint(0, h - 256)
                left = random.randint(0, w - 256)
                patch = img_rgb[top : top + 256, left : left + 256]
            else:
                patch = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

            # losowe odbicie lustrzane (wzbogacenie zbioru treningowego)
            if random.random() > 0.5:
                patch = np.fliplr(patch).copy()
        else:
            # Podczas walidacji zawsze skalujemy całe zdjęcie deterministycznie do 256x256
            patch = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

        # Normalizacja do <0.0, 1.0>
        gt_img = patch.astype(np.float32) / 255.0

        # Aplikacja wybranej degradacji
        if self.task == "denoising":
            degraded_img = add_gaussian_noise(gt_img, sigma=float(self.param))
        elif self.task == "deblurring":
            degraded_img = apply_gaussian_blur(gt_img, kernel_size=int(self.param))
        else:
            raise ValueError(f"Nieznane zadanie: {self.task}")

        # Zamiana wymiarów z (H, W, C) na (C, H, W), wymagane przez PyTorch
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float()
        degraded_tensor = torch.from_numpy(degraded_img).permute(2, 0, 1).float()

        return degraded_tensor, gt_tensor
