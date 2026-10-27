import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


class MetricsCalculator:
    """
    Klasa obliczająca metryki jakości obrazu: SNE, PSNR, SSIM oraz LPIPS.
    Inicjalizuje sieć AlexNet (dla LPIPS) tylko raz w pamięci komputera,
    dzięki czemu ewaluacja 100 zdjęć działa szybko nawet na CPU
    """

    def __init__(self):
        # Automatyczny wybór urządzenia: karta NVIDIA (cuda) lub procesor (cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MetricsCalculator] Inicjalizacja metryki LPIPS na urządzeniu: {self.device}")

        # Wczytanie gotowej sieci AlexNet do oceny percepcyjnej LPIPS
        self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(self.device)
        self.lpips_model.eval()

    def compute(self, pred_img: np.ndarray, gt_img: np.ndarray) -> dict:
        """
        Oblicza metryki pomiędzy obrazem przetworzonym (pred_img) a oryginałem (gt_img).
        Oba obrazy muszą być tablicami numpy (256, 256, 3) o wartościach z przedziału <0.0, 1.0>.
        """
        # Upewniamy się, że wartości mieszczą się dokładnie w <0.0, 1.0>
        pred_img = np.clip(pred_img, 0.0, 1.0).astype(np.float32)
        gt_img = np.clip(gt_img, 0.0, 1.0).astype(np.float32)

        # SNE (Squared Norm Error)
        sne_val = float(np.sum((gt_img - pred_img) ** 2))

        # PSNR (Peak Signal-to-Noise Ratio) w decybelach [dB]
        psnr_val = float(psnr_fn(gt_img, pred_img, data_range=1.0))

        # SSIM (Structural Similarity Index) w przedziale [0, 1]
        ssim_val = float(ssim_fn(gt_img, pred_img, data_range=1.0, channel_axis=-1))

        # LPIPS (Learned Perceptual Image Patch Similarity)
        # Biblioteka LPIPS wymaga tensorów o wymiarach (1, 3, H, W) znormalizowanych do [-1.0, 1.0]
        pred_tensor = self._numpy_to_lpips_tensor(pred_img)
        gt_tensor = self._numpy_to_lpips_tensor(gt_img)

        with torch.no_grad():
            lpips_val = float(self.lpips_model(pred_tensor, gt_tensor).item())

        return {"SNE": sne_val, "PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val}

    def _numpy_to_lpips_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        # Zamiana osi z (H, W, C) na (C, H, W), dodanie wymiaru batcha (1, C, H, W)
        # oraz przeskalowanie wartości z <0.0, 1.0> na <-1.0, 1.0>
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor * 2.0) - 1.0
