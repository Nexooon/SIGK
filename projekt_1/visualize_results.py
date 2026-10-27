import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from evaluate_all import load_trained_model, run_network_inference
from src.baselines import baseline_deblur, baseline_denoise
from src.degradation import add_gaussian_noise, apply_gaussian_blur, load_and_preprocess_image


def get_quick_metrics(pred: np.ndarray, gt: np.ndarray) -> str:
    psnr = psnr_fn(gt, pred, data_range=1.0)
    ssim = ssim_fn(gt, pred, data_range=1.0, channel_axis=-1)
    return f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}"


def generate_comparison_board(image_path: str, sample_idx: int, models: dict, device: torch.device):
    gt_img = load_and_preprocess_image(image_path, size=256)

    # Przygotowanie wszystkich 4 przypadków
    noisy_001 = add_gaussian_noise(gt_img, sigma=0.01)
    base_den_001 = baseline_denoise(noisy_001, sigma=0.01)
    net_den_001 = run_network_inference(models["noise_0_01"], noisy_001, device)

    noisy_003 = add_gaussian_noise(gt_img, sigma=0.03)
    base_den_003 = baseline_denoise(noisy_003, sigma=0.03)
    net_den_003 = run_network_inference(models["noise_0_03"], noisy_003, device)

    blur_3 = apply_gaussian_blur(gt_img, kernel_size=3)
    base_deb_3 = baseline_deblur(blur_3, kernel_size=3)
    net_deb_3 = run_network_inference(models["blur_3_x_3"], blur_3, device)

    blur_5 = apply_gaussian_blur(gt_img, kernel_size=5)
    base_deb_5 = baseline_deblur(blur_5, kernel_size=5)
    net_deb_5 = run_network_inference(models["blur_5_x_5"], blur_5, device)

    rows_data = [
        (
            "2.2 Odszumianie (sigma=0.01)",
            noisy_001,
            "denoise_bilateral",
            base_den_001,
            "ResUNet (Nasza sieć)",
            net_den_001,
        ),
        (
            "2.2 Odszumianie (sigma=0.03)",
            noisy_003,
            "denoise_bilateral",
            base_den_003,
            "ResUNet (Nasza sieć)",
            net_den_003,
        ),
        ("2.3 Deblurowanie (kernel 3x3)", blur_3, "richardson_lucy", base_deb_3, "ResUNet (Nasza sieć)", net_deb_3),
        ("2.3 Deblurowanie (kernel 5x5)", blur_5, "richardson_lucy", base_deb_5, "ResUNet (Nasza sieć)", net_deb_5),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(
        f"Porównanie wizualne metod - Obraz testowy #{sample_idx + 1} ({os.path.basename(image_path)})",
        fontsize=16,
        fontweight="bold",
    )

    for r, (deg_name, deg_img, base_name, base_img, net_name, net_img) in enumerate(rows_data):
        # Kolumna 1: Obraz zdegradowany (wejście)
        axes[r, 0].imshow(deg_img)
        axes[r, 0].set_title(f"Wejście: {deg_name}\n{get_quick_metrics(deg_img, gt_img)}", fontsize=10)
        axes[r, 0].axis("off")

        # Kolumna 2: Metoda bazowa
        axes[r, 1].imshow(base_img)
        axes[r, 1].set_title(f"Baseline: {base_name}\n{get_quick_metrics(base_img, gt_img)}", fontsize=10)
        axes[r, 1].axis("off")

        # Kolumna 3: Nasza sieć (ResUNet)
        axes[r, 2].imshow(net_img)
        axes[r, 2].set_title(
            f"{net_name}\n{get_quick_metrics(net_img, gt_img)}", fontsize=10, color="darkgreen", fontweight="bold"
        )
        axes[r, 2].axis("off")

        # Kolumna 4: Oryginał (Ground Truth)
        axes[r, 3].imshow(gt_img)
        axes[r, 3].set_title("Oryginał (Ground Truth)\n256x256", fontsize=10)
        axes[r, 3].axis("off")

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    out_path = f"results/comparison_sample_{sample_idx + 1}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Zapisano planszę wizualną: {out_path}")


def main():
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = sorted(glob.glob("data/valid/**/*.png", recursive=True))
    if not image_paths:
        print("BŁĄD: Brak zdjęć w 'data/valid/'.")
        return

    models = {
        "noise_0_01": load_trained_model("models/resunet_gaussian_noise_0_01.pth", device),
        "noise_0_03": load_trained_model("models/resunet_gaussian_noise_0_03.pth", device),
        "blur_3_x_3": load_trained_model("models/resunet_gaussian_blur_3_x_3.pth", device),
        "blur_5_x_5": load_trained_model("models/resunet_gaussian_blur_5_x_5.pth", device),
    }

    os.makedirs("results", exist_ok=True)
    # wybieramy 3 zdjęcia ze zbioru walidacyjnego (np. indeksy 0, 5, 12)
    sample_indices = [0, 5, 12]
    for i, idx in enumerate(sample_indices):
        if idx < len(image_paths):
            generate_comparison_board(image_paths[idx], i, models, device)

    print("\nGotowe! Wszystkie wizualizacje znajdują się w folderze 'results/'.")


if __name__ == "__main__":
    main()
