import csv
import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from src.baselines import baseline_deblur, baseline_denoise
from src.degradation import add_gaussian_noise, apply_gaussian_blur, load_and_preprocess_image
from src.metrics import MetricsCalculator
from src.model import ImageRestorationUNet


def load_trained_model(weight_path: str, device: torch.device) -> ImageRestorationUNet:
    """
    Wczytuje wytrenowaną architekturę ResUNet z pliku .pth.
    Parametr map_location pozwala wczytać na CPU model wytrenowany w Colab na GPU.
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Brak pliku wag modelu: {weight_path}")

    model = ImageRestorationUNet(base_filters=48).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_network_inference(model: ImageRestorationUNet, img_np: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Przepuszcza pojedynczy obraz numpy (256, 256, 3) przez sieć PyTorch
    i zwraca naprawiony obraz numpy (256, 256, 3) w zakresie <0.0, 1.0>.
    """
    with torch.no_grad():
        tensor_in = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        tensor_out = model(tensor_in)
        img_out = tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(img_out, 0.0, 1.0).astype(np.float32)


def format_table(title: str, results: dict) -> str:
    lines = [f"\n=== {title} ==="]
    header = f"{'Metoda':<45} | {'SNE':>10} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for method_name, m in results.items():
        lines.append(
            f"{method_name:<45} | {m['SNE']:>10.2f} | {m['PSNR']:>8.2f} | {m['SSIM']:>8.4f} | {m['LPIPS']:>8.4f}"
        )
    return "\n".join(lines)


def main():
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uruchamianie pełnej ewaluacji na urządzeniu: {device}")

    image_paths = sorted(glob.glob("data/valid/**/*.png", recursive=True))
    if not image_paths:
        print("BŁĄD: Brak zdjęć w 'data/valid/'.")
        return

    # 1. Wczytanie 4 wytrenowanych modeli z folderu models/
    models = {
        "noise_0_01": load_trained_model("models/resunet_gaussian_noise_0_01.pth", device),
        "noise_0_03": load_trained_model("models/resunet_gaussian_noise_0_03.pth", device),
        "blur_3_x_3": load_trained_model("models/resunet_gaussian_blur_3_x_3.pth", device),
        "blur_5_x_5": load_trained_model("models/resunet_gaussian_blur_5_x_5.pth", device),
    }

    calculator = MetricsCalculator()

    # Lista metod do tabeli 1 (Zadanie 2.2 - Odszumianie)
    denoising_methods = [
        "degraded_gaussian_noise_0_01",
        "denoise_bilateral_gaussian_noise_0_01",
        "resunet_gaussian_noise_0_01",
        "degraded_gaussian_noise_0_03",
        "denoise_bilateral_gaussian_noise_0_03",
        "resunet_gaussian_noise_0_03",
    ]

    # Lista metod do tabeli 2 (Zadanie 2.3 - Deblurowanie)
    deblurring_methods = [
        "degraded_gaussian_blur_3_x_3",
        "richardson_lucy_gaussian_blur_3_x_3",
        "resunet_gaussian_blur_3_x_3",
        "degraded_gaussian_blur_5_x_5",
        "richardson_lucy_gaussian_blur_5_x_5",
        "resunet_gaussian_blur_5_x_5",
    ]

    all_methods = denoising_methods + deblurring_methods
    accumulated = {m: {"SNE": [], "PSNR": [], "SSIM": [], "LPIPS": []} for m in all_methods}

    # 2. Pętla po wszystkich 100 obrazach testowych DIV2K
    for path in tqdm(image_paths, desc="Ewaluacja (Baseline vs ResUNet)"):
        gt_img = load_and_preprocess_image(path, size=256)

        # Zadanie 2.2: Odszumianie (sigma = 0.01)
        noisy_001 = add_gaussian_noise(gt_img, sigma=0.01)
        base_den_001 = baseline_denoise(noisy_001, sigma=0.01)
        net_den_001 = run_network_inference(models["noise_0_01"], noisy_001, device)

        # Zadanie 2.2: Odszumianie (sigma = 0.03)
        noisy_003 = add_gaussian_noise(gt_img, sigma=0.03)
        base_den_003 = baseline_denoise(noisy_003, sigma=0.03)
        net_den_003 = run_network_inference(models["noise_0_03"], noisy_003, device)

        # Zadanie 2.3: Deblurowanie (kernel = 3x3)
        blur_3 = apply_gaussian_blur(gt_img, kernel_size=3)
        base_deb_3 = baseline_deblur(blur_3, kernel_size=3)
        net_deb_3 = run_network_inference(models["blur_3_x_3"], blur_3, device)

        # Zadanie 2.3: Deblurowanie (kernel = 5x5)
        blur_5 = apply_gaussian_blur(gt_img, kernel_size=5)
        base_deb_5 = baseline_deblur(blur_5, kernel_size=5)
        net_deb_5 = run_network_inference(models["blur_5_x_5"], blur_5, device)

        predictions = {
            "degraded_gaussian_noise_0_01": noisy_001,
            "denoise_bilateral_gaussian_noise_0_01": base_den_001,
            "resunet_gaussian_noise_0_01": net_den_001,
            "degraded_gaussian_noise_0_03": noisy_003,
            "denoise_bilateral_gaussian_noise_0_03": base_den_003,
            "resunet_gaussian_noise_0_03": net_den_003,
            "degraded_gaussian_blur_3_x_3": blur_3,
            "richardson_lucy_gaussian_blur_3_x_3": base_deb_3,
            "resunet_gaussian_blur_3_x_3": net_deb_3,
            "degraded_gaussian_blur_5_x_5": blur_5,
            "richardson_lucy_gaussian_blur_5_x_5": base_deb_5,
            "resunet_gaussian_blur_5_x_5": net_deb_5,
        }

        for name, pred_img in predictions.items():
            metrics = calculator.compute(pred_img, gt_img)
            for k, v in metrics.items():
                accumulated[name][k].append(v)

    # 3. Uśrednianie wyników na całym zbiorze testowym
    avg_denoising = {
        m: {k: float(np.mean(accumulated[m][k])) for k in ["SNE", "PSNR", "SSIM", "LPIPS"]} for m in denoising_methods
    }
    avg_deblurring = {
        m: {k: float(np.mean(accumulated[m][k])) for k in ["SNE", "PSNR", "SSIM", "LPIPS"]} for m in deblurring_methods
    }

    report_lines = [
        "FINALNE WYNIKI EKSPERYMENTÓW NA ZBIORZE TESTOWYM DIV2K (256x256)",
        format_table("Tabela 1: Zadanie 2.2 - Odszumianie", avg_denoising),
        format_table("Tabela 2: Zadanie 2.3 - Deblurowanie", avg_deblurring),
    ]
    report_text = "\n".join(report_lines)
    print(report_text)

    os.makedirs("results", exist_ok=True)
    with open("results/final_tables.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open("results/final_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Zadanie", "Metoda", "SNE", "PSNR", "SSIM", "LPIPS"])
        for m, vals in avg_denoising.items():
            writer.writerow([
                "2.2 Odszumianie",
                m,
                f"{vals['SNE']:.4f}",
                f"{vals['PSNR']:.4f}",
                f"{vals['SSIM']:.4f}",
                f"{vals['LPIPS']:.4f}",
            ])
        for m, vals in avg_deblurring.items():
            writer.writerow([
                "2.3 Deblurowanie",
                m,
                f"{vals['SNE']:.4f}",
                f"{vals['PSNR']:.4f}",
                f"{vals['SSIM']:.4f}",
                f"{vals['LPIPS']:.4f}",
            ])

    print("\nSukces! Finalne tabele zapisano w 'results/final_tables.txt' oraz 'results/final_metrics.csv'.")


if __name__ == "__main__":
    main()
