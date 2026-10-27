import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DIV2KRestorationDataset
from src.model import ImageRestorationUNet


def train_single_variant(task: str, param, model_name: str, train_dir: str, epochs: int = 15, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==========================================")
    print(f"Trening modelu: {model_name}")
    print(f"Zadanie: {task} | Parametr: {param} | Urządzenie: {device}")
    print("==========================================")

    dataset = DIV2KRestorationDataset(data_dir=train_dir, task=task, param=param, is_train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = ImageRestorationUNet(base_filters=48).to(device)

    # Funkcja straty L1 + MSE daje ostre krawędzie i wysoki wynik PSNR/SSIM
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoka {epoch}/{epochs}", leave=False)
        for degraded, gt in pbar:
            degraded = degraded.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            output = model(degraded)

            # Kombinacja straty L1 (ostrość) i MSE (minimalizacja SNE / maksymalizacja PSNR)
            loss = 0.8 * criterion_l1(output, gt) + 0.2 * criterion_mse(output, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * degraded.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        print(f"Epoka [{epoch:02d}/{epochs:02d}] - Średni błąd (Loss): {epoch_loss:.6f}")

    # Zapis wytrenowanych wag modelu do pliku
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"-> Zapisano wagi modelu w: {save_path}")

    return loss_history


def main():
    train_dir = "data/train"
    print(f"Używany katalog treningowy: {train_dir}")

    experiments = [
        ("denoising", 0.01, "resunet_gaussian_noise_0_01"),
        ("denoising", 0.03, "resunet_gaussian_noise_0_03"),
        ("deblurring", 3, "resunet_gaussian_blur_3_x_3"),
        ("deblurring", 5, "resunet_gaussian_blur_5_x_5"),
    ]

    all_histories = {}
    for task, param, name in experiments:
        history = train_single_variant(
            task=task,
            param=param,
            model_name=name,
            train_dir=train_dir,
            epochs=15,
            batch_size=16,
        )
        all_histories[name] = history

    # wykres przebiegu uczenia
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        plt.plot(range(1, len(hist) + 1), hist, marker="o", label=name)
    plt.title("Przebieg procesu uczenia sieci ResUNet (Funkcja straty)")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji straty (L1 + MSE)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    print("\nSukces! Wykres uczenia zapisano w: results/training_curves.png")


if __name__ == "__main__":
    main()
