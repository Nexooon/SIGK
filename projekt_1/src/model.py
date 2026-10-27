import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Podstawowy blok filtrów konwolucyjnych z połączeniem skrótowym (skip connection).
    Pozwala na głębokie przetwarzanie cech bez zanikania sygnału podczas nauki.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class ImageRestorationUNet(nn.Module):
    """
    Uniwersalna architektura typu Residual U-Net dla zadań modyfikacji obrazów (256x256).
    Przyjmuje tensor (Batch, 3, 256, 256) w zakresie <0.0, 1.0> i zwraca naprawiony obraz
    o tych samych wymiarach.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 48):
        super().__init__()

        # Poziom 1 (256x256)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_filters),
        )
        self.down1 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1)

        # Poziom 2 (128x128)
        self.enc2 = ResidualBlock(base_filters * 2)
        self.down2 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1)

        # Dno sieci - Bottleneck (64x64)
        self.bottleneck = nn.Sequential(ResidualBlock(base_filters * 4), ResidualBlock(base_filters * 4))

        # Dekoder Poziom 2 (powrót do 128x128)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_filters * 2),
        )

        # Dekoder Poziom 1 (powrót do 256x256)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_filters),
        )

        # Warstwa wyjściowa generująca mapę poprawek (residual)
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # kodowanie (ekstrakcja cech)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        # dekodowanie (rekonstrukcja z połączeniami z enkodera)
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # obliczenie poprawki i dodanie jej do obrazu wejściowego
        correction = self.final_conv(d1)
        return torch.clamp(x + correction, 0.0, 1.0)
