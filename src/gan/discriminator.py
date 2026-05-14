import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def build_discriminator(img_shape=(1, 128, 128)):
    return Discriminator()

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 128 -> 64
            spectral_norm(nn.Conv2d(1, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # 64 -> 32
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # 32 -> 16
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # 16 -> 8
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)