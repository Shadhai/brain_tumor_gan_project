import torch
import torch.nn as nn

def build_generator(latent_dim=100):
    return Generator(latent_dim)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            # 4x4x512
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 1, 3, 1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # z shape: (batch, latent_dim, 1, 1)
        return self.main(z)