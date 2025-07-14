# models/encoder.py

import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # (B, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B, 64, 8, 8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # (B, 128, 8, 8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                           # (B, 128, 1, 1)
        )
        self.fc = nn.Linear(128, latent_dim)                        # Final latent vector

    def forward(self, x):
        x = self.encoder(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 128)
        return self.fc(x)          # (B, latent_dim)
