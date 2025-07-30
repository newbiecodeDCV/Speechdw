import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioClassifier, self).__init__()

        # Feature extraction
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),  # thêm dropout sớm

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Attention: dạng Squeeze-and-Excitation đơn giản
        self.attention = nn.Sequential(
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)           # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 128)
        attn = self.attention(x)         # (B, 128)
        x = x * attn                     # Attention
        x = self.classifier(x)
        return x
