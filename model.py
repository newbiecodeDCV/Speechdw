import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioClassifier, self).__init__()

        # Feature extraction blocks
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 1, T, F) -> (B, 32, T, F)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (B, 32, T/2, F/2)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (B, 64, T/4, F/4)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )

        # Attention block (Squeeze-and-Excitation style)
        self.attention = nn.Sequential(
            nn.Flatten(),  # (B, 128)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        attn = self.attention(x)  # (B, 128)
        x = x * attn  # Apply attention
        x = self.classifier(x)
        return x
