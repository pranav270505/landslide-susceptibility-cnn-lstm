"""CNN branch, LSTM branch and the late-fusion classifier.

mode = "hybrid"  CNN + LSTM, concatenated (the proposed model)
mode = "cnn"     terrain patch only        (ablation)
mode = "lstm"    rainfall sequence only    (ablation)
"""
import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    """3 conv blocks (32 -> 64 -> 128 filters, 3x3, BatchNorm, ReLU, 2x2 max-pool), flattened."""

    def __init__(self, in_channels, patch_size):
        super().__init__()
        layers, c = [], in_channels
        for f in (32, 64, 128):
            layers += [nn.Conv2d(c, f, 3, padding=1), nn.BatchNorm2d(f), nn.ReLU(), nn.MaxPool2d(2)]
            c = f
        self.net = nn.Sequential(*layers, nn.Flatten())
        self.out_dim = 128 * (patch_size // 8) ** 2

    def forward(self, x):
        return self.net(x)


class LSTMBranch(nn.Module):
    """2 stacked LSTM layers (128 -> 64 units); the final hidden state is the embedding."""

    def __init__(self, in_features=1, dropout=0.2):
        super().__init__()
        self.l1 = nn.LSTM(in_features, 128, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.l2 = nn.LSTM(128, 64, batch_first=True)
        self.out_dim = 64

    def forward(self, x):
        out, _ = self.l1(x)
        _, (h, _) = self.l2(self.drop(out))
        return h[-1]


class LandslideNet(nn.Module):
    def __init__(self, in_channels, patch_size, mode="hybrid", dropout=0.3):
        super().__init__()
        assert mode in ("hybrid", "cnn", "lstm")
        self.mode = mode
        dim = 0
        if mode in ("hybrid", "cnn"):
            self.cnn = CNNBranch(in_channels, patch_size)
            dim += self.cnn.out_dim
        if mode in ("hybrid", "lstm"):
            self.lstm = LSTMBranch()
            dim += self.lstm.out_dim
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),          # sigmoid is applied in the loss / at prediction time
        )

    def forward(self, patch, seq):
        parts = []
        if self.mode in ("hybrid", "cnn"):
            parts.append(self.cnn(patch))
        if self.mode in ("hybrid", "lstm"):
            parts.append(self.lstm(seq))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)
