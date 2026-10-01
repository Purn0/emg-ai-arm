"""
1D CNN baseline on raw windows (samples x 8 channels), as in the thesis:
three convolutional blocks (32, 64, 128 filters) with batch normalisation,
global average pooling and a linear layer, trained with cross-entropy
weighted by the square root of inverse class frequency. Input channels are
standardised with training-set statistics (or already calibrated per session
by the caller). Fixed schedule, no early stopping: 15 epochs, Adam, lr 1e-3.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn


class EMGNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        def block(i, o, k):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]

        self.net = nn.Sequential(
            *block(n_channels, 32, 7), nn.MaxPool1d(2),
            *block(32, 64, 5), nn.MaxPool1d(2),
            *block(64, 128, 3), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def fit_predict_cnn(Xtr, ytr, Xte, seed=42, weighted=True, epochs=15, batch=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)

    classes = np.unique(ytr)
    remap = {k: i for i, k in enumerate(classes)}
    y = torch.tensor([remap[k] for k in ytr], dtype=torch.long)

    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    to_tensor = lambda X: torch.tensor(((X - mu) / sd).transpose(0, 2, 1), dtype=torch.float32)
    xtr, xte = to_tensor(Xtr), to_tensor(Xte)

    counts = np.bincount(y.numpy(), minlength=len(classes)).astype(np.float64)
    w = np.sqrt(counts.sum() / counts) if weighted else np.ones(len(classes))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32))

    model = EMGNet(xtr.shape[1], len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(y), generator=g).split(batch):
            opt.zero_grad()
            loss_fn(model(xtr[idx]), y[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(xb).argmax(1) for xb in xte.split(1024)]).numpy()
    return classes[pred]
