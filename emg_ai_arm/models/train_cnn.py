"""
1-D Convolutional Neural Network baseline for EMG window classification.

Operates directly on (SAMPLES, C) windows — no hand-crafted features —
which lets us compare a learned representation against the Random
Forest baseline in train_ml.py for Section 5.4 of the proposal.

The architecture is intentionally small: three Conv1D blocks + global
average pooling + a linear head. Trains in seconds on CPU, ~1 minute
on a laptop GPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run train_cnn.py.\n"
        "Install it from https://pytorch.org/get-started/locally/"
    ) from e

from emg_ai_arm.utils.config import WINDOWS_DIR, MODELS_DIR


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class EMGCNN(nn.Module):
    """
    Tiny 1-D CNN.

    Input  : (B, C, T)   where C = n_channels, T = win_samples
    Output : (B, n_classes)
    """

    def __init__(self, n_channels: int, n_classes: int, base: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, base, kernel_size=5, padding=2),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base, base * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #

def _to_tensor(x: np.ndarray, y: np.ndarray):
    # Xw is (N, T, C) -> torch expects (N, C, T)
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def train(
    data_path: Path,
    out_path: Path,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(str(data_path))
    Xw = data["X"]                  # (N, T, C)
    y = data["y"]                   # (N,)
    n_classes = int(y.max() + 1)
    n_channels = Xw.shape[2]

    # train/val split
    idx = np.random.permutation(len(y))
    n_val = max(1, int(0.2 * len(y)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = _to_tensor(Xw[train_idx], y[train_idx])
    X_val, y_val = _to_tensor(Xw[val_idx], y[val_idx])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EMGCNN(n_channels=n_channels, n_classes=n_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            train_loss += float(loss) * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(1, total)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_channels": n_channels,
                    "n_classes": n_classes,
                    "win_samples": Xw.shape[1],
                },
                str(out_path),
            )

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Saved CNN to: {out_path}")
    return {"best_val_acc": best_acc, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=WINDOWS_DIR / "fake_windows.npz")
    ap.add_argument("--out", type=Path,
                    default=MODELS_DIR / "cnn_model.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    train(args.data, args.out,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr)


if __name__ == "__main__":
    main()
