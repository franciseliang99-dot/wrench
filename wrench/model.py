"""1D convolutional autoencoder over windows of 1-minute sensor means. Anomaly score = reconstruction error."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

WINDOW = 60  # minutes per window


class ConvAE(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(channels, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 8, 5, stride=1, padding=2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, channels, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, channels, WINDOW)
        return self.dec(self.enc(x))


def build(channels: int, seed: int = 0) -> ConvAE:
    """Seed first, then construct: weight initialisation draws from the RNG."""
    torch.manual_seed(seed)
    return ConvAE(channels)


def windows(z: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut WINDOW-long windows ending at each index in `ends` (inclusive).

    z: (minutes, channels), NaN where data is missing. Windows containing any NaN are dropped.
    Returns (windows (n, channels, WINDOW), kept end indices).
    """
    ends = ends[ends >= WINDOW - 1]
    idx = ends[:, None] - np.arange(WINDOW - 1, -1, -1)[None, :]
    w = z[idx]  # (n, WINDOW, channels)
    ok = ~np.isnan(w).any(axis=(1, 2))
    return np.ascontiguousarray(w[ok].transpose(0, 2, 1), dtype=np.float32), ends[ok]


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    # ROCm builds of PyTorch expose AMD GPUs through the cuda device type
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model: ConvAE, x: np.ndarray, *, epochs: int, device: torch.device, seed: int = 0, batch: int = 512, log=print) -> list[float]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.from_numpy(x)
    losses = []
    for ep in range(epochs):
        order = rng.permutation(len(data))
        total = 0.0
        for i in range(0, len(order), batch):
            xb = data[order[i:i + batch]].to(device)
            loss = torch.mean((model(xb) - xb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        losses.append(total / len(data))
        log(f"epoch {ep + 1}/{epochs} loss {losses[-1]:.5f}")
    return losses


@torch.no_grad()
def score(model: ConvAE, x: np.ndarray, device: torch.device, batch: int = 4096) -> np.ndarray:
    model.to(device).eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device)
        out.append(torch.mean((model(xb) - xb) ** 2, dim=(1, 2)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
