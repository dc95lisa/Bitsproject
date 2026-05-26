"""
Custom neural network use case:
Demand forecasting for a grocery retail chain.

This script demonstrates:
- GPU requirement estimation utilities
- A custom PyTorch model (tabular + temporal features)
- Training loop with mixed precision
- Bottleneck-aware production notes embedded as comments
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class GpuEstimate:
    params_millions: float
    model_memory_gb: float
    activation_memory_gb: float
    optimizer_memory_gb: float
    total_estimated_gb: float


def estimate_training_gpu_memory(
    parameter_count: int,
    batch_size: int,
    activation_elements_per_sample: int,
    bytes_per_element: int = 2,
    optimizer_multiplier: float = 2.0,
    gradient_multiplier: float = 1.0,
    safety_factor: float = 1.3,
) -> GpuEstimate:
    """
    Rough training VRAM estimate.

    Assumptions for mixed precision training:
    - Weights in fp16/bf16 + master weights/optimizer states in fp32
    - Optimizer states are roughly 2x parameter memory for Adam-like optimizers
    - Gradient memory ~ parameter memory
    """
    params_mem = parameter_count * bytes_per_element
    activations_mem = batch_size * activation_elements_per_sample * bytes_per_element
    optimizer_mem = params_mem * optimizer_multiplier
    grads_mem = params_mem * gradient_multiplier

    total = (params_mem + activations_mem + optimizer_mem + grads_mem) * safety_factor

    gb = 1024**3
    return GpuEstimate(
        params_millions=parameter_count / 1e6,
        model_memory_gb=params_mem / gb,
        activation_memory_gb=activations_mem / gb,
        optimizer_memory_gb=optimizer_mem / gb,
        total_estimated_gb=total / gb,
    )


class GroceryDemandDataset(Dataset):
    """
    Example dataset shape:
    - static features (store and product attributes)
    - sequence features (previous daily sales)
    - target: next-day sales quantity
    """

    def __init__(self, n_samples: int = 5000, static_dim: int = 32, seq_len: int = 30):
        self.static = torch.randn(n_samples, static_dim)
        base = torch.randn(n_samples, seq_len)
        trend = torch.linspace(0.0, 1.0, steps=seq_len).unsqueeze(0)
        self.sequence = base + trend

        # Synthetic target with nonlinear interaction
        self.target = (
            0.4 * self.static[:, :8].sum(dim=1)
            + 0.8 * self.sequence[:, -7:].mean(dim=1)
            + torch.sin(self.sequence[:, -1])
            + 0.1 * torch.randn(n_samples)
        ).unsqueeze(1)

    def __len__(self) -> int:
        return self.static.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.static[idx], self.sequence[idx], self.target[idx]


class DemandNet(nn.Module):
    """Custom NN combining MLP (static) + temporal encoder (sequence)."""

    def __init__(self, static_dim: int = 32, seq_len: int = 30, hidden: int = 64):
        super().__init__()
        self.static_tower = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.seq_tower = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, static_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        s = self.static_tower(static_x)
        t = self.seq_tower(seq_x)
        return self.head(torch.cat([s, t], dim=1))


def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    criterion = nn.MSELoss()
    running = 0.0

    for static_x, seq_x, y in loader:
        static_x = static_x.to(device)
        seq_x = seq_x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(static_x, seq_x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * static_x.size(0)

    return running / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DemandNet(static_dim=32, seq_len=30, hidden=128).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    estimate = estimate_training_gpu_memory(
        parameter_count=parameter_count,
        batch_size=512,
        activation_elements_per_sample=128 * 8,
    )

    print("\n=== GPU Estimate (rough) ===")
    print(f"Params (M): {estimate.params_millions:.3f}")
    print(f"Model memory (GB): {estimate.model_memory_gb:.4f}")
    print(f"Activation memory (GB): {estimate.activation_memory_gb:.4f}")
    print(f"Optimizer memory (GB): {estimate.optimizer_memory_gb:.4f}")
    print(f"Total estimate (GB): {estimate.total_estimated_gb:.4f}")

    dataset = GroceryDemandDataset(n_samples=12000, static_dim=32, seq_len=30)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, 6):
        loss = train_one_epoch(model, loader, optimizer, device, scaler)
        print(f"Epoch {epoch:02d} | train_loss={loss:.4f}")

    # Production-ready model export placeholder
    model.eval()
    dummy_static = torch.randn(1, 32, device=device)
    dummy_seq = torch.randn(1, 30, device=device)
    with torch.no_grad():
        _ = model(dummy_static, dummy_seq)

    print("Training complete. Next steps: validation, calibration, model registry, A/B rollout.")


if __name__ == "__main__":
    main()
