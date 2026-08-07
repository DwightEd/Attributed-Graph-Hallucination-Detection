from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RobustScaler:
    median: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, trajectories: list[torch.Tensor]) -> "RobustScaler":
        values = torch.cat(trajectories, dim=0).float()
        median = values.median(dim=0).values
        scale = (
            torch.quantile(values, 0.75, dim=0)
            - torch.quantile(values, 0.25, dim=0)
        ).clamp_min(1e-4)
        return cls(median, scale)

    def transform(self, trajectory: torch.Tensor) -> torch.Tensor:
        return (trajectory.float() - self.median) / self.scale


class TrajectoryRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        sequence, hidden = self.encoder(self.project(trajectory))
        summary = torch.cat(
            (sequence[:, 0], hidden[-1], sequence[:, -1] - sequence[:, 0]),
            dim=-1,
        )
        return self.score(summary).squeeze(-1)
