from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FlowConfig:
    residual_weight: float = 0.5
    head_reducer: str = "median"


@dataclass(frozen=True)
class CorruptionConfig:
    prompt_suppression: float = 0.45
    locality_strength: float = 1.0
    local_window: float = 4.0
    concentration_power: float = 1.8


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 48
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    margin: float = 0.5
    origin_regularization: float = 0.02
    validation_fraction: float = 0.20
    patience: int = 12
    seed: int = 42


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int = 8
    torch_threads: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    attention_root: str
    ragtruth_root: str
    output_dir: str
    device: str = "cuda"
    resume: bool = True
    flow: FlowConfig = field(default_factory=FlowConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            attention_root=raw["attention_root"],
            ragtruth_root=raw["ragtruth_root"],
            output_dir=raw["output_dir"],
            device=raw.get("device", "cuda"),
            resume=raw.get("resume", True),
            flow=FlowConfig(**raw.get("flow", {})),
            corruption=CorruptionConfig(**raw.get("corruption", {})),
            training=TrainConfig(**raw.get("training", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
