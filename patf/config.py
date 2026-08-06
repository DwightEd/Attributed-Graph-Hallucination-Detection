from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TopologyConfig:
    mass_cover: float = 0.80
    relay_discount: float = 0.85
    head_reducer: str = "median"


@dataclass(frozen=True)
class CorruptionConfig:
    modes: tuple[str, ...] = ("incidence", "collapse", "composite")
    prompt_transfer: float = 0.45
    local_window: int = 4
    support_keep_fraction: float = 0.55
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
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        corruption = dict(raw.get("corruption", {}))
        corruption["modes"] = tuple(
            corruption.get("modes", CorruptionConfig().modes)
        )
        return cls(
            attention_root=raw["attention_root"],
            ragtruth_root=raw["ragtruth_root"],
            output_dir=raw["output_dir"],
            device=raw.get("device", "cuda"),
            resume=raw.get("resume", True),
            topology=TopologyConfig(**raw.get("topology", {})),
            corruption=CorruptionConfig(**corruption),
            training=TrainConfig(**raw.get("training", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
