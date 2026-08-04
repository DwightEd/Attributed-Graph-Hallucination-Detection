"""Self-supervised training and label-free two-pattern graph assignment."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .graph import AttentionGraph, RP
from .ablation import relation_preserving_source_shuffle
from .model import (
    MaskedGraphView,
    RelationAwareMaskGAE,
    make_masked_view,
    reconstruction_energy_by_node,
    reconstruction_losses,
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    edge_mask_rate: float = 0.35
    node_mask_rate: float = 0.30
    channel_drop_rate: float = 0.10
    gradient_clip: float = 1.0
    support_weight: float = 1.0
    attention_weight: float = 1.0
    distribution_weight: float = 1.0
    node_weight: float = 0.25
    max_support_edges: int | None = 8_192
    max_weight_traces: int | None = 65_536
    max_distribution_groups: int | None = 512
    decoder_chunk_size: int = 16_384
    seed: int = 42

    def validate(self) -> None:
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        for name in ("edge_mask_rate", "node_mask_rate", "channel_drop_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.edge_mask_rate == 0.0 and self.node_mask_rate == 0.0:
            raise ValueError("at least one graph reconstruction mask must be active")
        for name in (
            "support_weight",
            "attention_weight",
            "distribution_weight",
            "node_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if not any(
            float(getattr(self, name)) > 0.0
            for name in (
                "support_weight",
                "attention_weight",
                "distribution_weight",
                "node_weight",
            )
        ):
            raise ValueError("at least one reconstruction loss weight must be positive")
        for name in (
            "max_support_edges",
            "max_weight_traces",
            "max_distribution_groups",
        ):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when provided")
        if self.decoder_chunk_size < 1:
            raise ValueError("decoder_chunk_size must be positive")


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path


@dataclass(frozen=True)
class TwoComponentMixture:
    """Persistable diagonal-Gaussian mixture with unrestricted component size."""

    feature_median: list[float]
    feature_scale: list[float]
    component_weights: list[float]
    component_means: list[list[float]]
    component_variances: list[list[float]]
    component_direction_means: list[float]
    hallucination_component: int

    def component_probability(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        median = np.asarray(self.feature_median, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        means = np.asarray(self.component_means, dtype=np.float64)
        variances = np.asarray(self.component_variances, dtype=np.float64)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != median.size:
            raise ValueError("mixture features have the wrong dimension")
        standardized = (values - median) / scale
        difference = standardized[:, None, :] - means[None, :, :]
        log_probability = -0.5 * (
            np.log(2.0 * np.pi * variances)[None, :, :]
            + np.square(difference) / variances[None, :, :]
        ).sum(axis=2)
        log_probability += np.log(np.maximum(weights, np.finfo(np.float64).tiny))[None, :]
        log_probability -= log_probability.max(axis=1, keepdims=True)
        probability = np.exp(log_probability)
        return probability / probability.sum(axis=1, keepdims=True)

    def hallucination_probability(
        self, features: np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        return self.component_probability(features)[:, self.hallucination_component]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "attention-graph-free-two-component-mixture-v1",
            **asdict(self),
            "orientation": (
                "component with higher pre-registered structural direction: "
                "lower RP/fewer edges/more local, higher RR/concentration"
            ),
            "prevalence_constraint": None,
        }


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters") from error


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _loss_arguments(config: TrainingConfig) -> dict[str, float | int | None]:
    return {
        "support_weight": config.support_weight,
        "attention_weight": config.attention_weight,
        "distribution_weight": config.distribution_weight,
        "node_weight": config.node_weight,
        "max_support_edges": config.max_support_edges,
        "max_weight_traces": config.max_weight_traces,
        "max_distribution_groups": config.max_distribution_groups,
        "decoder_chunk_size": config.decoder_chunk_size,
    }


def _one_epoch(
    model: RelationAwareMaskGAE,
    graphs: Sequence[AttentionGraph],
    config: TrainingConfig,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    if not graphs:
        raise ValueError("an epoch requires at least one graph")
    training = optimizer is not None
    model.train(training)
    device = _model_device(model)
    totals = {name: 0.0 for name in ("support", "weight", "distribution", "node", "total")}
    order_generator = torch.Generator().manual_seed(config.seed + epoch * 1_000_003)
    order = torch.randperm(len(graphs), generator=order_generator).tolist() if training else list(range(len(graphs)))
    for position, graph_index in enumerate(order):
        graph = graphs[graph_index].to(device)
        seed = (
            config.seed + epoch * 100_003 + position * 997
            if training
            else config.seed + 50_000_003 + graph_index * 997
        )
        generator = _generator(device, seed)
        view = make_masked_view(
            graph,
            edge_mask_rate=config.edge_mask_rate,
            node_mask_rate=config.node_mask_rate,
            channel_drop_rate=config.channel_drop_rate,
            generator=generator,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses = reconstruction_losses(
                model, graph, view, generator=generator, **_loss_arguments(config)
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                losses = reconstruction_losses(
                    model, graph, view, generator=generator, **_loss_arguments(config)
                )
        for name in totals:
            totals[name] += float(getattr(losses, name).detach().cpu())
        if progress_callback is not None:
            progress_callback(position + 1, len(graphs))
    return {name: value / len(graphs) for name, value in totals.items()}


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_relation_mae(
    model: RelationAwareMaskGAE,
    *,
    train_graphs: Sequence[AttentionGraph],
    validation_graphs: Sequence[AttentionGraph],
    config: TrainingConfig,
    output_dir: str | Path,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> TrainingResult:
    """Train for real epochs; validation never reads hallucination labels."""

    config.validate()
    if not train_graphs or not validation_graphs:
        raise ValueError("train and validation graph sets must be non-empty")
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output = Path(output_dir)
    checkpoint = output / "encoder.pt"
    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = _one_epoch(
            model,
            train_graphs,
            config,
            epoch=epoch,
            optimizer=optimizer,
            progress_callback=(
                None
                if progress_callback is None
                else lambda current, total: progress_callback(
                    f"train_epoch_{epoch}", current, total
                )
            ),
        )
        validation_loss = _one_epoch(
            model,
            validation_graphs,
            config,
            epoch=epoch,
            optimizer=None,
            progress_callback=(
                None
                if progress_callback is None
                else lambda current, total: progress_callback(
                    f"validation_epoch_{epoch}", current, total
                )
            ),
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_loss.items()},
            **{
                f"validation_{name}": value
                for name, value in validation_loss.items()
            },
        }
        history.append(record)
        print(json.dumps({"event": "epoch", **record}, sort_keys=True), flush=True)
        current = validation_loss["total"]
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            _atomic_torch_save(
                checkpoint,
                {
                    "schema": "relation-aware-attention-maskgae-v1",
                    "model_config": {
                        "num_layers": model.num_layers,
                        "num_heads": model.num_heads,
                        "embedding_dim": model.embedding_dim,
                        "message_passing_steps": len(model.message_layers),
                    },
                    "training_config": asdict(config),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "state_dict": best_state,
                },
            )
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        checkpoint_path=checkpoint,
    )


def structural_direction_anchor(graph: AttentionGraph) -> dict[str, float]:
    """Compute a pre-registered cluster orientation, never a model input."""

    if graph.trace_value.numel():
        trace_type = (
            graph.edge_index[0, graph.trace_edge_id] >= graph.response_idx
        )
        rp_mass = float(graph.trace_value[trace_type == RP].sum())
        rr_mass = float(graph.trace_value[trace_type != RP].sum())
        total_mass = max(rp_mass + rr_mass, np.finfo(np.float64).eps)
        prompt_fraction = rp_mass / total_mass
        response_fraction = rr_mass / total_mass
        trace_edge = graph.trace_edge_id
        source = graph.edge_index[0, trace_edge].float()
        target = graph.edge_index[1, trace_edge].float()
        lag = (target - source) / target.clamp_min(1.0)
        normalized_lag = float(
            (lag * graph.trace_value).sum() / graph.trace_value.sum().clamp_min(1e-8)
        )
        group_key = graph.edge_index[1, trace_edge] * graph.num_channels + graph.trace_channel
        _, inverse = torch.unique(group_key, return_inverse=True)
        groups = int(inverse.max()) + 1
        mass = torch.zeros(groups, dtype=graph.trace_value.dtype, device=graph.trace_value.device)
        square = torch.zeros_like(mass)
        mass.index_add_(0, inverse, graph.trace_value)
        square.index_add_(0, inverse, graph.trace_value.square())
        concentration = float((square / mass.square().clamp_min(1e-12)).mean())
    else:
        prompt_fraction = response_fraction = normalized_lag = concentration = 0.0
    response_count = max(int(graph.response_mask.sum()), 1)
    mean_degree = graph.num_edges / response_count
    degree_density = mean_degree / max(graph.num_nodes - 1, 1)
    direction = (
        -prompt_fraction
        + response_fraction
        - degree_density
        - normalized_lag
        + concentration
    )
    return {
        "prompt_mass_fraction": float(prompt_fraction),
        "response_mass_fraction": float(response_fraction),
        "mean_in_degree": float(mean_degree),
        "normalized_lag": float(normalized_lag),
        "retained_concentration": float(concentration),
        "direction_score": float(direction),
    }


def fit_two_component_mixture(
    features: np.ndarray | Sequence[Sequence[float]],
    direction_scores: np.ndarray | Sequence[float],
    *,
    seed: int,
) -> TwoComponentMixture:
    """Fit K=2 with free weights; orient components using structural prior only."""

    values = np.asarray(features, dtype=np.float64)
    direction = np.asarray(direction_scores, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or len(values) < 3 or values.shape[1] < 1:
        raise ValueError("mixture fitting requires at least three feature vectors")
    if direction.shape != (len(values),):
        raise ValueError("direction scores must align with mixture features")
    if not np.isfinite(values).all() or not np.isfinite(direction).all():
        raise ValueError("mixture inputs must be finite")
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    standard_deviation = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(standard_deviation > 1e-8, standard_deviation, 1.0))
    standardized = (values - median) / scale
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as error:
        raise RuntimeError("scikit-learn is required for free two-component scoring") from error
    estimator = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        reg_covar=1e-5,
        n_init=10,
        random_state=int(seed),
    ).fit(standardized)
    responsibilities = estimator.predict_proba(standardized)
    direction_means = [
        float(
            np.dot(responsibilities[:, component], direction)
            / max(
                float(responsibilities[:, component].sum()),
                np.finfo(np.float64).eps,
            )
        )
        for component in range(2)
    ]
    hallucination_component = int(np.argmax(direction_means))
    return TwoComponentMixture(
        feature_median=median.tolist(),
        feature_scale=scale.tolist(),
        component_weights=estimator.weights_.tolist(),
        component_means=estimator.means_.tolist(),
        component_variances=estimator.covariances_.tolist(),
        component_direction_means=direction_means,
        hallucination_component=hallucination_component,
    )


@torch.no_grad()
def _representation(
    model: RelationAwareMaskGAE,
    graph: AttentionGraph,
    *,
    num_views: int,
    seed: int,
    include_reconstruction: bool = True,
    max_support_edges: int | None = 8_192,
    max_weight_traces: int | None = 65_536,
    max_distribution_groups: int | None = 512,
    decoder_chunk_size: int = 16_384,
) -> tuple[np.ndarray, dict[str, float]]:
    if num_views < 1:
        raise ValueError("num_views must be positive")
    model.eval()
    device = _model_device(model)
    graph = graph.to(device)
    embeddings: list[torch.Tensor] = []
    energies = {name: [] for name in ("support", "weight", "distribution", "node", "total")}
    for view_index in range(num_views):
        generator = _generator(device, seed + view_index * 104_729)
        view = make_masked_view(
            graph,
            edge_mask_rate=0.35,
            node_mask_rate=0.35,
            channel_drop_rate=0.0,
            generator=generator,
        )
        hidden, embedding = model(graph, view)
        embeddings.append(embedding)
        if include_reconstruction:
            losses = reconstruction_losses(
                model,
                graph,
                view,
                generator=generator,
                hidden=hidden,
                max_support_edges=max_support_edges,
                max_weight_traces=max_weight_traces,
                max_distribution_groups=max_distribution_groups,
                decoder_chunk_size=decoder_chunk_size,
            )
            for name in energies:
                energies[name].append(float(getattr(losses, name).cpu()))
    embedding_value = torch.stack(embeddings).mean(dim=0).cpu().numpy()
    mean_energy = (
        {name: float(np.mean(values)) for name, values in energies.items()}
        if include_reconstruction
        else {}
    )
    return embedding_value, mean_energy


def score_graphs(
    model: RelationAwareMaskGAE,
    *,
    fit_graphs: Sequence[AttentionGraph],
    score_graphs: Sequence[AttentionGraph],
    num_views: int = 8,
    include_reconstruction: bool = True,
    max_support_edges: int | None = 8_192,
    max_weight_traces: int | None = 65_536,
    max_distribution_groups: int | None = 512,
    decoder_chunk_size: int = 16_384,
    seed: int = 42,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[list[dict[str, object]], TwoComponentMixture]:
    """Learn two behavior patterns without forcing either one to be rare."""

    if len(fit_graphs) < 3 or not score_graphs:
        raise ValueError("scoring requires at least three fit graphs and one score graph")
    fit_embeddings, fit_energies, directions = [], [], []
    for index, graph in enumerate(fit_graphs):
        embedding, energy = _representation(
            model,
            graph,
            num_views=num_views,
            seed=seed + index * 1_000_003,
            include_reconstruction=include_reconstruction,
            max_support_edges=max_support_edges,
            max_weight_traces=max_weight_traces,
            max_distribution_groups=max_distribution_groups,
            decoder_chunk_size=decoder_chunk_size,
        )
        fit_embeddings.append(embedding)
        fit_energies.append(energy)
        directions.append(structural_direction_anchor(graph)["direction_score"])
        if progress_callback is not None:
            progress_callback("mixture_fit", index + 1, len(fit_graphs))
    fit_features = np.asarray(
        [
            (
                np.concatenate(
                    (
                        embedding,
                        np.log1p(
                            [
                                energy["support"],
                                energy["weight"],
                                energy["distribution"],
                                energy["node"],
                            ]
                        ),
                    )
                )
                if include_reconstruction
                else embedding
            )
            for embedding, energy in zip(fit_embeddings, fit_energies)
        ]
    )
    mixture = fit_two_component_mixture(fit_features, directions, seed=seed)

    records: list[dict[str, object]] = []
    for index, graph in enumerate(score_graphs):
        embedding, energy = _representation(
            model,
            graph,
            num_views=num_views,
            seed=seed + (len(fit_graphs) + index) * 1_000_003,
            include_reconstruction=include_reconstruction,
            max_support_edges=max_support_edges,
            max_weight_traces=max_weight_traces,
            max_distribution_groups=max_distribution_groups,
            decoder_chunk_size=decoder_chunk_size,
        )
        feature = (
            np.concatenate(
                (
                    embedding,
                    np.log1p(
                        [
                            energy["support"],
                            energy["weight"],
                            energy["distribution"],
                            energy["node"],
                        ]
                    ),
                )
            )
            if include_reconstruction
            else embedding
        )
        probability = float(mixture.hallucination_probability(feature)[0])
        record: dict[str, object] = {
                "source_id": graph.source_id,
                "sample_id": graph.sample_id,
                "response_id": graph.response_id,
                "hallucination_probability": probability,
                "assigned_component": int(
                    np.argmax(mixture.component_probability(feature)[0])
                ),
                "graph_embedding": embedding.tolist(),
                "structural_orientation": structural_direction_anchor(graph),
            }
        if include_reconstruction:
            record.update(
                {
                "support_energy": energy["support"],
                "attention_energy": energy["weight"],
                "distribution_energy": energy["distribution"],
                "node_energy": energy["node"],
                }
            )
        records.append(record)
        if progress_callback is not None:
            progress_callback("test_scoring", index + 1, len(score_graphs))
    return records, mixture


def _full_view(graph: AttentionGraph) -> MaskedGraphView:
    """Return an uncorrupted view for behavior embeddings at inference."""

    device = graph.edge_index.device
    return MaskedGraphView(
        graph=graph,
        visible_edge_mask=torch.ones(
            graph.num_edges, dtype=torch.bool, device=device
        ),
        masked_edge_ids=torch.empty(0, dtype=torch.long, device=device),
        node_mask=torch.zeros(graph.num_nodes, dtype=torch.bool, device=device),
        channel_keep_mask=torch.ones(
            graph.num_channels, dtype=torch.bool, device=device
        ),
    )


def _targeted_token_view(
    graph: AttentionGraph,
    selected_targets: torch.Tensor,
    *,
    edge_mask_rate: float,
    generator: torch.Generator,
) -> MaskedGraphView:
    """Mask selected response nodes and only their incoming RP/RR groups.

    A singleton relation group remains visible.  Larger groups always retain
    at least one incoming edge, so the corruption cannot manufacture a token
    with no grounding relation at all.
    """

    if not 0.0 <= edge_mask_rate <= 1.0:
        raise ValueError("edge_mask_rate must be in [0, 1]")
    if selected_targets.ndim != 1 or selected_targets.dtype != torch.long:
        raise ValueError("selected_targets must be a one-dimensional long tensor")
    if selected_targets.numel() and (
        bool((selected_targets < graph.response_idx).any())
        or bool((selected_targets >= graph.num_nodes).any())
    ):
        raise ValueError("only response tokens can be selected for token scoring")

    device = graph.edge_index.device
    visible = torch.ones(graph.num_edges, dtype=torch.bool, device=device)
    selected = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    selected[selected_targets] = True
    if graph.num_edges and selected_targets.numel() and edge_mask_rate > 0.0:
        incoming = selected[graph.edge_index[1]]
        causal_relation = (graph.edge_index[0] >= graph.response_idx).long()
        group = graph.edge_index[1] * 2 + causal_relation
        members = torch.nonzero(incoming, as_tuple=False).flatten()
        if members.numel() == 0:
            return MaskedGraphView(
                graph=graph,
                visible_edge_mask=visible,
                masked_edge_ids=torch.empty(
                    0, dtype=torch.long, device=device
                ),
                node_mask=selected,
                channel_keep_mask=torch.ones(
                    graph.num_channels, dtype=torch.bool, device=device
                ),
            )
        random_order = torch.randperm(
            members.numel(),
            generator=generator,
            device=torch.device(generator.device),
        ).to(device)
        order = random_order[
            torch.argsort(group[members[random_order]], stable=True)
        ]
        ordered_group = group[members[order]]
        _groups, counts = torch.unique_consecutive(
            ordered_group, return_counts=True
        )
        repeated_count = torch.repeat_interleave(counts, counts)
        position = torch.arange(members.numel(), device=device)
        starts = torch.where(
            torch.cat(
                (
                    torch.ones(1, dtype=torch.bool, device=device),
                    ordered_group[1:] != ordered_group[:-1],
                )
            ),
            position,
            torch.zeros_like(position),
        )
        rank = position - torch.cummax(starts, dim=0).values
        count = torch.round(
            repeated_count.to(torch.float32) * edge_mask_rate
        ).long()
        count = torch.maximum(count, torch.ones_like(count))
        count = torch.minimum(count, repeated_count - 1)
        visible[members[order[rank < count]]] = False
    return MaskedGraphView(
        graph=graph,
        visible_edge_mask=visible,
        masked_edge_ids=torch.nonzero(~visible, as_tuple=False).flatten(),
        node_mask=selected,
        channel_keep_mask=torch.ones(
            graph.num_channels, dtype=torch.bool, device=device
        ),
    )


def _token_direction_anchors(
    graph: AttentionGraph,
) -> dict[int, dict[str, float]]:
    """Vectorize the pre-registered orientation prior for all response nodes."""

    node_count = graph.num_nodes
    device = graph.edge_index.device
    dtype = graph.trace_value.dtype
    rp_mass = torch.zeros(node_count, dtype=dtype, device=device)
    rr_mass = torch.zeros_like(rp_mass)
    trace_mass = torch.zeros_like(rp_mass)
    lag_mass = torch.zeros_like(rp_mass)
    concentration_sum = torch.zeros_like(rp_mass)
    concentration_count = torch.zeros_like(rp_mass)

    if graph.trace_value.numel():
        trace_edge = graph.trace_edge_id
        trace_target = graph.edge_index[1, trace_edge]
        trace_source = graph.edge_index[0, trace_edge]
        trace_is_rr = trace_source >= graph.response_idx
        rp_mass.index_add_(
            0, trace_target[~trace_is_rr], graph.trace_value[~trace_is_rr]
        )
        rr_mass.index_add_(
            0, trace_target[trace_is_rr], graph.trace_value[trace_is_rr]
        )
        trace_mass.index_add_(0, trace_target, graph.trace_value)
        lag = (trace_target.float() - trace_source.float()) / trace_target.float().clamp_min(1.0)
        lag_mass.index_add_(0, trace_target, lag * graph.trace_value)

        group_key = trace_target * graph.num_channels + graph.trace_channel
        unique_group, inverse = torch.unique(
            group_key, sorted=True, return_inverse=True
        )
        group_mass = torch.zeros(
            unique_group.numel(), dtype=dtype, device=device
        )
        group_square = torch.zeros_like(group_mass)
        group_mass.index_add_(0, inverse, graph.trace_value)
        group_square.index_add_(0, inverse, graph.trace_value.square())
        group_concentration = group_square / group_mass.square().clamp_min(1e-12)
        group_target = unique_group // graph.num_channels
        concentration_sum.index_add_(0, group_target, group_concentration)
        concentration_count.index_add_(
            0, group_target, torch.ones_like(group_concentration)
        )

    total_mass = (rp_mass + rr_mass).clamp_min(np.finfo(np.float64).eps)
    prompt_fraction = rp_mass / total_mass
    response_fraction = rr_mass / total_mass
    normalized_lag = lag_mass / trace_mass.clamp_min(1e-8)
    concentration = concentration_sum / concentration_count.clamp_min(1.0)
    in_degree = torch.bincount(
        graph.edge_index[1], minlength=node_count
    ).to(dtype=dtype)
    target_index = torch.arange(node_count, device=device, dtype=dtype)
    degree_density = in_degree / target_index.clamp_min(1.0)
    direction = (
        -prompt_fraction
        + response_fraction
        - degree_density
        - normalized_lag
        + concentration
    )
    response = torch.arange(graph.response_idx, node_count, device=device)
    values = torch.stack(
        (
            prompt_fraction[response],
            response_fraction[response],
            in_degree[response],
            normalized_lag[response],
            concentration[response],
            direction[response],
        ),
        dim=1,
    ).detach().cpu().tolist()
    return {
        target: {
            "prompt_mass_fraction": float(row[0]),
            "response_mass_fraction": float(row[1]),
            "in_degree": float(row[2]),
            "normalized_lag": float(row[3]),
            "retained_concentration": float(row[4]),
            "direction_score": float(row[5]),
        }
        for target, row in zip(range(graph.response_idx, node_count), values)
    }


def _token_direction_anchor(graph: AttentionGraph, target: int) -> dict[str, float]:
    """Compatibility wrapper for one response token's vectorized anchor."""

    if not graph.response_idx <= target < graph.num_nodes:
        raise ValueError("token orientation is defined only for response nodes")
    return _token_direction_anchors(graph)[target]


@torch.no_grad()
def _token_features(
    model: RelationAwareMaskGAE,
    graph: AttentionGraph,
    *,
    mask_stride: int,
    edge_mask_rate: float,
    seed: int,
    include_reconstruction: bool = True,
    max_support_edges: int | None = 8_192,
    max_weight_traces: int | None = 65_536,
    max_distribution_groups: int | None = 512,
    decoder_chunk_size: int = 16_384,
) -> tuple[np.ndarray, list[int], list[dict[str, float]], list[dict[str, float]]]:
    """Encode every response token and obtain targeted reconstruction energy."""

    if mask_stride < 1:
        raise ValueError("mask_stride must be positive")
    model.eval()
    device = _model_device(model)
    device_graph = graph.to(device)
    hidden = model.encode(device_graph, _full_view(device_graph))
    response_nodes = torch.nonzero(
        device_graph.response_mask, as_tuple=False
    ).flatten()
    token_indices = [int(value) for value in response_nodes.tolist()]
    anchor_by_target = _token_direction_anchors(graph)
    anchors = [anchor_by_target[target] for target in token_indices]
    if not include_reconstruction:
        return (
            hidden[response_nodes].cpu().numpy(),
            token_indices,
            anchors,
            [{} for _ in token_indices],
        )
    energy_names = (
        "support_rp",
        "support_rr",
        "weight_rp",
        "weight_rr",
        "distribution",
        "node",
    )
    accumulated = {
        name: torch.zeros(
            device_graph.num_nodes, dtype=hidden.dtype, device=device
        )
        for name in energy_names
    }
    seen = torch.zeros(device_graph.num_nodes, dtype=torch.bool, device=device)
    phases = min(mask_stride, max(int(response_nodes.numel()), 1))
    for phase in range(phases):
        selected = response_nodes[
            torch.arange(response_nodes.numel(), device=device) % phases == phase
        ]
        if selected.numel() == 0:
            continue
        generator = _generator(device, seed + phase * 104_729)
        view = _targeted_token_view(
            device_graph,
            selected,
            edge_mask_rate=edge_mask_rate,
            generator=generator,
        )
        energy = reconstruction_energy_by_node(
            model,
            device_graph,
            view,
            generator=generator,
            max_support_edges=max_support_edges,
            max_weight_traces=max_weight_traces,
            max_distribution_groups=max_distribution_groups,
            decoder_chunk_size=decoder_chunk_size,
        )
        for name in energy_names:
            accumulated[name][selected] = energy[name][selected]
        seen[selected] = True
    if not bool(seen[response_nodes].all()):
        raise RuntimeError("targeted masking did not score every response token")

    feature_rows: list[np.ndarray] = []
    energy_rows: list[dict[str, float]] = []
    token_indices = [int(value) for value in response_nodes.tolist()]
    for target in token_indices:
        energy = {
            name: float(accumulated[name][target].cpu()) for name in energy_names
        }
        feature_rows.append(
            np.concatenate(
                (
                    hidden[target].cpu().numpy(),
                    np.log1p(np.asarray(list(energy.values()), dtype=np.float64)),
                )
            )
        )
        energy_rows.append(energy)
    return np.asarray(feature_rows), token_indices, anchors, energy_rows


def score_tokens(
    model: RelationAwareMaskGAE,
    *,
    fit_graphs: Sequence[AttentionGraph],
    score_graphs: Sequence[AttentionGraph],
    mask_stride: int = 8,
    edge_mask_rate: float = 0.5,
    max_fit_tokens: int | None = 100_000,
    include_reconstruction: bool = True,
    max_support_edges: int | None = 8_192,
    max_weight_traces: int | None = 65_536,
    max_distribution_groups: int | None = 512,
    decoder_chunk_size: int = 16_384,
    seed: int = 42,
) -> tuple[list[dict[str, object]], TwoComponentMixture]:
    """Fit and score two token behavior modes without a rare-token premise."""

    if not fit_graphs or not score_graphs:
        raise ValueError("token scoring requires non-empty fit and score graphs")
    if max_fit_tokens is not None and max_fit_tokens < 3:
        raise ValueError("max_fit_tokens must be at least three when provided")

    fit_feature_parts: list[np.ndarray] = []
    fit_directions: list[float] = []
    for index, graph in enumerate(fit_graphs):
        features, _tokens, anchors, _energies = _token_features(
            model,
            graph,
            mask_stride=mask_stride,
            edge_mask_rate=edge_mask_rate,
            seed=seed + index * 1_000_003,
            include_reconstruction=include_reconstruction,
            max_support_edges=max_support_edges,
            max_weight_traces=max_weight_traces,
            max_distribution_groups=max_distribution_groups,
            decoder_chunk_size=decoder_chunk_size,
        )
        fit_feature_parts.append(features)
        fit_directions.extend(anchor["direction_score"] for anchor in anchors)
    fit_features = np.concatenate(fit_feature_parts, axis=0)
    fit_direction = np.asarray(fit_directions, dtype=np.float64)
    if len(fit_features) < 3:
        raise ValueError("token mixture requires at least three fit tokens")
    if max_fit_tokens is not None and len(fit_features) > max_fit_tokens:
        random = np.random.default_rng(seed)
        chosen = np.sort(
            random.choice(len(fit_features), size=max_fit_tokens, replace=False)
        )
        fit_features = fit_features[chosen]
        fit_direction = fit_direction[chosen]
    mixture = fit_two_component_mixture(
        fit_features, fit_direction, seed=seed
    )

    records: list[dict[str, object]] = []
    offset = len(fit_graphs)
    for graph_index, graph in enumerate(score_graphs):
        features, tokens, anchors, energies = _token_features(
            model,
            graph,
            mask_stride=mask_stride,
            edge_mask_rate=edge_mask_rate,
            seed=seed + (offset + graph_index) * 1_000_003,
            include_reconstruction=include_reconstruction,
            max_support_edges=max_support_edges,
            max_weight_traces=max_weight_traces,
            max_distribution_groups=max_distribution_groups,
            decoder_chunk_size=decoder_chunk_size,
        )
        component_probability = mixture.component_probability(features)
        hallucination_probability = component_probability[
            :, mixture.hallucination_component
        ]
        for local_index, (target, anchor, energy) in enumerate(
            zip(tokens, anchors, energies)
        ):
            probability = float(hallucination_probability[local_index])
            record: dict[str, object] = {
                    "source_id": graph.source_id,
                    "sample_id": graph.sample_id,
                    "response_id": graph.response_id,
                    "token_idx": target,
                    "response_token_idx": target - graph.response_idx,
                    "token_id": int(graph.token_ids[target]),
                    "score": probability,
                    "hallucination_probability": probability,
                    "assigned_component": int(
                        np.argmax(component_probability[local_index])
                    ),
                    "structural_orientation": anchor,
                }
            if include_reconstruction:
                record.update(
                    {
                        "rp_support_energy": energy["support_rp"],
                        "rr_support_energy": energy["support_rr"],
                        "rp_attention_energy": energy["weight_rp"],
                        "rr_attention_energy": energy["weight_rr"],
                        "distribution_energy": energy["distribution"],
                        "node_energy": energy["node"],
                    }
                )
            records.append(record)
    return records, mixture


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "TwoComponentMixture",
    "fit_two_component_mixture",
    "relation_preserving_source_shuffle",
    "score_graphs",
    "score_tokens",
    "structural_direction_anchor",
    "train_relation_mae",
]
