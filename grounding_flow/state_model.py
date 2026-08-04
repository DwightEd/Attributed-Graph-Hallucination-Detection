"""A small label-free two-state sequence model for grounding transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    result = maximum + np.log(np.sum(shifted, axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.squeeze()


def _as_sequences(
    sequences: Sequence[np.ndarray | Sequence[Sequence[float]]],
) -> list[np.ndarray]:
    converted = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    if not converted or any(sequence.ndim != 2 or len(sequence) < 1 for sequence in converted):
        raise ValueError("training sequences must be non-empty two-dimensional arrays")
    dimensions = {sequence.shape[1] for sequence in converted}
    if len(dimensions) != 1 or next(iter(dimensions)) < 1:
        raise ValueError("training sequences must share a positive observation dimension")
    if sum(len(sequence) for sequence in converted) < 4:
        raise ValueError("two states require at least four observations across sequences")
    if not all(np.isfinite(sequence).all() for sequence in converted):
        raise ValueError("training observations must be finite")
    return converted


def _as_anchors(
    anchors: Sequence[np.ndarray | Sequence[float]],
    sequences: Sequence[np.ndarray],
) -> list[np.ndarray]:
    converted = [np.asarray(anchor, dtype=np.float64).reshape(-1) for anchor in anchors]
    if len(converted) != len(sequences) or any(
        len(anchor) != len(sequence) for anchor, sequence in zip(converted, sequences)
    ):
        raise ValueError("one finite mechanism anchor is required per token observation")
    if not all(np.isfinite(anchor).all() for anchor in converted):
        raise ValueError("mechanism anchors must be finite")
    return converted


@dataclass(frozen=True)
class TrajectoryProjector:
    """Training-only standardisation and PCA over full layer/head surfaces."""

    input_shape: tuple[int, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def __post_init__(self) -> None:
        dimension = int(np.prod(self.input_shape)) if self.input_shape else 0
        arrays = (
            self.feature_mean,
            self.feature_scale,
            self.pca_mean,
            self.components,
            self.explained_variance_ratio,
        )
        if dimension < 1 or not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("projector parameters must be finite with positive shape")
        if (
            self.feature_mean.shape != (dimension,)
            or self.feature_scale.shape != (dimension,)
            or self.pca_mean.shape != (dimension,)
            or self.components.ndim != 2
            or self.components.shape[1] != dimension
            or self.explained_variance_ratio.shape != (self.components.shape[0],)
            or self.components.shape[0] < 1
        ):
            raise ValueError("projector parameter shapes are inconsistent")
        if np.any(self.feature_scale <= 0.0):
            raise ValueError("projector feature scale must be positive")

    @property
    def output_dimension(self) -> int:
        return int(self.components.shape[0])

    def transform(self, surface: np.ndarray | Sequence[object]) -> np.ndarray:
        value = np.asarray(surface, dtype=np.float64)
        if value.ndim < 2 or tuple(value.shape[1:]) != self.input_shape:
            raise ValueError(
                f"trajectory surface must have shape [tokens, {self.input_shape}]"
            )
        if not len(value) or not np.isfinite(value).all():
            raise ValueError("trajectory surface must contain finite token observations")
        flattened = value.reshape(len(value), -1)
        standardised = (flattened - self.feature_mean) / self.feature_scale
        return (standardised - self.pca_mean) @ self.components.T

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "grounding-flow-training-pca-v1",
            "input_shape": list(self.input_shape),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "pca_mean": self.pca_mean.tolist(),
            "components": self.components.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrajectoryProjector":
        if value.get("schema") != "grounding-flow-training-pca-v1":
            raise ValueError("unsupported grounding-flow projector schema")
        return cls(
            input_shape=tuple(int(item) for item in value["input_shape"]),
            feature_mean=np.asarray(value["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(value["feature_scale"], dtype=np.float64),
            pca_mean=np.asarray(value["pca_mean"], dtype=np.float64),
            components=np.asarray(value["components"], dtype=np.float64),
            explained_variance_ratio=np.asarray(
                value["explained_variance_ratio"], dtype=np.float64
            ),
        )


def fit_trajectory_projector(
    surfaces: Sequence[np.ndarray | Sequence[object]],
    *,
    max_components: int = 32,
    max_fit_tokens: int = 20_000,
    seed: int = 0,
) -> TrajectoryProjector:
    """Fit an unsupervised projection without averaging heads or layers."""

    values = [np.asarray(surface, dtype=np.float64) for surface in surfaces]
    if not values or any(value.ndim < 2 or len(value) < 1 for value in values):
        raise ValueError("projector requires non-empty token trajectory surfaces")
    input_shapes = {tuple(value.shape[1:]) for value in values}
    if len(input_shapes) != 1 or not all(np.isfinite(value).all() for value in values):
        raise ValueError("trajectory surfaces must be finite and share layer/head shape")
    if max_components < 1 or max_fit_tokens < 2:
        raise ValueError("projector component and token limits must be positive")
    input_shape = next(iter(input_shapes))
    flattened_values = [value.reshape(len(value), -1) for value in values]
    generator = np.random.default_rng(seed)
    if len(flattened_values) > max_fit_tokens:
        selected_responses = np.sort(
            generator.choice(
                len(flattened_values), size=max_fit_tokens, replace=False
            )
        )
        flattened_values = [flattened_values[index] for index in selected_responses]
    per_response = min(
        max(len(value) for value in flattened_values),
        max(1, max_fit_tokens // len(flattened_values)),
    )
    balanced: list[np.ndarray] = []
    for value in flattened_values:
        indices = generator.choice(
            len(value),
            size=per_response,
            replace=len(value) < per_response,
        )
        balanced.append(value[np.sort(indices)])
    fit_values = (
        balanced[0] if len(balanced) == 1 else np.concatenate(balanced, axis=0)
    )
    feature_mean = fit_values.mean(axis=0)
    feature_scale = fit_values.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardised = (fit_values - feature_mean) / feature_scale
    if float(np.sum(standardised * standardised)) <= 1e-12:
        raise ValueError("trajectory projector requires non-zero training variation")
    component_count = min(
        max_components, standardised.shape[0] - 1, standardised.shape[1]
    )
    if component_count < 1:
        raise ValueError("projector needs at least two fit tokens")
    solver = (
        "randomized"
        if component_count < min(standardised.shape) and min(standardised.shape) > 128
        else "full"
    )
    pca = PCA(n_components=component_count, svd_solver=solver, random_state=seed)
    pca.fit(standardised)
    if not (
        np.isfinite(pca.mean_).all()
        and np.isfinite(pca.components_).all()
        and np.isfinite(pca.explained_variance_ratio_).all()
    ):
        raise ValueError("trajectory PCA produced non-finite parameters")
    variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    effective_components = int(np.sum(variance_ratio > 1e-12))
    if effective_components < 1:
        raise ValueError("trajectory PCA found no effective variation")
    return TrajectoryProjector(
        input_shape=input_shape,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        components=np.asarray(pca.components_[:effective_components], dtype=np.float64),
        explained_variance_ratio=variance_ratio[:effective_components],
    )


@dataclass(frozen=True)
class TwoStateGaussianHMM:
    """Diagonal-Gaussian HMM with a post-fit structural state orientation."""

    initial_probability: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    detached_state: int
    state_anchor: np.ndarray
    state_occupancy: np.ndarray
    log_likelihood_history: tuple[float, ...]
    fit_weighting: str = "response_balanced"

    def __post_init__(self) -> None:
        if (
            self.initial_probability.shape != (2,)
            or self.transition.shape != (2, 2)
            or self.means.ndim != 2
            or self.means.shape[0] != 2
            or self.means.shape[1] < 1
            or self.variances.shape != self.means.shape
            or self.state_anchor.shape != (2,)
            or self.state_occupancy.shape != (2,)
        ):
            raise ValueError("HMM parameter shapes are inconsistent")
        arrays = (
            self.initial_probability,
            self.transition,
            self.means,
            self.variances,
            self.state_anchor,
            self.state_occupancy,
            np.asarray(self.log_likelihood_history),
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("HMM parameters must be finite")
        if (
            np.any(self.initial_probability < 0.0)
            or not np.isclose(self.initial_probability.sum(), 1.0, atol=1e-8)
        ):
            raise ValueError("HMM initial probabilities must sum to one")
        if np.any(self.transition < 0.0) or not np.allclose(
            self.transition.sum(axis=1), 1.0, atol=1e-8
        ):
            raise ValueError("HMM transition rows must sum to one")
        if np.any(self.variances <= 0.0):
            raise ValueError("HMM variances must be positive")
        if np.any(self.state_occupancy < 0.0) or not np.isclose(
            self.state_occupancy.sum(), 1.0, atol=1e-8
        ):
            raise ValueError("HMM state occupancy must sum to one")
        if self.detached_state not in (0, 1):
            raise ValueError("HMM detached_state must be 0 or 1")
        if self.fit_weighting != "response_balanced":
            raise ValueError("HMM fit weighting must be response_balanced")

    @property
    def orientation_margin(self) -> float:
        return float(abs(self.state_anchor[1] - self.state_anchor[0]))

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def _forward_backward(
        self, sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        observations = np.asarray(sequence, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != self.dimension:
            raise ValueError("score sequence has the wrong observation dimension")
        if not len(observations) or not np.isfinite(observations).all():
            raise ValueError("score sequence must contain finite observations")
        return _forward_backward_parameters(
            observations,
            self.initial_probability,
            self.transition,
            self.means,
            self.variances,
        )

    def posterior(self, sequence: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        gamma, _, _ = self._forward_backward(np.asarray(sequence, dtype=np.float64))
        return gamma

    def log_likelihood(self, sequence: np.ndarray | Sequence[Sequence[float]]) -> float:
        _, _, value = self._forward_backward(np.asarray(sequence, dtype=np.float64))
        return value

    def score(self, sequence: np.ndarray | Sequence[Sequence[float]]) -> dict[str, object]:
        probability = self.posterior(sequence)[:, self.detached_state]
        top_count = max(1, int(math.ceil(0.1 * len(probability))))
        order = np.sort(probability)
        above = np.flatnonzero(probability >= 0.5)
        return {
            "token_probability": probability,
            "mean": float(probability.mean()),
            "max": float(probability.max()),
            "top_10_percent_mean": float(order[-top_count:].mean()),
            "first_detached_offset": int(above[0]) if len(above) else None,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "grounding-flow-two-state-diagonal-gaussian-hmm-v1",
            "initial_probability": self.initial_probability.tolist(),
            "transition": self.transition.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
            "detached_state": int(self.detached_state),
            "state_anchor": self.state_anchor.tolist(),
            "state_occupancy": self.state_occupancy.tolist(),
            "orientation_margin": self.orientation_margin,
            "log_likelihood_history": list(self.log_likelihood_history),
            "fit_weighting": self.fit_weighting,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TwoStateGaussianHMM":
        if value.get("schema") != "grounding-flow-two-state-diagonal-gaussian-hmm-v1":
            raise ValueError("unsupported grounding-flow state-model schema")
        return cls(
            initial_probability=np.asarray(value["initial_probability"], dtype=np.float64),
            transition=np.asarray(value["transition"], dtype=np.float64),
            means=np.asarray(value["means"], dtype=np.float64),
            variances=np.asarray(value["variances"], dtype=np.float64),
            detached_state=int(value["detached_state"]),
            state_anchor=np.asarray(value["state_anchor"], dtype=np.float64),
            state_occupancy=np.asarray(value["state_occupancy"], dtype=np.float64),
            log_likelihood_history=tuple(
                float(item) for item in value["log_likelihood_history"]
            ),
            fit_weighting=str(value.get("fit_weighting", "response_balanced")),
        )


def _forward_backward_parameters(
    sequence: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    difference = sequence[:, None, :] - means[None, :, :]
    emission = -0.5 * np.sum(
        np.log(2.0 * np.pi * variances)[None, :, :]
        + difference * difference / variances[None, :, :],
        axis=2,
    )
    log_initial = np.log(np.clip(initial, 1e-300, None))
    log_transition = np.log(np.clip(transition, 1e-300, None))
    length = emission.shape[0]
    forward = np.empty((length, 2), dtype=np.float64)
    forward[0] = log_initial + emission[0]
    for index in range(1, length):
        forward[index] = emission[index] + _logsumexp(
            forward[index - 1][:, None] + log_transition, axis=0
        )
    log_likelihood = float(_logsumexp(forward[-1], axis=0))
    backward = np.zeros((length, 2), dtype=np.float64)
    for index in range(length - 2, -1, -1):
        backward[index] = _logsumexp(
            log_transition
            + emission[index + 1][None, :]
            + backward[index + 1][None, :],
            axis=1,
        )
    log_gamma = forward + backward - log_likelihood
    gamma = np.exp(log_gamma - _logsumexp(log_gamma, axis=1)[:, None])
    xi_sum = np.zeros((2, 2), dtype=np.float64)
    for index in range(length - 1):
        log_xi = (
            forward[index][:, None]
            + log_transition
            + emission[index + 1][None, :]
            + backward[index + 1][None, :]
            - log_likelihood
        )
        xi_sum += np.exp(log_xi - _logsumexp(log_xi, axis=None))
    return gamma, xi_sum, log_likelihood


def fit_two_state_hmm(
    sequences: Sequence[np.ndarray | Sequence[Sequence[float]]],
    mechanism_anchors: Sequence[np.ndarray | Sequence[float]],
    *,
    seed: int = 0,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    variance_floor: float = 1e-4,
    pseudocount: float = 1e-3,
    progress_callback: Callable[[int, float], None] | None = None,
) -> TwoStateGaussianHMM:
    """Fit two freely weighted routing states without hallucination labels.

    The mechanism anchor is consulted only after EM to name one of the two
    otherwise exchangeable states.  It does not enter the likelihood,
    posterior, transition estimates, or Gaussian parameters.
    """

    observations = _as_sequences(sequences)
    anchors = _as_anchors(mechanism_anchors, observations)
    if max_iterations < 1 or tolerance < 0.0:
        raise ValueError("max_iterations must be positive and tolerance non-negative")
    if variance_floor <= 0.0 or pseudocount <= 0.0:
        raise ValueError("variance_floor and pseudocount must be positive")
    concatenated = np.concatenate(observations, axis=0)
    observation_weights = np.concatenate(
        [np.full(len(sequence), 1.0 / len(sequence)) for sequence in observations]
    )
    if not bool(np.any(np.any(concatenated != concatenated[0], axis=1))):
        raise ValueError("two states require at least two distinct observations")
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
    kmeans.fit(concatenated, sample_weight=observation_weights)
    labels = np.asarray(kmeans.labels_, dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("two-state initialization did not produce two non-empty states")
    dimension = concatenated.shape[1]
    global_mean = np.average(concatenated, axis=0, weights=observation_weights)
    global_variance = (
        np.average(
            (concatenated - global_mean) ** 2,
            axis=0,
            weights=observation_weights,
        )
        + variance_floor
    )
    means_list: list[np.ndarray] = []
    variances_list: list[np.ndarray] = []
    for state in range(2):
        mask = labels == state
        state_weights = observation_weights[mask]
        state_mean = np.average(
            concatenated[mask], axis=0, weights=state_weights
        )
        means_list.append(state_mean)
        variances_list.append(
            np.average(
                (concatenated[mask] - state_mean) ** 2,
                axis=0,
                weights=state_weights,
            )
            + variance_floor
        )
    means = np.stack(means_list)
    variances = np.stack(variances_list).reshape(2, dimension)
    initial = np.full(2, 0.5, dtype=np.float64)
    transition = np.full((2, 2), pseudocount, dtype=np.float64)
    offset = 0
    initial_counts = np.full(2, pseudocount, dtype=np.float64)
    for sequence in observations:
        local = labels[offset : offset + len(sequence)]
        sequence_weight = 1.0 / len(local)
        initial_counts[local[0]] += sequence_weight
        for left, right in zip(local[:-1], local[1:]):
            transition[left, right] += sequence_weight
        offset += len(sequence)
    initial = initial_counts / initial_counts.sum()
    transition /= transition.sum(axis=1, keepdims=True)

    history: list[float] = []
    for _ in range(max_iterations):
        initial_counts = np.zeros(2, dtype=np.float64)
        transition_counts = np.zeros((2, 2), dtype=np.float64)
        state_mass = np.zeros(2, dtype=np.float64)
        weighted_sum = np.zeros((2, dimension), dtype=np.float64)
        weighted_square = np.zeros((2, dimension), dtype=np.float64)
        log_likelihood = 0.0
        for sequence in observations:
            gamma, xi_sum, sequence_likelihood = _forward_backward_parameters(
                sequence, initial, transition, means, variances
            )
            emission_weight = 1.0 / len(sequence)
            initial_counts += gamma[0] * emission_weight
            transition_counts += xi_sum * emission_weight
            state_mass += gamma.sum(axis=0) * emission_weight
            weighted_sum += (gamma.T @ sequence) * emission_weight
            weighted_square += (
                gamma.T @ (sequence * sequence)
            ) * emission_weight
            log_likelihood += sequence_likelihood * emission_weight
        history.append(float(log_likelihood))
        if not math.isfinite(log_likelihood):
            raise ValueError("HMM training produced a non-finite log likelihood")
        if progress_callback is not None:
            progress_callback(len(history), float(log_likelihood))
        initial = initial_counts / initial_counts.sum()
        transition_mass = transition_counts.sum(axis=1, keepdims=True)
        transition = np.divide(
            transition_counts,
            transition_mass,
            out=transition.copy(),
            where=transition_mass > 0.0,
        )
        means = weighted_sum / np.maximum(state_mass[:, None], 1e-12)
        variances = (
            weighted_square / np.maximum(state_mass[:, None], 1e-12) - means * means
        )
        variances = np.maximum(variances, variance_floor)
        if not (
            np.isfinite(initial).all()
            and np.isfinite(transition).all()
            and np.isfinite(means).all()
            and np.isfinite(variances).all()
        ):
            raise ValueError("HMM training produced non-finite parameters")
        if len(history) >= 2:
            improvement = history[-1] - history[-2]
            scale = max(1.0, abs(history[-2]))
            if improvement >= 0.0 and improvement <= tolerance * scale:
                break

    state_mass = np.zeros(2, dtype=np.float64)
    anchor_sum = np.zeros(2, dtype=np.float64)
    for sequence, anchor in zip(observations, anchors):
        gamma, _, _ = _forward_backward_parameters(
            sequence, initial, transition, means, variances
        )
        response_weight = 1.0 / len(sequence)
        state_mass += gamma.sum(axis=0) * response_weight
        anchor_sum += (gamma.T @ anchor) * response_weight
    state_anchor = anchor_sum / np.maximum(state_mass, 1e-12)
    occupancy = state_mass / state_mass.sum()
    if not np.isfinite(state_anchor).all() or not np.isfinite(occupancy).all():
        raise ValueError("HMM state orientation produced non-finite parameters")
    if float(abs(state_anchor[1] - state_anchor[0])) <= 1e-8:
        raise ValueError("HMM structural orientation is unresolved")
    detached_state = int(np.argmax(state_anchor))
    return TwoStateGaussianHMM(
        initial_probability=initial,
        transition=transition,
        means=means,
        variances=variances,
        detached_state=detached_state,
        state_anchor=state_anchor,
        state_occupancy=occupancy,
        log_likelihood_history=tuple(history),
    )


__all__ = [
    "TrajectoryProjector",
    "TwoStateGaussianHMM",
    "fit_trajectory_projector",
    "fit_two_state_hmm",
]
