"""Label-blind anomaly scorers."""

from __future__ import annotations

import numpy as np


class RobustMahalanobisScorer:
    """Median/MAD standardized Mahalanobis distance with covariance shrinkage."""

    def __init__(self, *, shrinkage: float = 0.1, ridge: float = 1e-6):
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between zero and one")
        self.shrinkage = float(shrinkage)
        self.ridge = float(ridge)

    @staticmethod
    def _matrix(features) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or not matrix.shape[1]:
            raise ValueError("features must be a non-empty two-dimensional matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("features must contain only finite values")
        return matrix

    def fit(self, features):
        matrix = self._matrix(features)
        if len(matrix) < 2:
            raise ValueError("at least two reference samples are required")
        self.center_ = np.median(matrix, axis=0)
        mad = 1.4826 * np.median(np.abs(matrix - self.center_), axis=0)
        fallback = np.std(matrix, axis=0)
        self.scale_ = np.where(mad > 1e-12, mad, np.where(fallback > 1e-12, fallback, 1.0))
        standardized = (matrix - self.center_) / self.scale_
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            (1.0 - self.shrinkage) * covariance
            + self.shrinkage * diagonal
            + self.ridge * np.eye(matrix.shape[1])
        )
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score_samples(self, features) -> np.ndarray:
        if not hasattr(self, "precision_"):
            raise RuntimeError("fit must be called before score_samples")
        matrix = self._matrix(features)
        if matrix.shape[1] != self.center_.shape[0]:
            raise ValueError("feature dimension differs from the fitted reference")
        centered = (matrix - self.center_) / self.scale_
        squared_distance = np.einsum(
            "ni,ij,nj->n", centered, self.precision_, centered
        )
        return np.maximum(squared_distance, 0.0)
