from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from ..metrics import LogEuclideanTransformer


@dataclass(frozen=True)
class KSelectionResult:
    best_k: int
    scores: dict[int, float]


def _majority_vote(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def tune_k_leave_one_out(
    X_log: np.ndarray,
    y: np.ndarray,
    k_values: Sequence[int] = (1, 3, 5),
    n_jobs: int = -1,
) -> KSelectionResult:
    """Strict leave-one-out k selection for k-NN.

    The nearest-neighbor index is built once. Each sample excludes itself before
    voting, so the reported scores are true LOO scores for the candidate set.
    """

    y = np.asarray(y).ravel()
    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X_log)
    neighbor_indices = nn.kneighbors(X_log, return_distance=False)

    without_self = np.empty((X_log.shape[0], max_k), dtype=np.int64)
    for i, row in enumerate(neighbor_indices):
        filtered = row[row != i]
        if filtered.shape[0] < max_k:
            filtered = np.pad(filtered, (0, max_k - filtered.shape[0]), mode="edge")
        without_self[i] = filtered[:max_k]

    scores: dict[int, float] = {}
    for k in k_values:
        preds = np.fromiter(
            (_majority_vote(y[row[:k]]) for row in without_self),
            dtype=y.dtype,
            count=without_self.shape[0],
        )
        scores[int(k)] = float(np.mean(preds == y))

    best_k = max(scores.items(), key=lambda item: (item[1], -item[0]))[0]
    return KSelectionResult(best_k=best_k, scores=scores)


class LogEuclideanKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        k_values: Sequence[int] = (1, 3, 5),
        weights: str = "uniform",
        n_jobs: int = -1,
        tune_k: bool = True,
    ):
        self.k_values = tuple(k_values)
        self.weights = weights
        self.n_jobs = n_jobs
        self.tune_k = tune_k

    def fit(self, X, y):
        self.transformer_ = LogEuclideanTransformer().fit(X)
        X_log = self.transformer_.transform(X)
        y_array = np.asarray(y).ravel()
        if self.tune_k:
            self.k_selection_ = tune_k_leave_one_out(X_log, y_array, self.k_values, self.n_jobs)
            self.best_k_ = self.k_selection_.best_k
        else:
            self.best_k_ = int(self.k_values[0])
            self.k_selection_ = KSelectionResult(best_k=self.best_k_, scores={self.best_k_: np.nan})
        self.classes_ = np.unique(y_array)
        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.best_k_,
            metric="euclidean",
            weights=self.weights,
            n_jobs=self.n_jobs,
        )
        self.knn_.fit(X_log, y_array)
        return self

    def predict(self, X):
        check_is_fitted(self, "knn_")
        return self.knn_.predict(self.transformer_.transform(X))

    def predict_proba(self, X):
        check_is_fitted(self, "knn_")
        return self.knn_.predict_proba(self.transformer_.transform(X))
