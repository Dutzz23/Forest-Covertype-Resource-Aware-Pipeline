from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..metrics import LogEuclideanTransformer
from .knn_logeuclidean import KSelectionResult, tune_k_leave_one_out


class DistanceKNNClassifier(BaseEstimator, ClassifierMixin):
    """k-NN with strict leave-one-out k selection and selectable transforms."""

    def __init__(
        self,
        distance_transform: str = "log_euclidean",
        k_values: Sequence[int] = (1, 3, 5),
        weights: str = "uniform",
        n_jobs: int = -1,
        tune_k: bool = True,
    ):
        self.distance_transform = distance_transform
        self.k_values = tuple(k_values)
        self.weights = weights
        self.n_jobs = n_jobs
        self.tune_k = tune_k

    def fit(self, X, y):
        y_array = np.asarray(y).ravel()
        X_metric = self._fit_transform(X)

        if self.tune_k:
            self.k_selection_ = tune_k_leave_one_out(X_metric, y_array, self.k_values, self.n_jobs)
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
        self.knn_.fit(X_metric, y_array)
        return self

    def predict(self, X):
        check_is_fitted(self, "knn_")
        return self.knn_.predict(self._transform(X))

    def predict_proba(self, X):
        check_is_fitted(self, "knn_")
        return self.knn_.predict_proba(self._transform(X))

    def _fit_transform(self, X):
        if self.distance_transform == "raw_euclidean":
            self.transformer_ = None
            return np.asarray(X)
        if self.distance_transform == "standardized_euclidean":
            self.transformer_ = StandardScaler()
            return self.transformer_.fit_transform(X)
        if self.distance_transform == "log_euclidean":
            self.transformer_ = LogEuclideanTransformer().fit(X)
            return self.transformer_.transform(X)
        raise ValueError(
            "distance_transform must be one of "
            "{'raw_euclidean', 'standardized_euclidean', 'log_euclidean'}."
        )

    def _transform(self, X):
        if self.distance_transform == "raw_euclidean":
            return np.asarray(X)
        return self.transformer_.transform(X)
