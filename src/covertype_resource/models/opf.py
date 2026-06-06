from __future__ import annotations

import heapq
import warnings

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted

from ..metrics import LogEuclideanTransformer


class LibOPFClassifier(BaseEstimator, ClassifierMixin):
    """OPF baseline with a LibOPF-compatible backend when installed.

    Python environments do not always expose the original LibOPF bindings. This
    class first tries common OPF bindings and otherwise falls back to a compact
    supervised OPF implementation over a complete graph. The fallback is exact
    for the retained training set and remains parameter-free; ``max_train_samples``
    only protects machines from the O(n^2) graph memory cost.
    """

    def __init__(
        self,
        backend: str = "auto",
        max_train_samples: int | None = 5000,
        random_state: int = 42,
        prediction_batch_size: int = 5000,
    ):
        self.backend = backend
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.prediction_batch_size = prediction_batch_size

    def fit(self, X, y):
        self.transformer_ = LogEuclideanTransformer().fit(X)
        X_log = self.transformer_.transform(X)
        y_array = np.asarray(y).ravel()
        self.classes_ = np.unique(y_array)

        if self.backend in {"auto", "libopf", "opfython"} and self._try_external_fit(X_log, y_array):
            return self

        if self.backend in {"libopf", "opfython"}:
            raise ImportError(f"Requested OPF backend {self.backend!r} is not available.")

        X_work, y_work = self._maybe_subsample(X_log, y_array)
        self._fit_fallback(X_work, y_work)
        self.backend_ = "fallback_complete_graph_opf"
        return self

    def predict(self, X):
        check_is_fitted(self, "backend_")
        X_log = self.transformer_.transform(X)
        if self.backend_ == "opfython":
            return np.asarray(self.external_model_.predict(X_log)).ravel()
        return self._predict_fallback(X_log)

    def _try_external_fit(self, X_log, y):
        if self.backend in {"auto", "opfython"}:
            try:
                from opfython.models import SupervisedOPF

                model = SupervisedOPF()
                model.fit(X_log, y)
                self.external_model_ = model
                self.backend_ = "opfython"
                return True
            except Exception as exc:  # pragma: no cover - depends on optional binding
                if self.backend == "opfython":
                    raise
                warnings.warn(f"opfython backend unavailable; using fallback OPF. Reason: {exc}")
        return False

    def _maybe_subsample(self, X_log, y):
        if self.max_train_samples is None or X_log.shape[0] <= self.max_train_samples:
            self.subsample_indices_ = None
            return X_log, y
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.max_train_samples,
            random_state=self.random_state,
        )
        indices, _ = next(splitter.split(X_log, y))
        self.subsample_indices_ = indices
        return X_log[indices], y[indices]

    def _fit_fallback(self, X_log, y):
        self.X_train_log_ = np.asarray(X_log, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        distances = pairwise_distances(self.X_train_log_, metric="euclidean")
        mst = minimum_spanning_tree(distances).tocoo()

        prototypes = set()
        for i, j in zip(mst.row, mst.col):
            if self.y_train_[i] != self.y_train_[j]:
                prototypes.add(int(i))
                prototypes.add(int(j))

        if not prototypes:
            for cls in self.classes_:
                prototypes.add(int(np.flatnonzero(self.y_train_ == cls)[0]))

        self.prototype_indices_ = np.asarray(sorted(prototypes), dtype=np.int64)
        self.path_costs_, self.optimum_labels_ = self._minimax_paths(distances, self.prototype_indices_)

    def _minimax_paths(self, distances, prototypes):
        n = distances.shape[0]
        costs = np.full(n, np.inf, dtype=np.float64)
        labels = np.empty(n, dtype=self.y_train_.dtype)
        visited = np.zeros(n, dtype=bool)
        heap = []
        for proto in prototypes:
            costs[proto] = 0.0
            labels[proto] = self.y_train_[proto]
            heapq.heappush(heap, (0.0, int(proto)))

        while heap:
            cost, node = heapq.heappop(heap)
            if visited[node]:
                continue
            visited[node] = True
            candidates = np.maximum(cost, distances[node])
            improved = candidates < costs
            improved[visited] = False
            for target in np.flatnonzero(improved):
                costs[target] = candidates[target]
                labels[target] = labels[node]
                heapq.heappush(heap, (float(costs[target]), int(target)))
        return costs, labels

    def _predict_fallback(self, X_log):
        batch_size = max(1, int(self.prediction_batch_size))
        preds = []
        for start in range(0, X_log.shape[0], batch_size):
            stop = min(start + batch_size, X_log.shape[0])
            distances = pairwise_distances(X_log[start:stop], self.X_train_log_, metric="euclidean")
            conquest_costs = np.maximum(distances, self.path_costs_[None, :])
            winners = np.argmin(conquest_costs, axis=1)
            preds.append(self.optimum_labels_[winners])
        return np.concatenate(preds)
