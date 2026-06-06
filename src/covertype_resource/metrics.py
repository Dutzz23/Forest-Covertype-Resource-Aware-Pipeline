from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted


class LogEuclideanTransformer(BaseEstimator, TransformerMixin):
    """Map vectors into a log domain for Log-Euclidean instance distances.

    A Log-Euclidean metric is naturally defined on positive coordinates. The
    Covertype table has zeros and one signed hydrology-distance feature, so the
    transformer learns the smallest non-negative offset that makes every
    training coordinate at least one before applying ``log``.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def fit(self, X, y=None):
        X_checked = check_array(X, accept_sparse=False, dtype=np.float64)
        mins = np.nanmin(X_checked, axis=0)
        self.offset_ = np.where(mins <= 0.0, 1.0 - mins, 0.0)
        return self

    def transform(self, X):
        check_is_fitted(self, "offset_")
        X_checked = check_array(X, accept_sparse=False, dtype=np.float64)
        shifted = X_checked + self.offset_
        shifted = np.maximum(shifted, self.eps)
        return np.log(shifted)

    def distance(self, x, z) -> float:
        tx = self.transform(np.asarray(x).reshape(1, -1))
        tz = self.transform(np.asarray(z).reshape(1, -1))
        return float(np.linalg.norm(tx - tz))


def pairwise_log_euclidean(transformer: LogEuclideanTransformer, X, Y=None, n_jobs=None):
    """Pairwise Euclidean distances after the learned log-domain transform."""

    X_log = transformer.transform(X)
    Y_log = None if Y is None else transformer.transform(Y)
    return pairwise_distances(X_log, Y_log, metric="euclidean", n_jobs=n_jobs)
