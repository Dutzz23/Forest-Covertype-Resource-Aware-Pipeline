from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ScaleInvariantOnlineLinearClassifier(BaseEstimator, ClassifierMixin):
    """ScInOL-style online softmax model with per-feature scale invariance.

    The model keeps a running maximum absolute coordinate scale and learns in
    the normalized coordinate system. If an instance expands a coordinate scale,
    the corresponding weights are rescaled to preserve existing predictions.
    Updates use per-coordinate AdaGrad steps, avoiding hand-tuned learning rates
    per feature.
    """

    def __init__(self, epochs: int = 1, shuffle: bool = False, random_state: int = 42, eps: float = 1e-12):
        self.epochs = epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.eps = eps

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y).ravel()
        self.classes_ = np.unique(y_array)
        self._class_to_idx = {label: i for i, label in enumerate(self.classes_)}
        self.n_features_in_ = X_array.shape[1]
        self.scale_ = np.ones(self.n_features_in_, dtype=np.float64)
        self.coef_ = np.zeros((len(self.classes_), self.n_features_in_), dtype=np.float64)
        self.intercept_ = np.zeros(len(self.classes_), dtype=np.float64)
        self.grad_sq_ = np.zeros_like(self.coef_)
        self.intercept_grad_sq_ = np.zeros_like(self.intercept_)

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(X_array.shape[0])
        for _ in range(self.epochs):
            if self.shuffle:
                rng.shuffle(indices)
            for i in indices:
                self.partial_fit_one(X_array[i], y_array[i])
        return self

    def partial_fit(self, X, y, classes=None):
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y).ravel()
        if not hasattr(self, "classes_"):
            if classes is None:
                classes = np.unique(y_array)
            self.classes_ = np.asarray(classes)
            self._class_to_idx = {label: i for i, label in enumerate(self.classes_)}
            self.n_features_in_ = X_array.shape[1]
            self.scale_ = np.ones(self.n_features_in_, dtype=np.float64)
            self.coef_ = np.zeros((len(self.classes_), self.n_features_in_), dtype=np.float64)
            self.intercept_ = np.zeros(len(self.classes_), dtype=np.float64)
            self.grad_sq_ = np.zeros_like(self.coef_)
            self.intercept_grad_sq_ = np.zeros_like(self.intercept_)
        for x_i, y_i in zip(X_array, y_array):
            self.partial_fit_one(x_i, y_i)
        return self

    def partial_fit_one(self, x, y):
        abs_x = np.abs(x)
        expanded = abs_x > self.scale_
        if np.any(expanded):
            ratio = abs_x[expanded] / self.scale_[expanded]
            self.coef_[:, expanded] *= ratio
            self.scale_[expanded] = abs_x[expanded]

        x_norm = x / np.maximum(self.scale_, self.eps)
        probs = self._softmax(self.coef_ @ x_norm + self.intercept_)
        grad_scores = probs
        grad_scores[self._class_to_idx[y]] -= 1.0

        grad_w = grad_scores[:, None] * x_norm[None, :]
        self.grad_sq_ += grad_w * grad_w
        self.coef_ -= grad_w / (np.sqrt(self.grad_sq_) + self.eps)

        self.intercept_grad_sq_ += grad_scores * grad_scores
        self.intercept_ -= grad_scores / (np.sqrt(self.intercept_grad_sq_) + self.eps)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "coef_")
        X_array = np.asarray(X, dtype=np.float64)
        X_norm = X_array / np.maximum(self.scale_, self.eps)
        scores = X_norm @ self.coef_.T + self.intercept_
        return np.vstack([self._softmax(row) for row in scores])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @staticmethod
    def _softmax(scores):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum()
