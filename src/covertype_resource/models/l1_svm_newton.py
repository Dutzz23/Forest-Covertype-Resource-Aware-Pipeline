from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


@dataclass
class BinaryNewtonState:
    coef: np.ndarray
    intercept: float
    objective_history: list[float]
    n_iter: int
    active_sizes: list[int]


class L1LinearSVMNewton(BaseEstimator, ClassifierMixin):
    """One-vs-rest L1-regularized linear SVM.

    The optimizer uses a Newton step on an active feature set for the squared
    hinge loss, plus a backtracking line search against the full L1 objective.
    This keeps each Newton system tiny for Covertype's 54 features while still
    scaling linearly in the number of rows.
    """

    def __init__(
        self,
        C: float = 1.0,
        alpha: float = 1e-3,
        max_iter: int = 40,
        tol: float = 1e-4,
        damping: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = True,
        line_search_shrink: float = 0.5,
        random_state: int = 42,
    ):
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.line_search_shrink = line_search_shrink
        self.random_state = random_state

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y).ravel()
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X_array.shape[1]

        if self.standardize:
            self.scaler_ = StandardScaler().fit(X_array)
            X_work = self.scaler_.transform(X_array)
        else:
            self.scaler_ = None
            X_work = X_array

        states = []
        for cls in self.classes_:
            binary_y = np.where(y_array == cls, 1.0, -1.0)
            states.append(self._fit_binary(X_work, binary_y))

        self.coef_ = np.vstack([state.coef for state in states])
        self.intercept_ = np.asarray([state.intercept for state in states])
        self.objective_history_ = [state.objective_history for state in states]
        self.n_iter_ = int(max(state.n_iter for state in states))
        self.active_sizes_ = [state.active_sizes for state in states]
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        X_array = np.asarray(X, dtype=np.float64)
        if self.scaler_ is not None:
            X_array = self.scaler_.transform(X_array)
        return X_array @ self.coef_.T + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> BinaryNewtonState:
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        history: list[float] = []
        active_sizes: list[int] = []

        for iteration in range(1, self.max_iter + 1):
            obj = self._objective(X, y, w, b)
            history.append(obj)
            grad_loss, grad_b, violating = self._smooth_gradient(X, y, w, b)
            min_norm_subgrad = self._minimum_norm_subgradient(w, grad_loss)
            active = (np.abs(w) > self.tol) | (np.abs(grad_loss) > self.alpha + self.tol)
            active_sizes.append(int(active.sum()))

            stationarity = max(np.max(np.abs(min_norm_subgrad)), abs(grad_b))
            if stationarity < self.tol:
                break

            direction_w = np.zeros_like(w)
            if active.any():
                XA = X[violating][:, active]
                H = (self.C / n_samples) * (XA.T @ XA)
                H.flat[:: H.shape[0] + 1] += self.damping
                if self.fit_intercept:
                    xb = (self.C / n_samples) * XA.sum(axis=0)
                    hb = np.block(
                        [
                            [H, xb[:, None]],
                            [xb[None, :], np.asarray([[self.C * max(1, violating.sum()) / n_samples + self.damping]])],
                        ]
                    )
                    rhs = -np.concatenate([min_norm_subgrad[active], [grad_b]])
                    step = np.linalg.solve(hb, rhs)
                    direction_w[active] = step[:-1]
                    direction_b = float(step[-1])
                else:
                    direction_w[active] = np.linalg.solve(H, -min_norm_subgrad[active])
                    direction_b = 0.0
            else:
                direction_b = -grad_b / (self.C + self.damping)

            if not np.all(np.isfinite(direction_w)) or not np.isfinite(direction_b):
                direction_w = -min_norm_subgrad
                direction_b = -grad_b

            accepted, w_next, b_next = self._line_search(X, y, w, b, direction_w, direction_b, obj)
            if not accepted:
                fallback_w = -min_norm_subgrad
                fallback_b = -grad_b
                accepted, w_next, b_next = self._line_search(X, y, w, b, fallback_w, fallback_b, obj)
            if not accepted:
                break
            w, b = w_next, b_next

        return BinaryNewtonState(coef=w, intercept=b, objective_history=history, n_iter=iteration, active_sizes=active_sizes)

    def _line_search(self, X, y, w, b, direction_w, direction_b, objective):
        step_size = 1.0
        norm_step = float(np.dot(direction_w, direction_w) + direction_b * direction_b)
        if norm_step <= 0.0:
            return False, w, b
        for _ in range(40):
            trial_w = w + step_size * direction_w
            trial_b = b + step_size * direction_b
            trial_obj = self._objective(X, y, trial_w, trial_b)
            if trial_obj < objective:
                return True, trial_w, trial_b
            step_size *= self.line_search_shrink
        return False, w, b

    def _objective(self, X, y, w, b) -> float:
        margins = y * (X @ w + b)
        violations = np.maximum(0.0, 1.0 - margins)
        return float(0.5 * self.C * np.mean(violations * violations) + self.alpha * np.abs(w).sum())

    def _smooth_gradient(self, X, y, w, b):
        margins = y * (X @ w + b)
        violations = 1.0 - margins
        active = violations > 0.0
        if not np.any(active):
            return np.zeros_like(w), 0.0, active
        scaled = -self.C * y[active] * violations[active] / X.shape[0]
        grad_w = X[active].T @ scaled
        grad_b = float(scaled.sum()) if self.fit_intercept else 0.0
        return grad_w, grad_b, active

    def _minimum_norm_subgradient(self, w, grad_loss):
        subgrad = grad_loss.copy()
        positive = w > 0
        negative = w < 0
        zero = ~(positive | negative)
        subgrad[positive] += self.alpha
        subgrad[negative] -= self.alpha
        subgrad[zero] = np.sign(grad_loss[zero]) * np.maximum(np.abs(grad_loss[zero]) - self.alpha, 0.0)
        return subgrad
