from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted


def fit_random_forest_baseline(random_state: int = 42, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=40,
        criterion="entropy",
        random_state=random_state,
        n_jobs=n_jobs,
    )


@dataclass
class BudgetSolution:
    selected_trees: np.ndarray
    selected_features: np.ndarray
    objective_value: float
    solver: str
    status: str
    relaxed_solution: np.ndarray | None = None


class BudgetPrunedForestClassifier(BaseEstimator, ClassifierMixin):
    """BUDGETPRUNE-style constrained pruning for a fitted random forest.

    Decision variables select trees and feature-acquisition indicators. The
    constraints enforce feature reuse: a selected tree can only be used if all
    features it references have been acquired once.
    """

    def __init__(
        self,
        forest: Optional[RandomForestClassifier] = None,
        feature_costs: Optional[np.ndarray] = None,
        budget: Optional[float] = None,
        cost_penalty: float = 0.01,
        min_trees: int = 1,
        solver: str = "milp",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.forest = forest
        self.feature_costs = feature_costs
        self.budget = budget
        self.cost_penalty = cost_penalty
        self.min_trees = min_trees
        self.solver = solver
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, X_val=None, y_val=None):
        if self.forest is None:
            self.forest_ = fit_random_forest_baseline(self.random_state, self.n_jobs)
            self.forest_.fit(X, y)
        else:
            self.forest_ = self.forest

        if X_val is None or y_val is None:
            X_val = X
            y_val = y

        self.classes_ = np.asarray(self.forest_.classes_)
        self.n_features_in_ = self.forest_.n_features_in_
        self.feature_names_in_ = getattr(self.forest_, "feature_names_in_", None)
        self.tree_feature_incidence_ = self._tree_feature_incidence()
        self.tree_utilities_ = self._tree_utilities(X_val, y_val)

        costs = np.ones(self.n_features_in_, dtype=np.float64)
        if self.feature_costs is not None:
            costs = np.asarray(self.feature_costs, dtype=np.float64)
            if costs.shape[0] != self.n_features_in_:
                raise ValueError("feature_costs must match the number of features.")
        self.feature_costs_ = costs

        self.solution_ = self._solve_budget_problem()
        self.selected_tree_indices_ = np.flatnonzero(self.solution_.selected_trees)
        if self.selected_tree_indices_.size == 0:
            best = int(np.argmax(self.tree_utilities_))
            self.selected_tree_indices_ = np.asarray([best])
            selected = np.zeros(len(self.forest_.estimators_), dtype=bool)
            selected[best] = True
            self.solution_.selected_trees = selected

        self.selected_estimators_ = [self.forest_.estimators_[i] for i in self.selected_tree_indices_]
        self.selected_features_ = np.flatnonzero(
            self.tree_feature_incidence_[self.selected_tree_indices_].any(axis=0)
        )
        self.feature_acquisition_cost_ = float(self.feature_costs_[self.selected_features_].sum())
        return self

    def predict(self, X):
        check_is_fitted(self, "selected_estimators_")
        X_array = np.asarray(X)
        votes = np.asarray([self._predict_tree(est, X_array) for est in self.selected_estimators_])
        pred = []
        class_to_idx = {label: i for i, label in enumerate(self.classes_)}
        for column in votes.T:
            encoded = np.fromiter((class_to_idx[label] for label in column), dtype=np.int64)
            pred.append(self.classes_[np.bincount(encoded, minlength=len(self.classes_)).argmax()])
        return np.asarray(pred)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _tree_feature_incidence(self) -> np.ndarray:
        incidence = np.zeros((len(self.forest_.estimators_), self.n_features_in_), dtype=bool)
        for t, estimator in enumerate(self.forest_.estimators_):
            used = np.unique(estimator.tree_.feature)
            used = used[used >= 0]
            incidence[t, used] = True
        return incidence

    def _tree_utilities(self, X_val, y_val) -> np.ndarray:
        X_array = np.asarray(X_val)
        return np.asarray(
            [accuracy_score(y_val, self._predict_tree(estimator, X_array)) for estimator in self.forest_.estimators_],
            dtype=np.float64,
        )

    def _predict_tree(self, estimator, X):
        raw = np.asarray(estimator.predict(X))
        integer_like = np.all(np.isclose(raw, np.round(raw)))
        encoded_range = raw.size > 0 and raw.min() >= 0 and raw.max() < len(self.classes_)
        if integer_like and encoded_range and not np.all(np.isin(raw, self.classes_)):
            return self.classes_[raw.astype(np.int64)]
        return raw

    def _build_problem(self):
        n_trees = len(self.forest_.estimators_)
        n_features = self.n_features_in_
        n_vars = n_trees + n_features

        # scipy solvers minimize, so maximize utility minus cost by negating.
        c = np.concatenate(
            [
                -self.tree_utilities_,
                self.cost_penalty * self.feature_costs_,
            ]
        )

        edge_count = int(self.tree_feature_incidence_.sum())
        n_rows = edge_count + 1 + int(self.budget is not None)
        A = lil_matrix((n_rows, n_vars), dtype=np.float64)
        lower = np.full(n_rows, -np.inf, dtype=np.float64)
        upper = np.full(n_rows, np.inf, dtype=np.float64)

        row = 0
        for tree_idx, feature_idx in np.argwhere(self.tree_feature_incidence_):
            # z_tree - u_feature <= 0
            A[row, tree_idx] = 1.0
            A[row, n_trees + feature_idx] = -1.0
            upper[row] = 0.0
            row += 1

        A[row, :n_trees] = 1.0
        lower[row] = float(self.min_trees)
        row += 1

        if self.budget is not None:
            A[row, n_trees:] = self.feature_costs_
            upper[row] = float(self.budget)

        return c, A.tocsr(), lower, upper

    def _solve_lp_relaxation(self, c, A, lower, upper):
        bounds = [(0.0, 1.0)] * len(c)
        a_ub = []
        b_ub = []
        for i in range(A.shape[0]):
            row = A.getrow(i)
            if np.isfinite(upper[i]):
                a_ub.append(row.toarray().ravel())
                b_ub.append(upper[i])
            if np.isfinite(lower[i]):
                a_ub.append(-row.toarray().ravel())
                b_ub.append(-lower[i])
        res = linprog(c, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
        return res

    def _solve_budget_problem(self) -> BudgetSolution:
        c, A, lower, upper = self._build_problem()
        n_trees = len(self.forest_.estimators_)
        relaxed = self._solve_lp_relaxation(c, A, lower, upper)
        relaxed_x = relaxed.x if relaxed.success else None

        if self.solver == "lp_relaxation":
            if not relaxed.success:
                raise RuntimeError(f"LP relaxation failed: {relaxed.message}")
            x = relaxed.x
            status = relaxed.message
            solver = "lp_relaxation"
        else:
            constraints = LinearConstraint(A, lower, upper)
            bounds = Bounds(np.zeros_like(c), np.ones_like(c))
            res = milp(
                c=c,
                integrality=np.ones_like(c),
                bounds=bounds,
                constraints=constraints,
                options={"time_limit": 300.0},
            )
            if not res.success:
                if not relaxed.success:
                    raise RuntimeError(f"MILP failed: {res.message}; LP relaxation failed: {relaxed.message}")
                x = relaxed.x
                status = f"MILP failed ({res.message}); used LP relaxation."
                solver = "lp_relaxation_fallback"
            else:
                x = res.x
                status = res.message
                solver = "milp"

        selected_trees = x[:n_trees] >= 0.5
        selected_features = x[n_trees:] >= 0.5
        return BudgetSolution(
            selected_trees=selected_trees,
            selected_features=selected_features,
            objective_value=float(-np.dot(c, x)),
            solver=solver,
            status=status,
            relaxed_solution=relaxed_x,
        )
