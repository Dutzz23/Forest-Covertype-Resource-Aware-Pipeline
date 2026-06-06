from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..evaluation import SequentialKappa


class StreamTreeNBLeaves(BaseEstimator, ClassifierMixin):
    """Hoeffding/EFDT stream tree wrapper with Naive Bayes leaves."""

    def __init__(
        self,
        tree_type: str = "efdt",
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nb",
        report_every: int = 50_000,
    ):
        self.tree_type = tree_type
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.report_every = report_every

    def fit(self, X, y):
        return self.fit_prequential(X, y)

    def fit_prequential(self, X, y):
        from river import tree

        if self.tree_type == "efdt" and hasattr(tree, "ExtremelyFastDecisionTreeClassifier"):
            model_cls = tree.ExtremelyFastDecisionTreeClassifier
        elif self.tree_type in {"efdt", "hoeffding"}:
            model_cls = tree.HoeffdingTreeClassifier
        else:
            raise ValueError("tree_type must be 'efdt' or 'hoeffding'.")

        self.model_ = model_cls(
            grace_period=self.grace_period,
            delta=self.delta,
            tau=self.tau,
            leaf_prediction=self.leaf_prediction,
        )
        y_array = np.asarray(y).ravel()
        self.classes_ = np.unique(y_array)
        metrics = SequentialKappa(self.classes_)
        seen = Counter()
        self.timeline_ = []

        for i, (x_dict, y_true) in enumerate(self._iter_rows(X, y_array), start=1):
            y_pred = self.model_.predict_one(x_dict)
            if y_pred is None:
                y_pred = seen.most_common(1)[0][0] if seen else y_true
            snapshot = metrics.update(y_true, y_pred)
            self.model_.learn_one(x_dict, y_true)
            seen[y_true] += 1
            if i == 1 or i % self.report_every == 0 or i == len(y_array):
                self.timeline_.append(snapshot.copy())

        self.final_metrics_ = metrics.get()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        preds = []
        for x_dict, _ in self._iter_rows(X, np.zeros(len(X), dtype=int)):
            y_pred = self.model_.predict_one(x_dict)
            preds.append(self.classes_[0] if y_pred is None else y_pred)
        return np.asarray(preds)

    @staticmethod
    def _iter_rows(X, y):
        if isinstance(X, pd.DataFrame):
            columns = list(X.columns)
            for values, y_true in zip(X.itertuples(index=False, name=None), y):
                yield dict(zip(columns, values)), y_true
        else:
            X_array = np.asarray(X)
            columns = [f"x{i}" for i in range(X_array.shape[1])]
            for values, y_true in zip(X_array, y):
                yield dict(zip(columns, values)), y_true
