from __future__ import annotations

import json
import pickle
import uuid
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model, path: Path | str, use_pickle: bool = False) -> Path:
    """Persist a model immediately after training."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    if use_pickle:
        with tmp_path.open("wb") as fh:
            pickle.dump(model, fh)
    else:
        joblib.dump(model, tmp_path)
    tmp_path.replace(path)
    return path


def save_json(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def batch_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


class SequentialKappa:
    """Online accuracy and Cohen's Kappa from a growing confusion matrix."""

    def __init__(self, classes: Iterable[int]):
        self.classes_ = np.asarray(list(classes))
        self.index_ = {label: i for i, label in enumerate(self.classes_)}
        n_classes = len(self.classes_)
        self.confusion_ = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.n_ = 0
        self.correct_ = 0

    def update(self, y_true, y_pred) -> dict:
        true_idx = self.index_[y_true]
        pred_idx = self.index_[y_pred]
        self.confusion_[true_idx, pred_idx] += 1
        self.n_ += 1
        self.correct_ += int(y_true == y_pred)
        return self.get()

    def get(self) -> dict:
        if self.n_ == 0:
            return {"n": 0, "accuracy": 0.0, "kappa": 0.0}
        observed = self.correct_ / self.n_
        true_marginals = self.confusion_.sum(axis=1)
        pred_marginals = self.confusion_.sum(axis=0)
        expected = float(np.dot(true_marginals, pred_marginals) / (self.n_ ** 2))
        denom = 1.0 - expected
        kappa = 0.0 if abs(denom) < 1e-15 else (observed - expected) / denom
        return {"n": int(self.n_), "accuracy": float(observed), "kappa": float(kappa)}
