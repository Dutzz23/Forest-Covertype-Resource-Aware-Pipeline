from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_CACHE = Path("data") / "covertype_ucimlrepo.csv"
TARGET_NAME = "Cover_Type"


@dataclass(frozen=True)
class CovertypeDataset:
    X: pd.DataFrame
    y: pd.Series
    metadata: object | None = None
    variables: object | None = None


def load_covertype(
    cache_path: Path | str = DATA_CACHE,
    limit: Optional[int] = None,
    force_download: bool = False,
) -> CovertypeDataset:
    """Load the UCI Forest Covertype data with a local CSV cache.

    The UCI dataset has 581,012 rows. A ``limit`` is useful for smoke tests but
    is deliberately applied after loading so full-run semantics stay unchanged.
    """

    cache_path = Path(cache_path)
    metadata = None
    variables = None

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path)
    else:
        from ucimlrepo import fetch_ucirepo

        covertype = fetch_ucirepo(id=31)
        X = covertype.data.features.copy()
        y = covertype.data.targets.copy()
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = pd.Series(y, name=TARGET_NAME)
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        metadata = covertype.metadata
        variables = covertype.variables

    if limit is not None:
        df = df.iloc[:limit].copy()

    if TARGET_NAME not in df.columns:
        # ucimlrepo usually names the target "Cover_Type", but keep this loader
        # tolerant of older local caches.
        target_candidates = [c for c in df.columns if c.lower() in {"cover_type", "covertype"}]
        if not target_candidates:
            raise ValueError(f"Could not find target column {TARGET_NAME!r}.")
        df = df.rename(columns={target_candidates[0]: TARGET_NAME})

    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME].astype(int)
    return CovertypeDataset(X=X, y=y, metadata=metadata, variables=variables)


def encode_labels(y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode cover labels to contiguous integers while returning class labels."""

    y_array = np.asarray(y).ravel()
    classes, encoded = np.unique(y_array, return_inverse=True)
    return encoded.astype(np.int64), classes


def exact_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Batch split used by all non-streaming algorithms.

    ``test_size=0.20`` is intentionally the default to match the requested
    exactly 20 percent holdout.
    """

    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
