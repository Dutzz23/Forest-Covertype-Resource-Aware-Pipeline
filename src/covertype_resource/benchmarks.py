from __future__ import annotations

import time

from sklearn.cluster import MiniBatchKMeans


def benchmark_first_150k_k2_means(X, n_clusters: int = 7, random_state: int = 42) -> dict:
    """Fast large-scale clustering benchmark over the first 150,000 rows.

    MiniBatchKMeans is used as a practical k^2-means-style low-energy baseline:
    it reports inertia as the clustering energy and wall time for the requested
    subset size.
    """

    X_subset = X.iloc[:150_000] if hasattr(X, "iloc") else X[:150_000]
    started = time.perf_counter()
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init="auto",
    ).fit(X_subset)
    elapsed = time.perf_counter() - started
    return {
        "rows": int(len(X_subset)),
        "n_clusters": int(n_clusters),
        "inertia": float(model.inertia_),
        "seconds": float(elapsed),
    }
