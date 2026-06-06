from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .benchmarks import benchmark_first_150k_k2_means
from .data import exact_holdout_split, load_covertype
from .evaluation import batch_metrics, save_json, save_model
from .models.knn_logeuclidean import LogEuclideanKNNClassifier
from .models.l1_svm_newton import L1LinearSVMNewton
from .models.opf import LibOPFClassifier
from .models.resource_rf import BudgetPrunedForestClassifier, fit_random_forest_baseline
from .models.scale_invariant_online import ScaleInvariantOnlineLinearClassifier
from .models.stream_trees import StreamTreeNBLeaves


ALL_MODELS = {"rf", "budgetprune", "stream", "l1svm", "opf", "knn", "online"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resource-aware Forest Covertype training pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'.")
    parser.add_argument("--model-dir", default="models", help="Directory for persisted models.")
    parser.add_argument("--report-dir", default="reports", help="Directory for metrics JSON files.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-budget", type=float, default=None, help="Optional total feature acquisition budget.")
    parser.add_argument("--rf-cost-penalty", type=float, default=0.01)
    parser.add_argument("--rf-solver", choices=["milp", "lp_relaxation"], default="milp")
    parser.add_argument("--opf-max-train", type=int, default=5000)
    parser.add_argument("--stream-tree", choices=["efdt", "hoeffding"], default="efdt")
    parser.add_argument("--stream-report-every", type=int, default=50_000)
    parser.add_argument("--online-epochs", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for sklearn models.")
    parser.add_argument("--run-k2means", action="store_true")
    return parser.parse_args()


def requested_models(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return set(ALL_MODELS)
    models = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = models - ALL_MODELS
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    return models


def main() -> None:
    args = parse_args()
    selected = requested_models(args.models)
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    metrics_path = report_dir / "covertype_metrics.json"
    metrics: dict[str, object] = {
        "dataset": {},
        "batch_holdout": {"test_size": 0.20, "random_state": args.random_state},
        "models": {},
    }

    dataset = load_covertype(limit=args.limit)
    X, y = dataset.X, dataset.y
    metrics["dataset"] = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "classes": [int(cls) for cls in np.unique(y)],
        "limit": args.limit,
    }

    if "stream" in selected:
        stream_model = StreamTreeNBLeaves(
            tree_type=args.stream_tree,
            leaf_prediction="nb",
            report_every=args.stream_report_every,
        )
        stream_model.fit_prequential(X, y)
        save_model(stream_model, model_dir / f"{args.stream_tree}_nb_leaves.pkl", use_pickle=True)
        metrics["models"][f"{args.stream_tree}_nb_leaves"] = {
            "final": stream_model.final_metrics_,
            "timeline": stream_model.timeline_,
        }
        save_json(metrics, metrics_path)

    batch_needed = selected & {"rf", "budgetprune", "l1svm", "opf", "knn", "online"}
    if batch_needed:
        X_train, X_test, y_train, y_test = exact_holdout_split(
            X,
            y,
            test_size=0.20,
            random_state=args.random_state,
        )
    else:
        X_train = X_test = y_train = y_test = None

    forest = None
    if {"rf", "budgetprune"} & selected:
        stratify = y_train if y_train.value_counts().min() >= 2 else None
        X_rf_fit, X_prune_val, y_rf_fit, y_prune_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=args.random_state,
            stratify=stratify,
        )
        forest = fit_random_forest_baseline(random_state=args.random_state, n_jobs=args.n_jobs)
        forest.fit(X_rf_fit, y_rf_fit)
        save_model(forest, model_dir / "random_forest_entropy_40.joblib")
        if "rf" in selected:
            metrics["models"]["random_forest_entropy_40"] = batch_metrics(y_test, forest.predict(X_test))
            save_json(metrics, metrics_path)

        if "budgetprune" in selected:
            pruned = BudgetPrunedForestClassifier(
                forest=forest,
                budget=args.rf_budget,
                cost_penalty=args.rf_cost_penalty,
                solver=args.rf_solver,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            pruned.fit(X_rf_fit, y_rf_fit, X_val=X_prune_val, y_val=y_prune_val)
            save_model(pruned, model_dir / "budgetpruned_random_forest.joblib")
            pruned_metrics = batch_metrics(y_test, pruned.predict(X_test))
            pruned_metrics["selected_trees"] = int(len(pruned.selected_tree_indices_))
            pruned_metrics["selected_features"] = int(len(pruned.selected_features_))
            pruned_metrics["feature_acquisition_cost"] = pruned.feature_acquisition_cost_
            pruned_metrics["solver"] = pruned.solution_.solver
            pruned_metrics["solver_status"] = pruned.solution_.status
            metrics["models"]["budgetpruned_random_forest"] = pruned_metrics
            save_json(metrics, metrics_path)

    if "online" in selected:
        online = ScaleInvariantOnlineLinearClassifier(epochs=args.online_epochs, shuffle=True, random_state=args.random_state)
        online.fit(X_train, y_train)
        save_model(online, model_dir / "scale_invariant_online_linear.joblib")
        metrics["models"]["scale_invariant_online_linear"] = batch_metrics(y_test, online.predict(X_test))
        save_json(metrics, metrics_path)

    if "l1svm" in selected:
        l1svm = L1LinearSVMNewton()
        l1svm.fit(X_train, y_train)
        save_model(l1svm, model_dir / "l1_svm_newton_active_set.joblib")
        l1_metrics = batch_metrics(y_test, l1svm.predict(X_test))
        l1_metrics["max_newton_iterations"] = l1svm.n_iter_
        metrics["models"]["l1_svm_newton_active_set"] = l1_metrics
        save_json(metrics, metrics_path)

    if "opf" in selected:
        opf_max = None if args.opf_max_train <= 0 else args.opf_max_train
        opf = LibOPFClassifier(backend="auto", max_train_samples=opf_max, random_state=args.random_state)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opf.fit(X_train, y_train)
        save_model(opf, model_dir / "opf_log_euclidean.joblib")
        opf_metrics = batch_metrics(y_test, opf.predict(X_test))
        opf_metrics["backend"] = opf.backend_
        opf_metrics["warnings"] = [str(item.message) for item in caught]
        metrics["models"]["opf_log_euclidean"] = opf_metrics
        save_json(metrics, metrics_path)

    if "knn" in selected:
        knn = LogEuclideanKNNClassifier(k_values=(1, 3, 5), n_jobs=args.n_jobs)
        knn.fit(X_train, y_train)
        save_model(knn, model_dir / "knn_log_euclidean_loocv.joblib")
        knn_metrics = batch_metrics(y_test, knn.predict(X_test))
        knn_metrics["best_k"] = int(knn.best_k_)
        knn_metrics["leave_one_out_scores"] = {str(k): v for k, v in knn.k_selection_.scores.items()}
        metrics["models"]["knn_log_euclidean_loocv"] = knn_metrics
        save_json(metrics, metrics_path)

    if args.run_k2means:
        metrics["k2means_first_150k"] = benchmark_first_150k_k2_means(X, random_state=args.random_state)
        save_json(metrics, metrics_path)

    save_json(metrics, metrics_path)
    print(f"Saved models to {model_dir.resolve()}")
    print(f"Saved metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
