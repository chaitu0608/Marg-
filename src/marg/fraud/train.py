from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    roc_curve,
)

from marg.fraud.data import FEATURES, LABEL, load_creditcard, time_split
from marg.fraud.thresholds import evaluate_cost_policy, select_cost_thresholds

ROOT = Path(__file__).resolve().parents[3]


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = np.flatnonzero(fpr <= max_fpr)
    i = int(eligible[np.argmax(tpr[eligible])])
    threshold = float(thresholds[i])
    if np.isinf(threshold):
        # If no positive is reachable within the FPR budget, use a finite reject-all cutoff.
        threshold = float(np.nextafter(float(scores.max()), np.inf))
    return float(tpr[i]), threshold


def train(path: str | Path = "data/creditcard.csv", seed: int = 7,
          max_review_rate: float = 0.01, review_cost_amount_units: float = 1.0) -> dict:
    frame = load_creditcard(path)
    train_df, val_df, test_df = time_split(frame)
    x_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df[LABEL].to_numpy(dtype=np.int8)
    x_val = val_df[FEATURES].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL].to_numpy(dtype=np.int8)
    amount_val = val_df["Amount"].to_numpy(dtype=np.float64)
    x_test = test_df[FEATURES].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL].to_numpy(dtype=np.int8)
    amount_test = test_df["Amount"].to_numpy(dtype=np.float64)
    positives = int(y_train.sum())
    model = LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31,
                           max_depth=-1, scale_pos_weight=(len(y_train) - positives) / positives,
                           # Class weighting on a rare class can produce extreme leaf updates;
                           # cap each step and regularize leaves to keep scores explainable.
                           max_delta_step=1.0, min_child_samples=50, reg_lambda=2.0,
                           random_state=seed, n_jobs=4, verbosity=-1)
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    cost_thresholds = select_cost_thresholds(
        y_val, val_scores, amount_val, max_review_rate, review_cost_amount_units)
    block_threshold, review_threshold = cost_thresholds.block, cost_thresholds.review
    validation_block_fpr = float(np.mean(
        (val_scores >= block_threshold) & (y_val == 0))) / max(float(np.mean(y_val == 0)), 1e-12)
    validation_review_fpr = float(np.mean(
        (val_scores >= review_threshold) & (y_val == 0))) / max(float(np.mean(y_val == 0)), 1e-12)
    test_scores = model.predict_proba(x_test)[:, 1]
    test_recall, _ = recall_at_fpr(y_test, test_scores, 0.01)
    y_pred = (test_scores >= block_threshold).astype(np.int8)
    validation_cost = evaluate_cost_policy(
        y_val, val_scores, amount_val, review_threshold, block_threshold,
        review_cost_amount_units)
    test_cost = evaluate_cost_policy(
        y_test, test_scores, amount_test, review_threshold, block_threshold,
        review_cost_amount_units)
    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "fraud_model.joblib")
    (models_dir / "thresholds.json").write_text(json.dumps({
        "block": float(block_threshold), "review": float(review_threshold),
        "selection": "validation amount-weighted cost with review capacity",
        "max_review_rate": max_review_rate,
        "review_cost_amount_units": review_cost_amount_units,
        "validation_cost_amount_units": validation_cost["total_cost_amount_units"],
        "validation_review_count": validation_cost["review_count"]}, indent=2))
    metrics = {
        "dataset": "real Kaggle Credit Card Fraud Detection via Zenodo mirror",
        "dataset_path": str(Path(path)), "rows": len(frame), "fraud_rate": float(frame[LABEL].mean()),
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "split_time_ranges": {"train": [float(train_df.Time.min()), float(train_df.Time.max())],
                              "validation": [float(val_df.Time.min()), float(val_df.Time.max())],
                              "test": [float(test_df.Time.min()), float(test_df.Time.max())]},
        "pr_auc": float(average_precision_score(y_test, test_scores)),
        "recall_at_1pct_fpr": test_recall,
        "precision_at_block_threshold": float(precision_score(y_test, y_pred, zero_division=0)),
        "cost_policy": {
            "selection_period": "validation; test is a strictly newer time period",
            "amount_interpretation": "dataset Amount units; currency and actual fraud loss are unknown",
            "review_assumption": "manual review resolves fraud without further loss",
            "max_review_rate": max_review_rate,
            "review_cost_amount_units": review_cost_amount_units,
            "validation": validation_cost,
            "newer_test_period": test_cost,
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "block_threshold": float(block_threshold), "review_threshold": float(review_threshold),
        "validation_fpr_at_block_threshold": validation_block_fpr,
        "validation_fpr_at_review_threshold": validation_review_fpr,
        "feature_names": FEATURES,
        "model_parameters": {"n_estimators": 250, "learning_rate": 0.05,
                             "num_leaves": 31, "max_delta_step": 1.0,
                             "min_child_samples": 50, "reg_lambda": 2.0,
                             "scale_pos_weight": float((len(y_train) - positives) / positives)},
    }
    (results_dir / "fraud_metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set(title="Marg test-set precision-recall curve", xlabel="Recall", ylabel="Precision")
    fig.tight_layout()
    fig.savefig(results_dir / "fraud_pr_curve.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(test_scores[y_test == 0], bins=60, alpha=0.7, label="legitimate")
    ax.hist(test_scores[y_test == 1], bins=40, alpha=0.7, label="fraud")
    ax.axvline(block_threshold, color="black", linestyle="--", label="block threshold")
    ax.set(title="Marg fraud model test scores", xlabel="Fraud score", ylabel="Transactions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "fraud_scores.png", dpi=140)
    plt.close(fig)
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-review-rate", type=float, default=0.01)
    parser.add_argument("--review-cost-amount-units", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.seed, args.max_review_rate,
                          args.review_cost_amount_units), indent=2))


if __name__ == "__main__":
    main()
