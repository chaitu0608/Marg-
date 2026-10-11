import numpy as np
import pandas as pd

from marg.fraud.data import LABEL, time_split
from marg.fraud.thresholds import evaluate_cost_policy, select_cost_thresholds
from marg.fraud.train import recall_at_fpr


def test_time_split_has_strictly_ordered_disjoint_time_ranges():
    # Small synthetic imbalanced rows exist only in tests, never as reported model results.
    frame = pd.DataFrame({"Time": np.repeat(np.arange(50), 2), LABEL: [1] + [0] * 99})
    train, validation, test = time_split(frame.sample(frac=1, random_state=5))
    assert train.Time.max() < validation.Time.min()
    assert validation.Time.max() < test.Time.min()
    assert sum(map(len, (train, validation, test))) == len(frame)
    assert all(abs(len(part) / len(frame) - expected) <= 0.03
               for part, expected in zip((train, validation, test), (0.70, 0.15, 0.15)))


def test_fpr_threshold_is_finite_when_only_reject_all_meets_target():
    recall, threshold = recall_at_fpr(np.array([0, 0, 1, 1]),
                                     np.array([0.8, 0.7, 0.1, 0.2]), 0.01)
    assert recall == 0.0
    assert np.isfinite(threshold) and threshold > 0.8


def test_cost_thresholds_obey_validation_review_capacity_and_report_newer_period_cost():
    validation_labels = np.array([1, 0, 1, 0], dtype=np.int8)
    validation_scores = np.array([0.9, 0.9, 0.2, 0.1])
    validation_amounts = np.array([100.0, 100.0, 50.0, 50.0])
    policy = select_cost_thresholds(
        validation_labels, validation_scores, validation_amounts,
        max_review_rate=0.5, review_cost_amount_units=1.0, grid_size=5)
    validation = evaluate_cost_policy(
        validation_labels, validation_scores, validation_amounts,
        policy.review, policy.block, policy.review_cost_amount_units)

    newer_labels = np.array([1, 0], dtype=np.int8)
    newer_scores = np.array([0.3, 0.85])
    newer_amounts = np.array([70.0, 30.0])
    newer_period = evaluate_cost_policy(
        newer_labels, newer_scores, newer_amounts,
        policy.review, policy.block, policy.review_cost_amount_units)

    assert validation["review_count"] == 2
    assert validation["total_cost_amount_units"] == policy.validation_cost_amount_units
    assert validation["total_cost_amount_units"] < validation_amounts[validation_labels == 1].sum()
    assert newer_period["missed_fraud_count"] == 1
    assert newer_period["total_cost_amount_units"] == 71.0
