import numpy as np
import pandas as pd

from marg.fraud.data import LABEL, time_split
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
