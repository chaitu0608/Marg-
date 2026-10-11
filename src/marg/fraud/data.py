from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL = "Class"
TIME = "Time"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_creditcard(path: str | Path = "data/creditcard.csv") -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Put the real Kaggle Credit Card Fraud Detection CSV at {source}")
    frame = pd.read_csv(source)
    expected = {TIME, LABEL, "Amount", *[f"V{i}" for i in range(1, 29)]}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Unexpected dataset columns; missing {sorted(expected - set(frame.columns))}")
    if len(frame) < 200_000:
        raise ValueError(f"Dataset looks incomplete ({len(frame):,} rows); expected about 284,807")
    return frame.sort_values(TIME, kind="stable").reset_index(drop=True)


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(TIME, kind="stable").reset_index(drop=True)
    n = len(ordered)
    # Use row quantiles to choose boundaries, then keep equal timestamps in one split.
    first_time = ordered.loc[int(n * 0.70), TIME]
    second_time = ordered.loc[int(n * 0.85), TIME]
    train = ordered[ordered[TIME] <= first_time]
    validation = ordered[(ordered[TIME] > first_time) & (ordered[TIME] <= second_time)]
    test = ordered[ordered[TIME] > second_time]
    return train.copy(), validation.copy(), test.copy()
