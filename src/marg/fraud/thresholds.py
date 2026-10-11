from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CostThresholds:
    review: float
    block: float
    max_review_rate: float
    review_cost_amount_units: float
    candidate_count: int
    validation_cost_amount_units: float
    validation_review_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_cost_policy(
    labels: NDArray[np.integer],
    scores: NDArray[np.floating],
    amounts: NDArray[np.floating],
    review_threshold: float,
    block_threshold: float,
    review_cost_amount_units: float,
) -> dict[str, float | int]:
    """Score allow/review/block decisions using transaction Amount as a loss proxy."""
    reviewed = (scores >= review_threshold) & (scores < block_threshold)
    blocked = scores >= block_threshold
    missed_fraud = (labels == 1) & (scores < review_threshold)
    false_blocks = (labels == 0) & blocked
    missed_amount = float(amounts[missed_fraud].sum())
    false_block_amount = float(amounts[false_blocks].sum())
    review_cost = int(reviewed.sum()) * review_cost_amount_units
    return {
        "total_cost_amount_units": missed_amount + false_block_amount + review_cost,
        "missed_fraud_amount_units": missed_amount,
        "false_block_amount_units": false_block_amount,
        "review_cost_amount_units": review_cost,
        "review_count": int(reviewed.sum()),
        "review_rate": float(reviewed.mean()) if len(reviewed) else 0.0,
        "false_block_count": int(false_blocks.sum()),
        "missed_fraud_count": int(missed_fraud.sum()),
    }


def select_cost_thresholds(
    labels: NDArray[np.integer],
    scores: NDArray[np.floating],
    amounts: NDArray[np.floating],
    max_review_rate: float = 0.01,
    review_cost_amount_units: float = 1.0,
    grid_size: int = 201,
) -> CostThresholds:
    """Choose validation thresholds by minimum amount-weighted cost under review capacity."""
    if not 0.0 <= max_review_rate <= 1.0:
        raise ValueError("max_review_rate must be between zero and one")
    if not np.isfinite(review_cost_amount_units) or review_cost_amount_units < 0.0:
        raise ValueError("review_cost_amount_units must be finite and nonnegative")
    if grid_size < 2 or len(labels) == 0 or len(scores) != len(labels) or len(amounts) != len(labels):
        raise ValueError("threshold selection requires aligned nonempty arrays and grid_size >= 2")
    if (not np.isfinite(scores).all() or not np.isfinite(amounts).all()
            or np.any((scores < 0.0) | (scores > 1.0)) or np.any(amounts < 0.0)
            or np.any((labels != 0) & (labels != 1))):
        raise ValueError("scores, labels, or amounts contain invalid values")

    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, grid_size)))
    # Include a finite threshold above every observed score so all-allow/all-review policies exist.
    candidates = np.append(candidates, np.nextafter(float(scores.max()), np.inf))
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    fraud_amount = np.where(labels[order] == 1, amounts[order], 0.0)
    legitimate_amount = np.where(labels[order] == 0, amounts[order], 0.0)
    fraud_prefix = np.concatenate(([0.0], np.cumsum(fraud_amount)))
    legitimate_prefix = np.concatenate(([0.0], np.cumsum(legitimate_amount)))
    positions = np.searchsorted(ordered_scores, candidates, side="left")
    max_reviews = int(np.floor(max_review_rate * len(labels)))

    best: tuple[float, int, float, float] | None = None
    for review_index, review_threshold in enumerate(candidates):
        review_position = int(positions[review_index])
        missed_amount = float(fraud_prefix[review_position])
        for block_index in range(review_index, len(candidates)):
            block_threshold = float(candidates[block_index])
            block_position = int(positions[block_index])
            review_count = block_position - review_position
            if review_count > max_reviews:
                continue
            false_block_amount = float(legitimate_prefix[-1] - legitimate_prefix[block_position])
            cost = (missed_amount + false_block_amount
                    + review_count * review_cost_amount_units)
            proposal = (cost, review_count, float(review_threshold), block_threshold)
            if best is None or proposal < best:
                best = proposal

    if best is None:
        raise RuntimeError("no thresholds satisfy the configured review capacity")
    cost, review_count, review_threshold, block_threshold = best
    return CostThresholds(
        review=review_threshold,
        block=block_threshold,
        max_review_rate=max_review_rate,
        review_cost_amount_units=review_cost_amount_units,
        candidate_count=len(candidates),
        validation_cost_amount_units=cost,
        validation_review_count=review_count,
    )
