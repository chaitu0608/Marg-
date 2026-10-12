from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marg.fraud.data import FEATURES, LABEL, load_creditcard, time_split
from marg.fraud.thresholds import evaluate_cost_policy, select_cost_thresholds

ROOT = Path(__file__).resolve().parents[1]
REVIEW_CAPS = (0.0, 0.0025, 0.005, 0.01, 0.02)
REVIEW_COSTS = (0.0, 0.5, 1.0, 5.0)


def evaluate_scenarios(validation: pd.DataFrame, test: pd.DataFrame,
                       validation_scores: np.ndarray, test_scores: np.ndarray) -> list[dict]:
    scenarios = []
    for cap in REVIEW_CAPS:
        for review_cost in REVIEW_COSTS:
            thresholds = select_cost_thresholds(
                validation[LABEL].to_numpy(np.int8), validation_scores,
                validation["Amount"].to_numpy(np.float64), cap, review_cost, grid_size=101)
            result = evaluate_cost_policy(
                test[LABEL].to_numpy(np.int8), test_scores,
                test["Amount"].to_numpy(np.float64), thresholds.review, thresholds.block,
                review_cost)
            scenarios.append({
                "max_review_rate": cap,
                "review_cost_amount_units_per_case": review_cost,
                "review_threshold": thresholds.review,
                "block_threshold": thresholds.block,
                "validation_cost_amount_units": thresholds.validation_cost_amount_units,
                "newer_test_period": result,
            })
    return scenarios


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    args = parser.parse_args()
    frame = load_creditcard(args.data)
    _, validation, test = time_split(frame)
    model = joblib.load(ROOT / "models" / "fraud_model.joblib")
    validation_scores = model.predict_proba(validation[FEATURES].to_numpy(np.float32))[:, 1]
    test_scores = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]
    scenarios = evaluate_scenarios(validation, test, validation_scores, test_scores)
    result = {
        "dataset_path": str(Path(args.data)),
        "scenario_note": "Amount is a proxy only; unit, currency, actual fraud loss, and review benefit are unknown. Each threshold pair is selected on validation data and evaluated on the later test period.",
        "review_caps": list(REVIEW_CAPS),
        "review_cost_amount_units_per_case": list(REVIEW_COSTS),
        "scenarios": scenarios,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "fraud_sensitivity.json").write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cost in REVIEW_COSTS:
        rows = [row for row in scenarios if row["review_cost_amount_units_per_case"] == cost]
        ax.plot([row["max_review_rate"] * 100 for row in rows],
                [row["newer_test_period"]["total_cost_amount_units"] for row in rows],
                marker="o", label=f"Review cost={cost:g} Amount units")
    ax.set(title="Fraud policy sensitivity scenarios (later test period)",
           xlabel="Maximum validation review capacity (%)",
           ylabel="Cost proxy (dataset Amount units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fraud_sensitivity.png", dpi=140)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
