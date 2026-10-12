from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from marg.fraud.data import FEATURES, load_creditcard, time_split

ROOT = Path(__file__).resolve().parents[1]


def generate_examples(data_path: str, model_path: str, thresholds_path: str) -> dict:
    frame = load_creditcard(data_path)
    _, _, test = time_split(frame)
    model = joblib.load(model_path)
    matrix = test[FEATURES].to_numpy(np.float32)
    scores = model.predict_proba(matrix)[:, 1]
    thresholds = json.loads(Path(thresholds_path).read_text())
    decisions = np.where(scores >= thresholds["block"], "block",
                         np.where(scores >= thresholds["review"], "review", "allow"))

    examples = []
    for decision in ("allow", "review", "block"):
        candidates = np.flatnonzero(decisions == decision)
        if not len(candidates):
            continue
        # Pick a representative from the middle of each model-decision band.
        target = float(np.median(scores[candidates]))
        row_index = int(candidates[np.argmin(np.abs(scores[candidates] - target))])
        examples.append({
            "name": f"{decision.title()} example",
            "decision": decision,
            "score": float(scores[row_index]),
            "features": [float(value) for value in matrix[row_index]],
        })

    return {
        "source": str(data_path),
        "split": "later time-ordered test period; anonymized features; labels omitted",
        "feature_count": len(FEATURES),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--thresholds", default="models/thresholds.json")
    parser.add_argument("--output", default="artifacts/demo-samples.json")
    args = parser.parse_args()

    samples = generate_examples(args.data, args.model, args.thresholds)
    if not samples["examples"]:
        raise RuntimeError("no scored examples were available for the current thresholds")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(samples, indent=2) + "\n")
    print(f"Wrote {len(samples['examples'])} anonymized demo examples to {output}")


if __name__ == "__main__":
    main()
