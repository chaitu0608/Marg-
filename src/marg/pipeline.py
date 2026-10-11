from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from marg.fraud.data import FEATURES, LABEL, load_creditcard, time_split
from marg.routing.thompson import SlidingWindowThompsonRouter
from marg.sim.constants import BANKS, CONTEXTS, GATEWAYS, METHODS
from marg.sim.traffic import Transaction
from marg.sim.world import make_world

ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(path: str = "data/creditcard.csv", limit: int | None = None,
                 seed: int = 37) -> dict:
    frame = load_creditcard(path)
    _, _, test = time_split(frame)
    if limit is not None:
        test = test.iloc[:limit]
    model = joblib.load(ROOT / "models" / "fraud_model.joblib")
    thresholds = json.loads((ROOT / "models" / "thresholds.json").read_text())
    scores = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]
    decisions = np.where(scores >= thresholds["block"], "block",
                         np.where(scores >= thresholds["review"], "review", "allow"))
    rng = np.random.default_rng(seed)
    world = make_world(seed, len(test))
    router = SlidingWindowThompsonRouter(seed=seed)
    routed = route_success = blocked = blocked_fraud = false_blocks = 0
    actual = test[LABEL].to_numpy(np.int8)
    review_queue: list[int] = []
    for i, decision in enumerate(decisions):
        blocked += decision == "block"
        blocked_fraud += decision == "block" and actual[i] == 1
        false_blocks += decision == "block" and actual[i] == 0
        if decision == "block":
            continue
        if decision == "review":
            review_queue.append(i)
        bank, method = str(rng.choice(BANKS)), str(rng.choice(METHODS, p=[0.72, 0.21, 0.07]))
        tx = Transaction(i, i, bank, method, float(test.iloc[i]["Amount"]))
        gateway = router.select(tx.context, i)
        gi = GATEWAYS.index(gateway)
        context_index = CONTEXTS.index(tx.context)
        ok = bool(world.uniforms[i, gi] < world.rates[i, gi, context_index])
        router.update(tx.context, gateway, ok, i)
        routed += 1
        route_success += ok
    return {
        "dataset_path": str(Path(path)), "test_transactions": len(test),
        "blocked_fraud_rate": float(blocked_fraud / int(actual.sum())) if actual.sum() else 0.0,
        "false_block_rate": float(false_blocks / int((actual == 0).sum())),
        "block_fraction": float(blocked / len(test)),
        "review_queue_count": len(review_queue),
        "routing_success_rate": float(route_success / routed) if routed else None,
        "routed_transactions": routed,
    }


def main() -> None:
    result = run_pipeline()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "pipeline.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
