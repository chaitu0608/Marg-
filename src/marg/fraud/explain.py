from __future__ import annotations

import joblib
import numpy as np
import shap

from marg.fraud.data import FEATURES


class Explainer:
    def __init__(self, model_path: str = "models/fraud_model.joblib") -> None:
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def top_features(self, row: np.ndarray, top_k: int = 5) -> list[dict[str, float | str]]:
        values = np.asarray(self.explainer.shap_values(row.reshape(1, -1)))
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        indices = np.argsort(np.abs(values))[::-1][:top_k]
        names = FEATURES
        return [{"feature": str(names[i]), "contribution": float(values[i])} for i in indices]
