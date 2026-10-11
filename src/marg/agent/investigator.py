from __future__ import annotations

import json
import logging

from marg.agent.llm import LLM
from marg.agent.schema import CaseReport

LOGGER = logging.getLogger(__name__)


class Investigator:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def report(self, transaction: dict, score: float,
               shap_factors: list[str]) -> CaseReport:
        allowed = set(shap_factors)
        prompt = json.dumps({"transaction": transaction, "score": score,
                             "allowed_risk_factors": shap_factors})
        LOGGER.info("investigator input: %s", prompt)
        for _ in range(2):
            try:
                value = CaseReport.model_validate_json(self.llm.complete(prompt))
                # The model may only repeat actual factors supplied by the explainer.
                if not set(value.risk_factors).issubset(allowed):
                    raise ValueError("risk_factors must come from the supplied SHAP list")
                LOGGER.info("investigator output: %s", value.model_dump_json())
                return value
            except ValueError as exc:
                LOGGER.warning("invalid investigator response: %s", exc)
        fallback = CaseReport(summary="Automated report unavailable; analyst review required.",
                              risk_factors=list(shap_factors), confidence=0.0,
                              recommended_action="review")
        LOGGER.info("investigator fallback: %s", fallback.model_dump_json())
        return fallback
