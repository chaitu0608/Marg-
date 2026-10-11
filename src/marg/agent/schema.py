from typing import Literal

from pydantic import BaseModel, Field


class CaseReport(BaseModel):
    summary: str
    risk_factors: list[str]
    confidence: float = Field(ge=0, le=1)
    recommended_action: Literal["allow", "review", "block"]
