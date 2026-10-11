import json
import os
from typing import Protocol
from urllib.request import Request, urlopen


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockLLM:
    def __init__(self, response: str | None = None) -> None:
        self.response = response
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.response is not None:
            return self.response
        return ('{"summary":"Flagged for analyst review.","risk_factors":[], '
                '"confidence":0.5,"recommended_action":"review"}')


class OpenAILLM:
    """Minimal provider adapter; API credentials are read only from the environment."""

    def __init__(self, api_key: str | None = None,
                 model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY to use OpenAILLM")
        self.model = model

    def complete(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "messages": [
            {"role": "system", "content": "Return only a JSON case report."},
            {"role": "user", "content": prompt},
        ]}).encode()
        request = Request("https://api.openai.com/v1/chat/completions", data=body,
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        return str(payload["choices"][0]["message"]["content"])


def default_llm() -> LLM:
    """Use a deterministic local response unless an API credential is configured."""
    return OpenAILLM() if os.environ.get("OPENAI_API_KEY") else MockLLM()
