from typing import Protocol


class Router(Protocol):
    name: str

    def select(self, context: tuple[str, str], t: int) -> str: ...
    def update(self, context: tuple[str, str], gateway: str, success: bool, t: int) -> None: ...
