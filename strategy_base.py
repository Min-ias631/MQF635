"""Strategy interface definitions."""

from __future__ import annotations

from typing import Protocol

from book import Book
from event import MarketEvent, SignalEvent


class Strategy(Protocol):
    """策略只能从行情和只读 book 产出 SignalEvent。"""

    def on_market(self, event: MarketEvent, book: Book) -> list[SignalEvent]:
        ...


__all__ = ["Strategy"]
