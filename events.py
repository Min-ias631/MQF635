"""events.py — canonical market-data event types shared by both surfaces.

Zero business dependencies; only the standard library. The Yumi backtest adapter
and the Streamlit UI adapter both construct these objects and hand them to the
alpha engine, so this is the wire format for the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar with taker buy/sell split (Binance kline semantics)."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float
    taker_sell_volume: float

    @property
    def tfi(self) -> float:
        """Taker-Flow-Imbalance: (buy - sell) / (buy + sell).

        Mirrors data_client.get_klines exactly: denominator is the taker
        buy+sell volume; returns 0.0 when there is no taker flow.
        """
        denom = self.taker_buy_volume + self.taker_sell_volume
        if denom <= 0:
            return 0.0
        return (self.taker_buy_volume - self.taker_sell_volume) / denom


@dataclass(frozen=True)
class BookSnapshot:
    """Top-N order-book snapshot. bids/asks are tuples of (price, qty)."""

    timestamp: datetime
    symbol: str
    bids: Tuple[Tuple[float, float], ...]
    asks: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class FundingRate:
    """Perp funding rate sample for a single symbol."""

    timestamp: datetime
    symbol: str
    rate: float


@dataclass(frozen=True)
class MarketSnapshot:
    """A cross-sectional snapshot of the universe at one timestamp."""

    timestamp: datetime
    bars: Dict[str, Bar]
    books: Dict[str, BookSnapshot]
    funding: Dict[str, FundingRate]
    btc_daily: Optional[Any] = None


__all__ = ["Bar", "BookSnapshot", "FundingRate", "MarketSnapshot"]
