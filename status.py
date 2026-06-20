"""
status.py —— UI/API 状态快照存储

StatusStore 是观察层，不参与交易决策。runner 将事件和最新模块状态写入这里，
FastAPI/Dash 只读 snapshot，避免 UI 直接侵入交易主链路。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from book import Book
from event import (
    FillEvent,
    MarketEvent,
    OrderAckEvent,
    OrderEvent,
    OrderRejectEvent,
    RiskEvent,
    SignalEvent,
)
from portfolio import Portfolio
from risk import RiskManager


class StatusStore:
    def __init__(
        self,
        *,
        max_recent_events: int = 200,
        max_history_points: int = 300,
        history_interval_ns: int = 1_000_000_000,
    ) -> None:
        self._lock = threading.RLock()
        self._max_history_points = max(10, int(max_history_points))
        self._history_interval_ns = max(1, int(history_interval_ns))
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=max_recent_events)
        self._latest_signals: dict[str, dict[str, Any]] = {}
        self._latest_orders: dict[str, dict[str, Any]] = {}
        self._latest_books: dict[str, dict[str, Any]] = {}
        self._latest_positions: dict[str, dict[str, Any]] = {}
        self._risk_state: str = "UNKNOWN"
        self._runner_stats: dict[str, Any] = {}
        self._daily_pnl: float = 0.0
        self._pnl_history: deque[dict[str, Any]] = deque(maxlen=self._max_history_points)
        self._mid_history: dict[str, deque[dict[str, Any]]] = {}
        self._last_pnl_history_ts_ns: int = 0
        self._last_mid_history_ts_ns: dict[str, int] = {}

    def on_event(self, event: object) -> None:
        item = self._event_to_dict(event)
        with self._lock:
            self._recent_events.append(item)
            if isinstance(event, SignalEvent):
                self._latest_signals[event.symbol.upper()] = item
            elif isinstance(event, OrderEvent):
                self._latest_orders[event.symbol.upper()] = item
            elif isinstance(event, (OrderAckEvent, FillEvent, OrderRejectEvent)):
                symbol = getattr(event, "symbol", None)
                if symbol:
                    self._latest_orders[str(symbol).upper()] = item
            elif isinstance(event, RiskEvent):
                self._risk_state = event.action.value

    def update_runtime(
        self,
        *,
        book: Book,
        portfolio: Portfolio,
        risk: RiskManager,
        stats: object,
    ) -> None:
        with self._lock:
            self._risk_state = risk.state
            self._runner_stats = self._jsonable(stats)
            self._daily_pnl = portfolio.daily_pnl()
            self._latest_books = {}
            latest_ts_ns = 0
            for sym in book.symbols():
                snap = self._book_snapshot(book, sym)
                if snap is None:
                    continue
                self._latest_books[sym] = snap
                latest_ts_ns = max(latest_ts_ns, int(snap["ts_ns"]))
            self._latest_positions = {
                sym: pos
                for sym in portfolio.symbols()
                if (pos := self._position_snapshot(portfolio, sym)) is not None
            }
            self._record_history(latest_ts_ns or time.time_ns())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = self._base_snapshot()
            payload["recent_events"] = list(self._recent_events)
            return payload

    def dashboard_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._base_snapshot()

    def books(self) -> dict[str, dict[str, Any]]:
        return self.snapshot()["books"]

    def positions(self) -> dict[str, dict[str, Any]]:
        return self.snapshot()["positions"]

    def recent_events(self) -> list[dict[str, Any]]:
        return self.snapshot()["recent_events"]

    def history(self) -> dict[str, Any]:
        with self._lock:
            return self._history_snapshot()

    def _base_snapshot(self) -> dict[str, Any]:
        return {
            "risk_state": self._risk_state,
            "daily_pnl": self._daily_pnl,
            "runner_stats": dict(self._runner_stats),
            "books": dict(self._latest_books),
            "positions": dict(self._latest_positions),
            "signals": dict(self._latest_signals),
            "orders": dict(self._latest_orders),
            "history": self._history_snapshot(),
        }

    def _history_snapshot(self) -> dict[str, Any]:
        return {
            "pnl": list(self._pnl_history),
            "mid_by_symbol": {sym: list(points) for sym, points in self._mid_history.items()},
        }

    def _record_history(self, ts_ns: int) -> None:
        if self._should_append(self._last_pnl_history_ts_ns, ts_ns):
            self._pnl_history.append({"ts_ns": ts_ns, "value": self._daily_pnl})
            self._last_pnl_history_ts_ns = ts_ns

        for sym, snap in self._latest_books.items():
            mid = snap.get("mid")
            snap_ts_ns = int(snap.get("ts_ns") or ts_ns)
            if mid is None:
                continue
            last_ts_ns = self._last_mid_history_ts_ns.get(sym, 0)
            if not self._should_append(last_ts_ns, snap_ts_ns):
                continue
            points = self._mid_history.setdefault(sym, deque(maxlen=self._max_history_points))
            points.append({"ts_ns": snap_ts_ns, "value": float(mid)})
            self._last_mid_history_ts_ns[sym] = snap_ts_ns

    def _should_append(self, last_ts_ns: int, current_ts_ns: int) -> bool:
        if last_ts_ns <= 0:
            return True
        return current_ts_ns - last_ts_ns >= self._history_interval_ns

    @staticmethod
    def _book_snapshot(book: Book, symbol: str) -> dict[str, Any] | None:
        snap = book.snapshot(symbol)
        if snap is None:
            return None
        imbalance = book.imbalance(symbol, levels=3)
        return {
            "symbol": snap.symbol,
            "ts_ns": snap.ts_ns,
            "best_bid": snap.best_bid,
            "best_bid_qty": snap.best_bid_qty,
            "best_ask": snap.best_ask,
            "best_ask_qty": snap.best_ask_qty,
            "mid": snap.mid,
            "spread": snap.spread,
            "last_price": snap.last_price,
            "last_qty": snap.last_qty,
            "imbalance_3": imbalance,
            "last_update_id": snap.last_update_id,
            "need_resync": snap.need_resync,
            "is_valid": snap.is_valid,
        }

    @staticmethod
    def _position_snapshot(portfolio: Portfolio, symbol: str) -> dict[str, Any] | None:
        view = portfolio.get_position(symbol)
        if view is None:
            return None
        return StatusStore._jsonable(view)

    @staticmethod
    def _event_to_dict(event: object) -> dict[str, Any]:
        return {
            "event_type": type(event).__name__,
            "payload": StatusStore._jsonable(event),
        }

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: StatusStore._jsonable(v) for k, v in asdict(obj).items()}
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, tuple):
            return [StatusStore._jsonable(x) for x in obj]
        if isinstance(obj, list):
            return [StatusStore._jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): StatusStore._jsonable(v) for k, v in obj.items()}
        return obj


__all__ = ["StatusStore"]
