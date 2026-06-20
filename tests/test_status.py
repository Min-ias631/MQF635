"""StatusStore tests."""

from __future__ import annotations

import time
import unittest

from book import Book
from event import BookTickEvent, PriceLevel, SignalAction, SignalEvent
from portfolio import Portfolio
from risk import RiskManager
from runner import RunnerStats
from status import StatusStore


class StatusStoreTests(unittest.TestCase):
    def test_status_store_tracks_books_signals_and_runtime(self) -> None:
        store = StatusStore()
        book = Book(symbols=("BTCUSDT",))
        portfolio = Portfolio()
        risk = RiskManager()
        market = BookTickEvent.of(
            time.time_ns(),
            "BTCUSDT",
            [PriceLevel(100.0, 2.0), PriceLevel(99.9, 1.0), PriceLevel(99.8, 1.0)],
            [PriceLevel(100.1, 1.0), PriceLevel(100.2, 1.0), PriceLevel(100.3, 1.0)],
        )
        signal = SignalEvent(
            ts_ns=market.ts_ns,
            symbol="BTCUSDT",
            strategy_id="test",
            action=SignalAction.LONG,
        )

        book.update(market)
        store.on_event(market)
        store.on_event(signal)
        store.update_runtime(book=book, portfolio=portfolio, risk=risk, stats=RunnerStats(market_events=1))
        snap = store.snapshot()

        self.assertEqual(snap["risk_state"], risk.state)
        self.assertIn("BTCUSDT", snap["books"])
        self.assertIn("BTCUSDT", snap["signals"])
        self.assertEqual(snap["runner_stats"]["market_events"], 1)
        self.assertGreaterEqual(len(snap["recent_events"]), 2)

    def test_status_store_records_history_series(self) -> None:
        store = StatusStore(history_interval_ns=1)
        book = Book(symbols=("BTCUSDT",))
        portfolio = Portfolio()
        risk = RiskManager()

        market1 = BookTickEvent.of(
            100,
            "BTCUSDT",
            [PriceLevel(100.0, 2.0)],
            [PriceLevel(100.2, 1.0)],
        )
        market2 = BookTickEvent.of(
            200,
            "BTCUSDT",
            [PriceLevel(101.0, 2.0)],
            [PriceLevel(101.2, 1.0)],
        )

        book.update(market1)
        store.update_runtime(book=book, portfolio=portfolio, risk=risk, stats=RunnerStats(market_events=1))
        book.update(market2)
        store.update_runtime(book=book, portfolio=portfolio, risk=risk, stats=RunnerStats(market_events=2))

        history = store.history()
        self.assertGreaterEqual(len(history["pnl"]), 2)
        self.assertIn("BTCUSDT", history["mid_by_symbol"])
        self.assertGreaterEqual(len(history["mid_by_symbol"]["BTCUSDT"]), 2)


if __name__ == "__main__":
    unittest.main()
