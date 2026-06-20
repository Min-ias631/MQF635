"""microstructure strategy tests."""

from __future__ import annotations

import unittest

from book import Book
from event import BookTickEvent, PriceLevel, SignalAction
from microstructure_strategy import MicrostructureImbalanceStrategy


def _event(
    *,
    ts_ns: int,
    bid_qty: float,
    ask_qty: float,
    bid: float = 100.0,
    ask: float = 100.01,
    symbol: str = "BTCUSDT",
) -> BookTickEvent:
    return BookTickEvent.of(
        ts_ns,
        symbol,
        [
            PriceLevel(bid, bid_qty),
            PriceLevel(bid - 0.01, bid_qty),
            PriceLevel(bid - 0.02, bid_qty),
        ],
        [
            PriceLevel(ask, ask_qty),
            PriceLevel(ask + 0.01, ask_qty),
            PriceLevel(ask + 0.02, ask_qty),
        ],
    )


class MicrostructureStrategyTests(unittest.TestCase):
    def test_positive_imbalance_emits_long(self) -> None:
        book = Book(symbols=("BTCUSDT",))
        strategy = MicrostructureImbalanceStrategy(cooldown_ns=0)
        ev = _event(ts_ns=1, bid_qty=10.0, ask_qty=1.0)

        book.update(ev)
        signals = strategy.on_market(ev, book)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, SignalAction.LONG)
        self.assertEqual(signals[0].symbol, "BTCUSDT")

    def test_negative_imbalance_emits_short(self) -> None:
        book = Book(symbols=("ETHUSDT",))
        strategy = MicrostructureImbalanceStrategy(cooldown_ns=0)
        ev = _event(ts_ns=1, bid_qty=1.0, ask_qty=10.0, symbol="ETHUSDT")

        book.update(ev)
        signals = strategy.on_market(ev, book)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, SignalAction.SHORT)

    def test_exit_zone_emits_flat_after_position_signal(self) -> None:
        book = Book(symbols=("BTCUSDT",))
        strategy = MicrostructureImbalanceStrategy(cooldown_ns=0)
        long_ev = _event(ts_ns=1, bid_qty=10.0, ask_qty=1.0)
        flat_ev = _event(ts_ns=2, bid_qty=5.0, ask_qty=5.0)

        book.update(long_ev)
        self.assertEqual(strategy.on_market(long_ev, book)[0].action, SignalAction.LONG)
        book.update(flat_ev)
        signals = strategy.on_market(flat_ev, book)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, SignalAction.FLAT)

    def test_repeated_same_action_is_suppressed(self) -> None:
        book = Book(symbols=("BTCUSDT",))
        strategy = MicrostructureImbalanceStrategy(cooldown_ns=0)
        ev1 = _event(ts_ns=1, bid_qty=10.0, ask_qty=1.0)
        ev2 = _event(ts_ns=2, bid_qty=11.0, ask_qty=1.0)

        book.update(ev1)
        self.assertEqual(len(strategy.on_market(ev1, book)), 1)
        book.update(ev2)
        self.assertEqual(strategy.on_market(ev2, book), [])

    def test_cooldown_suppresses_state_change(self) -> None:
        book = Book(symbols=("BTCUSDT",))
        strategy = MicrostructureImbalanceStrategy(cooldown_ns=10)
        long_ev = _event(ts_ns=100, bid_qty=10.0, ask_qty=1.0)
        short_ev = _event(ts_ns=105, bid_qty=1.0, ask_qty=10.0)

        book.update(long_ev)
        self.assertEqual(len(strategy.on_market(long_ev, book)), 1)
        book.update(short_ev)
        self.assertEqual(strategy.on_market(short_ev, book), [])

    def test_wide_spread_is_filtered(self) -> None:
        book = Book(symbols=("BTCUSDT",))
        strategy = MicrostructureImbalanceStrategy(cooldown_ns=0, max_spread_bps=5.0)
        ev = _event(ts_ns=1, bid_qty=10.0, ask_qty=1.0, bid=100.0, ask=101.0)

        book.update(ev)
        self.assertEqual(strategy.on_market(ev, book), [])


if __name__ == "__main__":
    unittest.main()
