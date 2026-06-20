"""runner.py 链路单测。"""

from __future__ import annotations

import queue
import time
import unittest

from book import Book
from broker import PaperBroker, SimBroker
from event import BookTickEvent, PriceLevel, SignalAction, SignalEvent
from portfolio import Portfolio
from risk import RiskManager
from runner import Runner
from status import StatusStore
from microstructure_strategy import MicrostructureImbalanceStrategy


class LongOnceStrategy:
    def __init__(self) -> None:
        self._sent = False

    def on_market(self, event, book: Book) -> list[SignalEvent]:
        if self._sent:
            return []
        self._sent = True
        return [
            SignalEvent(
                ts_ns=time.time_ns(),
                symbol="BTCUSDT",
                strategy_id="long-once",
                action=SignalAction.LONG,
                strength=1.0,
            )
        ]


class LossPortfolio(Portfolio):
    def daily_pnl(self) -> float:
        return -100.0


def _market_event() -> BookTickEvent:
    return BookTickEvent.of(
        time.time_ns(),
        "BTCUSDT",
        [PriceLevel(100.0, 1.0)],
        [PriceLevel(101.0, 1.0)],
    )


def _imbalanced_market_event() -> BookTickEvent:
    return BookTickEvent.of(
        time.time_ns(),
        "BTCUSDT",
        [
            PriceLevel(100.0, 10.0),
            PriceLevel(99.99, 10.0),
            PriceLevel(99.98, 10.0),
        ],
        [
            PriceLevel(100.01, 1.0),
            PriceLevel(100.02, 1.0),
            PriceLevel(100.03, 1.0),
        ],
    )


class RunnerTests(unittest.TestCase):
    def test_runner_market_to_fill_updates_portfolio(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        q.put(_market_event())

        portfolio = Portfolio()
        runner = Runner(
            event_queue=q,
            book=Book(symbols=("BTCUSDT",)),
            strategy=LongOnceStrategy(),
            risk=RiskManager(),
            broker=SimBroker(),
            portfolio=portfolio,
        )

        processed = runner.drain()

        self.assertEqual(processed, 1)
        self.assertAlmostEqual(portfolio.position_qty("BTCUSDT"), 0.01)
        view = portfolio.get_position("BTCUSDT")
        self.assertIsNotNone(view)
        assert view is not None
        self.assertAlmostEqual(view.avg_price, 101.0)
        self.assertEqual(runner.stats.market_events, 1)
        self.assertEqual(runner.stats.signals, 1)
        self.assertEqual(runner.stats.orders, 1)
        self.assertEqual(runner.stats.fills, 1)

    def test_runner_with_paper_broker_does_not_change_portfolio(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        q.put(_market_event())

        portfolio = Portfolio()
        runner = Runner(
            event_queue=q,
            book=Book(symbols=("BTCUSDT",)),
            strategy=LongOnceStrategy(),
            risk=RiskManager(),
            broker=PaperBroker(),
            portfolio=portfolio,
        )

        runner.drain()

        self.assertAlmostEqual(portfolio.position_qty("BTCUSDT"), 0.0)
        self.assertEqual(runner.stats.orders, 1)
        self.assertEqual(runner.stats.fills, 0)

    def test_runner_does_not_order_when_risk_rejects(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        q.put(_market_event())

        portfolio = LossPortfolio()
        runner = Runner(
            event_queue=q,
            book=Book(symbols=("BTCUSDT",)),
            strategy=LongOnceStrategy(),
            risk=RiskManager(),
            broker=SimBroker(),
            portfolio=portfolio,
        )

        runner.drain()

        self.assertEqual(runner.stats.signals, 1)
        self.assertEqual(runner.stats.orders, 0)
        self.assertEqual(runner.stats.risk_events, 1)

    def test_runner_with_real_microstructure_strategy(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        q.put(_imbalanced_market_event())

        portfolio = Portfolio()
        runner = Runner(
            event_queue=q,
            book=Book(symbols=("BTCUSDT",)),
            strategy=MicrostructureImbalanceStrategy(cooldown_ns=0),
            risk=RiskManager(),
            broker=SimBroker(),
            portfolio=portfolio,
        )

        runner.drain()

        self.assertEqual(runner.stats.signals, 1)
        self.assertEqual(runner.stats.orders, 1)
        self.assertEqual(runner.stats.fills, 1)
        self.assertAlmostEqual(portfolio.position_qty("BTCUSDT"), 0.01)

    def test_runner_updates_status_store(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        q.put(_imbalanced_market_event())
        status_store = StatusStore()

        runner = Runner(
            event_queue=q,
            book=Book(symbols=("BTCUSDT",)),
            strategy=MicrostructureImbalanceStrategy(cooldown_ns=0),
            risk=RiskManager(),
            broker=PaperBroker(),
            portfolio=Portfolio(),
            status_store=status_store,
        )

        runner.drain()
        snap = status_store.snapshot()

        self.assertIn("BTCUSDT", snap["books"])
        self.assertIn("BTCUSDT", snap["signals"])
        self.assertEqual(snap["runner_stats"]["orders"], 1)
        self.assertEqual(snap["runner_stats"]["fills"], 0)


if __name__ == "__main__":
    unittest.main()
