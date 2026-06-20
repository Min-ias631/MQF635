"""Tests for the funding wiring + real-trade replay fixes.

Covers:
- CSVFeed merging book/trade/funding rows in timestamp order (real TradeEvents).
- Runner routing FundingEvent into Book.funding_rate.
- The crypto_pca adapter reading real funding from the book (no longer hardcoded 0).
- PCAFlowAlpha sign-pinning of rolling PCs.
"""

from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path

import numpy as np

from book import Book
from broker import SimBroker
from event import (
    BookTickEvent,
    FundingEvent,
    PriceLevel,
    TradeEvent,
)
from feed import CSVFeed
from portfolio import Portfolio
from risk import RiskManager
from runner import Runner
from crypto_pca import CryptoPcaStrategy, _canonical_classes


def _book_event(ts_ns: int, symbol: str, bid_qty: float, ask_qty: float) -> BookTickEvent:
    base = 100.0 + (sum(ord(c) for c in symbol) % 20)
    return BookTickEvent.of(
        ts_ns,
        symbol,
        [PriceLevel(base, bid_qty), PriceLevel(base - 0.01, bid_qty * 0.8),
         PriceLevel(base - 0.02, bid_qty * 0.6)],
        [PriceLevel(base + 0.01, ask_qty), PriceLevel(base + 0.02, ask_qty * 0.8),
         PriceLevel(base + 0.03, ask_qty * 0.6)],
    )


class CsvFeedReplayTests(unittest.TestCase):
    def test_csv_feed_merges_book_trade_funding_in_time_order(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book_csv = Path(d) / "book.csv"
            trade_csv = Path(d) / "trade.csv"
            funding_csv = Path(d) / "funding.csv"
            book_csv.write_text(
                "ts_ns,symbol,bid,ask\n1,BTCUSDT,100,101\n3,BTCUSDT,100,101\n"
            )
            trade_csv.write_text(
                "ts_ns,symbol,price,qty,is_buyer_maker\n"
                "2,BTCUSDT,100.5,1.0,false\n4,BTCUSDT,100.5,2.0,true\n"
            )
            funding_csv.write_text("ts_ns,symbol,funding_rate\n5,BTCUSDT,0.0001\n")

            q: queue.Queue[object] = queue.Queue()
            feed = CSVFeed(
                q,
                book_csv=str(book_csv),
                trade_csv=str(trade_csv),
                funding_csv=str(funding_csv),
                background=False,
            )
            feed.start()

            events = []
            while not q.empty():
                events.append(q.get_nowait())

            self.assertEqual(
                [type(e).__name__ for e in events],
                ["BookTickEvent", "TradeEvent", "BookTickEvent", "TradeEvent", "FundingEvent"],
            )
            # Real taker flow carried through (not a book-imbalance proxy).
            self.assertFalse(events[1].is_buyer_maker)
            self.assertTrue(events[3].is_buyer_maker)
            self.assertAlmostEqual(events[4].funding_rate, 0.0001)


class FundingRoutingTests(unittest.TestCase):
    def test_runner_routes_funding_event_into_book(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        q.put(FundingEvent(
            ts_ns=1, account_id="default", symbol="ETHUSDT",
            funding_rate=0.0005, funding_amount=0.0,
        ))
        book = Book(symbols=("ETHUSDT",))
        runner = Runner(
            event_queue=q,
            book=book,
            strategy=CryptoPcaStrategy(),  # never called (no market event)
            risk=RiskManager(),
            broker=SimBroker(),
            portfolio=Portfolio(),
        )

        runner.drain()

        self.assertAlmostEqual(book.funding_rate("ETHUSDT"), 0.0005)
        # Funding events are not signals/orders.
        self.assertEqual(runner.stats.signals, 0)
        self.assertEqual(runner.stats.orders, 0)

    def test_adapter_reads_real_funding_from_book(self) -> None:
        symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
        trade_symbols = ("ETHUSDT", "BNBUSDT", "SOLUSDT")
        book = Book(symbols=symbols)
        strategy = CryptoPcaStrategy(
            pca_symbols=symbols, trade_symbols=trade_symbols,
            pca_window=3, beta_window=2, cooldown_ns=0,
        )
        # Non-uniform funding so the cross-sectional funding signal is non-zero.
        book.set_funding("ETHUSDT", 0.0010)
        book.set_funding("BNBUSDT", -0.0020)
        book.set_funding("SOLUSDT", 0.0006)

        ts = 1_700_000_000_000_000_000
        for minute in range(8):
            for idx, symbol in enumerate(symbols):
                ev = _book_event(ts + minute * 60_000_000_000 + idx, symbol,
                                 10.0 + minute + idx, 1.0 + idx)
                book.update(ev)
                strategy.on_market(ev, book)

        self.assertTrue(strategy.latest_signals)
        funding_signals = [s["funding_signal"] for s in strategy.latest_signals.values()]
        self.assertTrue(any(abs(f) > 1e-9 for f in funding_signals),
                        "funding term should be live now, not silently zero")


class SignPinningTests(unittest.TestCase):
    def test_sign_align_flips_to_match_previous_window(self) -> None:
        PCAFlowAlpha = _canonical_classes()["PCAFlowAlpha"]
        comps = np.array([[1.0, 0.2], [0.1, -1.0]])
        prev = np.array([[-1.0, -0.2], [0.1, -1.0]])  # PC1 opposite, PC2 aligned

        aligned = PCAFlowAlpha._sign_align(comps, prev)

        # Both components should now point the same way as the previous window.
        self.assertGreaterEqual(float(np.dot(aligned[0], prev[0])), 0.0)
        self.assertGreaterEqual(float(np.dot(aligned[1], prev[1])), 0.0)
        # PC1 was flipped; PC2 left unchanged.
        np.testing.assert_allclose(aligned[0], -comps[0])
        np.testing.assert_allclose(aligned[1], comps[1])

    def test_sign_align_first_window_anchor_is_deterministic(self) -> None:
        PCAFlowAlpha = _canonical_classes()["PCAFlowAlpha"]
        comps = np.array([[-0.9, 0.1], [0.2, -0.8]])  # largest-mag loading negative

        aligned = PCAFlowAlpha._sign_align(comps, None)

        # Largest-magnitude loading of each component is made positive.
        self.assertGreater(aligned[0][int(np.argmax(np.abs(aligned[0])))], 0.0)
        self.assertGreater(aligned[1][int(np.argmax(np.abs(aligned[1])))], 0.0)


if __name__ == "__main__":
    unittest.main()
