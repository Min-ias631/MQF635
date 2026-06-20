"""BinancePublicFeed parsing tests."""

from __future__ import annotations

import queue
import unittest

from event import BookTickEvent, TradeEvent
from feed import BinancePublicFeed


class BinancePublicFeedTests(unittest.TestCase):
    def test_combined_stream_url_contains_depth_and_trade(self) -> None:
        feed = BinancePublicFeed(queue.Queue(), symbols=("BTCUSDT", "ETHUSDT"), depth_levels=5)
        url = feed._url()  # unit-level check for public stream composition

        self.assertIn("btcusdt@depth5@100ms", url)
        self.assertIn("btcusdt@trade", url)
        self.assertIn("ethusdt@depth5@100ms", url)
        self.assertIn("ethusdt@trade", url)

    def test_parse_partial_depth_uses_stream_symbol(self) -> None:
        feed = BinancePublicFeed(queue.Queue(), symbols=("BTCUSDT",))
        event = feed._parse_data(
            {
                "lastUpdateId": 123,
                "bids": [["100.0", "2.5"]],
                "asks": [["100.1", "1.5"]],
            },
            stream="btcusdt@depth5@100ms",
        )

        self.assertIsInstance(event, BookTickEvent)
        assert isinstance(event, BookTickEvent)
        self.assertEqual(event.symbol, "BTCUSDT")
        self.assertEqual(event.last_update_id, 123)
        self.assertAlmostEqual(event.bids[0].price, 100.0)
        self.assertAlmostEqual(event.asks[0].qty, 1.5)

    def test_parse_trade(self) -> None:
        feed = BinancePublicFeed(queue.Queue(), symbols=("ETHUSDT",))
        event = feed._parse_data(
            {
                "e": "trade",
                "E": 1700000000000,
                "s": "ETHUSDT",
                "p": "2000.5",
                "q": "0.25",
                "T": 1700000000001,
                "m": True,
            }
        )

        self.assertIsInstance(event, TradeEvent)
        assert isinstance(event, TradeEvent)
        self.assertEqual(event.symbol, "ETHUSDT")
        self.assertAlmostEqual(event.price, 2000.5)
        self.assertAlmostEqual(event.qty, 0.25)
        self.assertTrue(event.is_buyer_maker)
        self.assertEqual(event.ts_ns, 1700000000001 * 1_000_000)


if __name__ == "__main__":
    unittest.main()
