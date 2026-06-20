"""Crypto PCA strategy adapter tests."""

from __future__ import annotations

import unittest

from book import Book
from event import BookTickEvent, PriceLevel
from crypto_pca import CryptoPcaStrategy


def _event(ts_ns: int, symbol: str, bid_qty: float, ask_qty: float) -> BookTickEvent:
    base = 100.0 + (sum(ord(c) for c in symbol) % 20)
    return BookTickEvent.of(
        ts_ns,
        symbol,
        [
            PriceLevel(base, bid_qty),
            PriceLevel(base - 0.01, bid_qty * 0.8),
            PriceLevel(base - 0.02, bid_qty * 0.6),
        ],
        [
            PriceLevel(base + 0.01, ask_qty),
            PriceLevel(base + 0.02, ask_qty * 0.8),
            PriceLevel(base + 0.03, ask_qty * 0.6),
        ],
    )


class CryptoPcaStrategyTests(unittest.TestCase):
    def test_adapter_emits_native_signal_events_after_warmup(self) -> None:
        symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
        trade_symbols = ("ETHUSDT", "BNBUSDT", "SOLUSDT")
        book = Book(symbols=symbols)
        strategy = CryptoPcaStrategy(
            pca_symbols=symbols,
            trade_symbols=trade_symbols,
            pca_window=3,
            beta_window=2,
            cooldown_ns=0,
        )

        emitted = []
        ts = 1_700_000_000_000_000_000
        for minute in range(8):
            for idx, symbol in enumerate(symbols):
                bid_qty = 10.0 + minute + idx
                ask_qty = 1.0 + idx
                event = _event(ts + minute * 60_000_000_000 + idx, symbol, bid_qty, ask_qty)
                book.update(event)
                emitted.extend(strategy.on_market(event, book))

        self.assertTrue(strategy.latest_signals)
        self.assertTrue(all(signal.strategy_id == "crypto_pca" for signal in emitted))
        self.assertTrue(all(signal.symbol in trade_symbols for signal in emitted))


if __name__ == "__main__":
    unittest.main()
