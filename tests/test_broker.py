"""broker.py 基础单测。"""

from __future__ import annotations

import time
import unittest

from book import Book
from broker import PaperBroker, SimBroker
from event import BookTickEvent, OrderEvent, OrderStatus, OrderType, PriceLevel, Side


def _book() -> Book:
    book = Book(symbols=("BTCUSDT",))
    book.update(
        BookTickEvent.of(
            time.time_ns(),
            "BTCUSDT",
            [PriceLevel(100.0, 1.0)],
            [PriceLevel(101.0, 1.0)],
        )
    )
    return book


def _order(*, side: Side = Side.BUY, order_type: OrderType = OrderType.MARKET, price: float | None = None) -> OrderEvent:
    return OrderEvent(
        ts_ns=time.time_ns(),
        client_order_id=f"cid-{time.time_ns()}",
        symbol="BTCUSDT",
        side=side,
        order_type=order_type,
        qty=0.5,
        price=price,
    )


class BrokerTests(unittest.TestCase):
    def test_sim_market_buy_fills_at_best_ask(self) -> None:
        broker = SimBroker(fee_rate=0.001)
        result = broker.send(_order(side=Side.BUY), book=_book())

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.fills), 1)
        fill = result.fills[0]
        self.assertAlmostEqual(fill.fill_price, 101.0)
        self.assertAlmostEqual(fill.fill_qty, 0.5)
        self.assertAlmostEqual(fill.fee, 0.0505)
        self.assertEqual(fill.status, OrderStatus.FILLED)

    def test_sim_market_sell_fills_at_best_bid(self) -> None:
        broker = SimBroker()
        result = broker.send(_order(side=Side.SELL), book=_book())

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.fills), 1)
        self.assertAlmostEqual(result.fills[0].fill_price, 100.0)

    def test_sim_non_marketable_limit_only_acks(self) -> None:
        broker = SimBroker()
        result = broker.send(_order(side=Side.BUY, order_type=OrderType.LIMIT, price=99.0), book=_book())

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.fills), 0)

    def test_paper_broker_acks_without_fill(self) -> None:
        broker = PaperBroker()
        result = broker.send(_order(), book=_book())

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.fills), 0)

    def test_duplicate_client_order_id_rejected(self) -> None:
        broker = SimBroker()
        order = _order()
        first = broker.send(order, book=_book())
        second = broker.send(order, book=_book())

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)


if __name__ == "__main__":
    unittest.main()
