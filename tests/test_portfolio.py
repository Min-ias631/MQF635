"""portfolio.py 基础单测。"""

from __future__ import annotations

import time
import unittest

from event import ExecutionType, FillEvent, OrderStatus, Side
from portfolio import Portfolio


def _fill(
    *,
    side: Side,
    qty: float,
    price: float,
    symbol: str = "BTCUSDT",
    trade_id: str = "t1",
    client_order_id: str = "c1",
    exchange_order_id: str = "e1",
    fee: float = 0.0,
    account_id: str = "default",
) -> FillEvent:
    return FillEvent(
        ts_ns=time.time_ns(),
        client_order_id=client_order_id,
        exchange_order_id=exchange_order_id,
        symbol=symbol,
        side=side,
        fill_price=price,
        fill_qty=qty,
        cum_qty=qty,
        leaves_qty=0.0,
        fee=fee,
        fee_asset="USDT",
        status=OrderStatus.FILLED,
        exec_type=ExecutionType.TRADE,
        exchange_trade_id=trade_id,
        account_id=account_id,
    )


class PortfolioTests(unittest.TestCase):
    def test_open_long_and_add_same_side_updates_average(self) -> None:
        pf = Portfolio()
        self.assertTrue(pf.on_fill(_fill(side=Side.BUY, qty=1.0, price=100.0, trade_id="1")))
        self.assertTrue(pf.on_fill(_fill(side=Side.BUY, qty=1.0, price=110.0, trade_id="2")))

        view = pf.get_position("BTCUSDT")
        self.assertIsNotNone(view)
        assert view is not None
        self.assertAlmostEqual(view.qty, 2.0)
        self.assertAlmostEqual(view.avg_price, 105.0)
        self.assertAlmostEqual(view.realized_pnl_gross, 0.0)
        self.assertAlmostEqual(view.realized_pnl_net, 0.0)

    def test_partial_close_realized_pnl_gross_and_net(self) -> None:
        pf = Portfolio()
        pf.on_fill(_fill(side=Side.BUY, qty=2.0, price=100.0, trade_id="1"))
        pf.on_fill(_fill(side=Side.SELL, qty=0.5, price=120.0, trade_id="2", fee=0.25))

        view = pf.get_position("BTCUSDT")
        self.assertIsNotNone(view)
        assert view is not None
        self.assertAlmostEqual(view.qty, 1.5)
        self.assertAlmostEqual(view.avg_price, 100.0)
        self.assertAlmostEqual(view.realized_pnl_gross, 10.0)
        self.assertAlmostEqual(view.realized_pnl_net, 9.75)
        self.assertAlmostEqual(view.realized_pnl, 9.75)
        self.assertAlmostEqual(view.fees, 0.25)
        self.assertAlmostEqual(pf.daily_pnl(), 9.75)

    def test_full_close_resets_average_price(self) -> None:
        pf = Portfolio()
        pf.on_fill(_fill(side=Side.BUY, qty=1.0, price=100.0, trade_id="1"))
        pf.on_fill(_fill(side=Side.SELL, qty=1.0, price=110.0, trade_id="2"))

        view = pf.get_position("BTCUSDT")
        self.assertIsNotNone(view)
        assert view is not None
        self.assertAlmostEqual(view.qty, 0.0)
        self.assertAlmostEqual(view.avg_price, 0.0)
        self.assertAlmostEqual(view.realized_pnl_gross, 10.0)

    def test_flip_long_to_short_uses_fill_price_as_new_average(self) -> None:
        pf = Portfolio()
        pf.on_fill(_fill(side=Side.BUY, qty=1.0, price=100.0, trade_id="1"))
        pf.on_fill(_fill(side=Side.SELL, qty=2.0, price=90.0, trade_id="2"))

        view = pf.get_position("BTCUSDT")
        self.assertIsNotNone(view)
        assert view is not None
        self.assertAlmostEqual(view.qty, -1.0)
        self.assertAlmostEqual(view.avg_price, 90.0)
        self.assertAlmostEqual(view.realized_pnl_gross, -10.0)

    def test_flip_short_to_long_uses_fill_price_as_new_average(self) -> None:
        pf = Portfolio()
        pf.on_fill(_fill(side=Side.SELL, qty=1.0, price=100.0, trade_id="1"))
        pf.on_fill(_fill(side=Side.BUY, qty=2.0, price=95.0, trade_id="2"))

        view = pf.get_position("BTCUSDT")
        self.assertIsNotNone(view)
        assert view is not None
        self.assertAlmostEqual(view.qty, 1.0)
        self.assertAlmostEqual(view.avg_price, 95.0)
        self.assertAlmostEqual(view.realized_pnl_gross, 5.0)

    def test_duplicate_fill_is_ignored(self) -> None:
        pf = Portfolio()
        f = _fill(side=Side.BUY, qty=1.0, price=100.0, trade_id="dup")

        self.assertTrue(pf.on_fill(f))
        self.assertFalse(pf.on_fill(f))
        self.assertAlmostEqual(pf.position_qty("BTCUSDT"), 1.0)

    def test_mark_to_market_daily_pnl_includes_unrealized(self) -> None:
        pf = Portfolio()
        pf.on_fill(_fill(side=Side.BUY, qty=1.0, price=100.0, trade_id="1", fee=0.1))
        pf.set_mark_price("BTCUSDT", 112.0)

        self.assertAlmostEqual(pf.unrealized_pnl_total, 12.0)
        self.assertAlmostEqual(pf.daily_pnl(), 11.9)

    def test_account_mismatch_does_not_book_fill(self) -> None:
        pf = Portfolio(account_id="acct-a")
        f = _fill(side=Side.BUY, qty=1.0, price=100.0, trade_id="1", account_id="acct-b")

        self.assertFalse(pf.on_fill(f))
        self.assertAlmostEqual(pf.position_qty("BTCUSDT"), 0.0)
        self.assertIsNone(pf.get_position("BTCUSDT"))

    def test_reset_daily_preserves_lifetime_totals_and_can_clear_dedup(self) -> None:
        pf = Portfolio()
        f = _fill(side=Side.BUY, qty=1.0, price=100.0, trade_id="1")
        close = _fill(side=Side.SELL, qty=1.0, price=110.0, trade_id="2", fee=0.5)
        pf.on_fill(f)
        pf.on_fill(close)

        self.assertAlmostEqual(pf.daily_pnl(), 9.5)
        pf.reset_daily(clear_seen=True)

        view = pf.get_position("BTCUSDT")
        self.assertIsNotNone(view)
        assert view is not None
        self.assertAlmostEqual(view.realized_pnl_gross, 10.0)
        self.assertAlmostEqual(view.realized_pnl_net, 9.5)
        self.assertAlmostEqual(view.daily_realized_pnl_gross, 0.0)
        self.assertAlmostEqual(view.daily_realized_pnl_net, 0.0)
        self.assertAlmostEqual(view.fees, 0.5)
        self.assertAlmostEqual(view.daily_fees, 0.0)
        self.assertAlmostEqual(pf.daily_pnl(), 0.0)

        # clear_seen=True starts a new session/day; duplicate ids may be replayed by caller policy.
        self.assertTrue(pf.on_fill(close))


if __name__ == "__main__":
    unittest.main()
