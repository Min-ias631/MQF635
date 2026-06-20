"""risk.py 基础单测。"""

from __future__ import annotations

import time
import unittest

from config import Settings
from event import OrderType, RiskAction, Side, SignalAction, SignalEvent
from risk import PortfolioStub, RiskManager, RiskState


def _settings(**overrides) -> Settings:
    base = dict(
        mode="paper",
        binance_api_key=None,
        binance_secret=None,
        symbols=("BTCUSDT",),
        window_length=60,
        signal_threshold=2.0,
        strategy_weight=1.0,
        max_single_order_qty=0.01,
        daily_max_loss=50.0,
        max_total_position_qty=0.02,
        max_order_rate_per_symbol=5,
        max_consecutive_order_failures=3,
        event_queue_maxsize=10000,
        log_level="INFO",
        data_dir="./data",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class RiskManagerTests(unittest.TestCase):
    def test_long_produces_buy_order(self) -> None:
        risk = RiskManager(_settings())
        pf = PortfolioStub(positions={})
        sig = SignalEvent(
            ts_ns=time.time_ns(),
            symbol="BTCUSDT",
            strategy_id="t",
            action=SignalAction.LONG,
            strength=1.0,
        )
        res = risk.check(sig, pf)
        self.assertIsNotNone(res.order)
        assert res.order is not None
        self.assertEqual(res.order.side, Side.BUY)
        self.assertEqual(res.order.order_type, OrderType.MARKET)
        self.assertAlmostEqual(res.order.qty, 0.01)

    def test_flat_with_no_position_rejects(self) -> None:
        risk = RiskManager(_settings())
        pf = PortfolioStub(positions={})
        sig = SignalEvent(
            ts_ns=time.time_ns(),
            symbol="BTCUSDT",
            strategy_id="t",
            action=SignalAction.FLAT,
        )
        res = risk.check(sig, pf)
        self.assertIsNone(res.order)
        self.assertEqual(res.reject_reason, "no_order_needed")

    def test_daily_max_loss_triggers_halt(self) -> None:
        risk = RiskManager(_settings(daily_max_loss=10.0))
        pf = PortfolioStub(positions={}, _daily_pnl=-20.0)
        sig = SignalEvent(
            ts_ns=time.time_ns(),
            symbol="BTCUSDT",
            strategy_id="t",
            action=SignalAction.LONG,
        )
        res = risk.check(sig, pf)
        self.assertIsNone(res.order)
        self.assertEqual(risk.state, RiskState.HALTED)
        self.assertIsNotNone(res.risk_event)
        assert res.risk_event is not None
        self.assertEqual(res.risk_event.action, RiskAction.HALT)

    def test_rate_limit(self) -> None:
        risk = RiskManager(_settings(max_order_rate_per_symbol=2))
        pf = PortfolioStub(positions={})
        ts = time.time_ns()
        for i in range(2):
            sig = SignalEvent(ts_ns=ts + i, symbol="BTCUSDT", strategy_id="t", action=SignalAction.LONG)
            res = risk.check(sig, pf)
            self.assertIsNotNone(res.order)
        sig = SignalEvent(ts_ns=ts + 2, symbol="BTCUSDT", strategy_id="t", action=SignalAction.LONG)
        res = risk.check(sig, pf)
        self.assertIsNone(res.order)
        self.assertEqual(res.reject_reason, "order_rate_limited")

    def test_flat_allowed_when_over_max_exposure(self) -> None:
        """P0：超限时仍必须能平仓，不能被敞口 cap 误拦。"""
        risk = RiskManager(_settings(max_total_position_qty=0.01, max_single_order_qty=0.005))
        pf = PortfolioStub(positions={"BTCUSDT": 0.02})  # 已超过 max_total
        sig = SignalEvent(
            ts_ns=time.time_ns(),
            symbol="BTCUSDT",
            strategy_id="t",
            action=SignalAction.FLAT,
        )
        res = risk.check(sig, pf)
        self.assertIsNotNone(res.order)
        assert res.order is not None
        self.assertEqual(res.order.side, Side.SELL)
        self.assertAlmostEqual(res.order.qty, 0.02)

    def test_consecutive_failures_halt(self) -> None:
        risk = RiskManager(_settings(max_consecutive_order_failures=2))
        ts = time.time_ns()
        risk.on_order_failure(ts)
        ev = risk.on_order_failure(ts)
        self.assertEqual(risk.state, RiskState.HALTED)
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertEqual(ev.action, RiskAction.HALT)


if __name__ == "__main__":
    unittest.main()
