"""Tests for the research overlays on the canonical engine + funding settlement.

Covers:
- IC direction stability gate: live direction from out-of-sample IC; unreliable
  signal -> direction 0; warmup -> configured prior.
- MicroConfirm ablation gate (confirm_gate=False -> all confirmed).
- 5-minute portfolio rebalance cadence with immediate RiskOff flatten.
- Funding settlement cashflow applied to portfolio PnL (long pays positive rate).
"""

from __future__ import annotations

import queue
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from book import Book
from broker import SimBroker
from event import BookTickEvent, FundingEvent, PriceLevel, TradeEvent
from portfolio import Portfolio
from risk import RiskManager
from runner import Runner
from crypto_pca import CryptoPcaStrategy, _canonical_classes


def _canonical_extras():
    from combined import CombinedRegimePcaStrategy
    from engine_config import (
        AlphaConfig,
        MicrostructureConfig,
        PortfolioConfig,
        StrategyConfig,
        UniverseConfig,
    )
    from events import Bar, BookSnapshot, MarketSnapshot
    from microstructure_confirmation import MicrostructureConfirmation
    from pca_flow import PCAFlowAlpha
    return {
        "CombinedRegimePcaStrategy": CombinedRegimePcaStrategy,
        "AlphaConfig": AlphaConfig,
        "MicrostructureConfig": MicrostructureConfig,
        "PortfolioConfig": PortfolioConfig,
        "StrategyConfig": StrategyConfig,
        "UniverseConfig": UniverseConfig,
        "Bar": Bar,
        "BookSnapshot": BookSnapshot,
        "MarketSnapshot": MarketSnapshot,
        "MicrostructureConfirmation": MicrostructureConfirmation,
        "PCAFlowAlpha": PCAFlowAlpha,
    }


class IcStabilityGateTests(unittest.TestCase):
    def _engine(self, **overrides):
        m = _canonical_extras()
        kwargs = dict(
            ic_stability_gate=True, ic_min=0.01, ic_tstat_min=2.0,
            ic_horizons=(1, 2), ic_min_periods=10, direction_sign=1,
        )
        kwargs.update(overrides)
        cfg = m["AlphaConfig"](**kwargs)
        return m["PCAFlowAlpha"](("BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"), cfg)

    @staticmethod
    def _seed_history(eng, n=40, predictive=+1.0, noise=0.0, seed=0):
        """Stuff finalized (alpha, close) pairs: next-period return = predictive*alpha."""
        rng = np.random.default_rng(seed)
        k = len(eng.trade_symbols)
        closes = np.full(k, 100.0)
        alphas = rng.normal(0, 1, (n, k))
        for t in range(n):
            eng._finalized.append((alphas[t].copy(), closes.copy()))
            ret = predictive * 0.001 * alphas[t] + noise * rng.normal(0, 0.001, k)
            closes = closes * (1.0 + ret)

    def test_gate_detects_continuation(self) -> None:
        eng = self._engine()
        self._seed_history(eng, predictive=+1.0)
        eng._refresh_gate()
        self.assertTrue(eng._gate_ready)
        self.assertEqual(eng._effective_sign(), 1)
        self.assertGreater(eng.direction_info["ic1"], 0.5)

    def test_gate_detects_reversal(self) -> None:
        eng = self._engine()
        self._seed_history(eng, predictive=-1.0)
        eng._refresh_gate()
        self.assertEqual(eng._effective_sign(), -1)

    def test_gate_zeroes_unreliable_signal(self) -> None:
        eng = self._engine()
        self._seed_history(eng, predictive=0.0, noise=1.0, seed=7)
        eng._refresh_gate()
        self.assertTrue(eng._gate_ready)
        self.assertEqual(eng._effective_sign(), 0)

    def test_warmup_falls_back_to_configured_prior(self) -> None:
        eng = self._engine(direction_sign=-1)
        self._seed_history(eng, n=5, predictive=+1.0)  # below ic_min_periods
        eng._refresh_gate()
        self.assertFalse(eng._gate_ready)
        self.assertEqual(eng._effective_sign(), -1)

    def test_gate_disabled_uses_prior(self) -> None:
        m = _canonical_extras()
        cfg = m["AlphaConfig"](ic_stability_gate=False, direction_sign=-1)
        eng = m["PCAFlowAlpha"](("BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"), cfg)
        self.assertEqual(eng._effective_sign(), -1)


class MicroConfirmGateTests(unittest.TestCase):
    def test_confirm_gate_off_confirms_everything(self) -> None:
        m = _canonical_extras()
        syms = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
        micro = m["MicrostructureConfirmation"](syms, m["MicrostructureConfig"](confirm_gate=False))
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        books = {s: m["BookSnapshot"](ts, s, ((99.99, 5.0),), ((100.01, 5.0),)) for s in syms}
        bars = {s: m["Bar"](ts, s, 100, 100, 100, 100, 10.0, 5.0, 5.0) for s in syms}
        snap = m["MarketSnapshot"](ts, bars, books, {}, None)

        # Zero alpha would normally fail direction confirmation.
        _, confirmed, _, _ = micro.evaluate(snapshot=snap, adjusted_alpha={s: 0.0 for s in syms})
        self.assertTrue(all(confirmed.values()))


class RebalanceCadenceTests(unittest.TestCase):
    def test_construct_called_on_cadence_and_riskoff_flattens(self) -> None:
        m = _canonical_extras()

        class ForcedRegime(m["CombinedRegimePcaStrategy"]):
            regime_override = ("Strong", 1.0)

            def _regime(self, _btc):
                return self.regime_override

        cfg = m["StrategyConfig"](
            universe=m["UniverseConfig"](
                pca_symbols=("BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"),
                trade_symbols=("AAAUSDT", "BBBUSDT", "CCCUSDT"),
            ),
            alpha=m["AlphaConfig"](pca_window=3, beta_window=2, ic_stability_gate=False),
            portfolio=m["PortfolioConfig"](rebalance_minutes=5),
        )
        strategy = ForcedRegime(cfg)

        calls = []
        original = strategy.portfolio.construct

        def counting(**kwargs):
            calls.append(kwargs["timestamp"])
            return original(**kwargs)

        strategy.portfolio.construct = counting

        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        syms = cfg.universe.pca_symbols
        for minute in range(12):
            ts = t0 + timedelta(minutes=minute)
            bars = {s: m["Bar"](ts, s, 100 + i, 100 + i, 100 + i, 100 + i, 10.0, 6.0, 4.0)
                    for i, s in enumerate(syms)}
            books = {s: m["BookSnapshot"](ts, s, ((99.99, 5.0),), ((100.01, 5.0),))
                     for s in cfg.universe.trade_symbols}
            target = strategy.on_market_snapshot(m["MarketSnapshot"](ts, bars, books, {}, None))

        # 12 one-minute snapshots, 5-minute cadence -> construct at minutes 0, 5, 10.
        self.assertEqual(len(calls), 3)

        # RiskOff flattens immediately, bypassing the cadence.
        ForcedRegime.regime_override = ("RiskOff", 0.0)
        ts = t0 + timedelta(minutes=12)
        bars = {s: m["Bar"](ts, s, 100, 100, 100, 100, 10.0, 6.0, 4.0) for s in syms}
        books = {s: m["BookSnapshot"](ts, s, ((99.99, 5.0),), ((100.01, 5.0),))
                 for s in cfg.universe.trade_symbols}
        target = strategy.on_market_snapshot(m["MarketSnapshot"](ts, bars, books, {}, None))
        self.assertEqual(target.regime_multiplier, 0.0)
        self.assertTrue(all(w == 0.0 for w in target.weights.values()))


class FundingSettlementTests(unittest.TestCase):
    def test_long_pays_positive_funding(self) -> None:
        pf = Portfolio()
        from event import FillEvent, OrderStatus, Side, ExecutionType
        pf.on_fill(FillEvent(
            ts_ns=1, client_order_id="c1", exchange_order_id="x1", symbol="ETHUSDT",
            side=Side.BUY, fill_price=100.0, fill_qty=2.0, cum_qty=2.0, leaves_qty=0.0,
            fee=0.0, fee_asset="USDT", status=OrderStatus.FILLED,
            exec_type=ExecutionType.TRADE, exchange_trade_id="t1",
        ))
        delta = pf.apply_funding("ETHUSDT", rate=0.0001, mark_price=100.0)
        self.assertAlmostEqual(delta, -0.02)              # 2 * 100 * 0.0001 paid
        self.assertAlmostEqual(pf.daily_funding_pnl, -0.02)
        self.assertAlmostEqual(pf.daily_pnl(), -0.02)     # no marks -> only funding

        # Short receives positive funding.
        delta_short = Portfolio()
        delta_short.on_fill(FillEvent(
            ts_ns=2, client_order_id="c2", exchange_order_id="x2", symbol="ETHUSDT",
            side=Side.SELL, fill_price=100.0, fill_qty=2.0, cum_qty=2.0, leaves_qty=0.0,
            fee=0.0, fee_asset="USDT", status=OrderStatus.FILLED,
            exec_type=ExecutionType.TRADE, exchange_trade_id="t2",
        ))
        self.assertAlmostEqual(delta_short.apply_funding("ETHUSDT", 0.0001, 100.0), +0.02)

    def test_runner_settles_funding_on_event(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        book = Book(symbols=("ETHUSDT",))
        pf = Portfolio()
        runner = Runner(
            event_queue=q, book=book, strategy=_NullStrategy(),
            risk=RiskManager(), broker=SimBroker(), portfolio=pf,
        )
        # Establish a price and a position, then a funding settlement.
        q.put(BookTickEvent.of(1, "ETHUSDT", [PriceLevel(99.99, 5.0)], [PriceLevel(100.01, 5.0)]))
        q.put(TradeEvent(2, "ETHUSDT", 100.0, 1.0, is_buyer_maker=False))
        runner.drain()
        from event import FillEvent, OrderStatus, Side, ExecutionType
        pf.on_fill(FillEvent(
            ts_ns=3, client_order_id="c3", exchange_order_id="x3", symbol="ETHUSDT",
            side=Side.BUY, fill_price=100.0, fill_qty=1.0, cum_qty=1.0, leaves_qty=0.0,
            fee=0.0, fee_asset="USDT", status=OrderStatus.FILLED,
            exec_type=ExecutionType.TRADE, exchange_trade_id="t3",
        ))
        q.put(FundingEvent(ts_ns=4, account_id="default", symbol="ETHUSDT",
                           funding_rate=0.0002, funding_amount=0.0))
        runner.drain()
        self.assertAlmostEqual(pf.funding_pnl_total, -1.0 * 100.0 * 0.0002)


class _NullStrategy:
    def on_market(self, event, book):
        return []


if __name__ == "__main__":
    unittest.main()
