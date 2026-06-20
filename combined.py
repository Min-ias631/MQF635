"""combined.py — regime-gated PCA flow strategy for the Streamlit UI surface.

Composes the canonical ``PCAFlowAlpha`` + ``MicrostructureConfirmation`` with a
BTC daily regime gate and a long-short portfolio constructor. This is the entry
point the eugene-ui adapter drives (``on_market_snapshot`` per snapshot, then
reads ``latest_signals`` / ``portfolio.construct`` / the returned target).

Portfolio and regime math are ports of ``eugene-ui/strategy/portfolio.py`` and
``features.compute_btc_regime``; no new alpha logic. numpy-only, duck-typed on
btc_daily (a pandas DataFrame in the UI) so this module has no hard pandas dep.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np

from microstructure_confirmation import MicrostructureConfirmation
from pca_flow import PCAFlowAlpha


@dataclass
class SignalSnapshot:
    """Per-symbol signal record exposed via ``strategy.latest_signals``."""

    symbol: str
    adjusted_alpha: float
    residual_z: float
    funding_signal: float
    micro_score: float
    micro_confirmed: bool
    eligible: bool


@dataclass
class Target:
    """Portfolio target returned by on_market_snapshot / portfolio.construct."""

    timestamp: Any
    regime: str
    regime_multiplier: float
    weights: Dict[str, float]


class Portfolio:
    """Long-short construction: Q80/Q20, vol-scaled, regime-gated, capped.

    Realized vol is tracked internally from the close prices seen on each
    snapshot (the UI adapter calls ``construct`` without passing klines), so
    vol scaling reuses portfolio.compute_realized_vol semantics.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._closes: Dict[str, deque] = {}
        self._prev_weights: Dict[str, float] = {}   # for the no-trade band

    def observe(self, snapshot) -> None:
        for sym, bar in snapshot.bars.items():
            buf = self._closes.get(sym)
            if buf is None:
                buf = deque(maxlen=self.config.vol_window + 1)
                self._closes[sym] = buf
            buf.append(float(bar.close))

    def _realized_vol(self, symbol: str) -> float:
        closes = list(self._closes.get(symbol, ()))
        if len(closes) < 2:
            return 1.0
        rets = np.diff(np.log(np.asarray(closes, dtype=float)))
        vol = np.std(rets, ddof=1) * np.sqrt(525600)   # minutes in a year
        return max(float(vol), 1e-6)

    def construct(
        self,
        *,
        timestamp,
        signals: Dict[str, SignalSnapshot],
        regime: str,
        regime_multiplier: float,
        long_quantile: float,
        short_quantile: float,
    ) -> Target:
        adjusted_alpha = {s: sig.adjusted_alpha for s, sig in signals.items()}
        micro_confirm = {s: sig.micro_confirmed for s, sig in signals.items()}
        liquidity_ok = {s: sig.eligible for s, sig in signals.items()}

        weights = {s: 0.0 for s in adjusted_alpha}
        if regime_multiplier == 0:
            self._prev_weights = dict(weights)   # flatten immediately, no band
            return Target(timestamp, regime, regime_multiplier, weights)

        gross = regime_multiplier * self.config.base_gross_exposure

        # Per the PDF: the eligible universe (and the Q80/Q20 thresholds) are
        # defined by the liquidity/tradability filters alone; MicroConfirm is an
        # additional per-asset selection condition applied within L_t / S_t.
        eligible = {
            s: a for s, a in adjusted_alpha.items() if liquidity_ok.get(s, False)
        }
        if len(eligible) < self.config.min_eligible:
            self._prev_weights = dict(weights)   # flatten, no band
            return Target(timestamp, regime, regime_multiplier, weights)

        syms = list(eligible.keys())
        alphas = np.array([eligible[s] for s in syms], dtype=float)
        q_long = np.quantile(alphas, long_quantile)
        q_short = np.quantile(alphas, short_quantile)

        long_syms = [s for s, a in zip(syms, alphas)
                     if a > q_long and micro_confirm.get(s, False)]
        short_syms = [s for s, a in zip(syms, alphas)
                      if a < q_short and micro_confirm.get(s, False)]

        long_w = self._vol_scale_leg(eligible, long_syms, +1, gross) if long_syms else {}
        short_w = self._vol_scale_leg(eligible, short_syms, -1, gross) if short_syms else {}

        combined = self._cap_and_renormalize({**long_w, **short_w}, self.config.single_asset_cap, gross)
        for s in adjusted_alpha:
            weights[s] = combined.get(s, 0.0)

        weights = self._apply_no_trade_band(weights)
        return Target(timestamp, regime, regime_multiplier, weights)

    def _apply_no_trade_band(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Hold a name's previous weight unless the target moved by > no_trade_band.

        Cuts turnover from tiny re-rankings. (May introduce small net-exposure
        drift since some names update and others don't; acceptable vs the churn.)
        """
        band = getattr(self.config, "no_trade_band", 0.0)
        if band and band > 0 and self._prev_weights:
            for s in list(weights.keys()):
                if abs(weights[s] - self._prev_weights.get(s, 0.0)) < band:
                    weights[s] = self._prev_weights.get(s, 0.0)
        self._prev_weights = dict(weights)
        return weights

    def _vol_scale_leg(self, eligible, leg_syms, leg_sign, gross) -> Dict[str, float]:
        vols = np.array([self._realized_vol(s) for s in leg_syms], dtype=float)
        scores = np.abs(np.array([eligible[s] for s in leg_syms], dtype=float)) / vols
        total = scores.sum()
        if total < 1e-10:
            return {}
        return {
            s: leg_sign * (sc / total) * 0.5 * gross
            for s, sc in zip(leg_syms, scores)
        }

    @staticmethod
    def _cap_and_renormalize(weights, cap, gross) -> Dict[str, float]:
        long_w = {s: w for s, w in weights.items() if w > 0}
        short_w = {s: w for s, w in weights.items() if w < 0}

        def _cap_leg(leg):
            capped = {s: min(abs(w), cap) * np.sign(w) for s, w in leg.items()}
            total_abs = sum(abs(v) for v in capped.values())
            target = 0.5 * gross
            if total_abs > 1e-10:
                scale = target / total_abs
                capped = {s: v * scale for s, v in capped.items()}
            return capped

        return {**_cap_leg(long_w), **_cap_leg(short_w)}


class CombinedRegimePcaStrategy:
    """Regime-gated PCA flow strategy consumed by the eugene-ui adapter."""

    def __init__(self, config) -> None:
        self.config = config
        self.flow_alpha = PCAFlowAlpha(config.universe.pca_symbols, config.alpha)
        self.microstructure = MicrostructureConfirmation(
            config.universe.trade_symbols, config.microstructure
        )
        self.portfolio = Portfolio(config.portfolio)
        self.latest_signals: Dict[str, SignalSnapshot] = {}
        self._last_rebalance_minute = None
        self._last_target: Optional[Target] = None

    def on_market_snapshot(self, snapshot) -> Target:
        self.portfolio.observe(snapshot)

        adjusted = self.flow_alpha.update(snapshot)
        residual_z, funding_signal = self.flow_alpha.raw_components(snapshot)
        micro_scores, confirmed, eligible, _reasons = self.microstructure.evaluate(
            snapshot=snapshot, adjusted_alpha=adjusted
        )

        self.latest_signals = {
            s: SignalSnapshot(
                symbol=s,
                adjusted_alpha=float(adjusted.get(s, 0.0)),
                residual_z=float(residual_z.get(s, 0.0)),
                funding_signal=float(funding_signal.get(s, 0.0)),
                micro_score=float(micro_scores.get(s, 0.0)),
                micro_confirmed=bool(confirmed.get(s, False)),
                eligible=bool(eligible.get(s, False)),
            )
            for s in self.config.universe.trade_symbols
        }

        regime, multiplier = self._regime(snapshot.btc_daily)

        # RiskOff flattens immediately (PDF trading rule 1), bypassing the cadence.
        if multiplier == 0:
            target = Target(
                snapshot.timestamp, regime, multiplier,
                {s: 0.0 for s in self.config.universe.trade_symbols},
            )
            self._last_target = target
            self._last_rebalance_minute = self._minute(snapshot.timestamp)
            return target

        # PDF Execution Algorithm: alpha updates every minute, but the target
        # portfolio refreshes only every rebalance_minutes to limit turnover.
        minute = self._minute(snapshot.timestamp)
        due = (
            self._last_target is None
            or self._last_rebalance_minute is None
            or self._elapsed_minutes(self._last_rebalance_minute, minute)
            >= self.config.portfolio.rebalance_minutes
        )
        if not due:
            held = self._last_target
            return Target(snapshot.timestamp, regime, multiplier, dict(held.weights))

        target = self.portfolio.construct(
            timestamp=snapshot.timestamp,
            signals=self.latest_signals,
            regime=regime,
            regime_multiplier=multiplier,
            long_quantile=self.config.alpha.long_quantile,
            short_quantile=self.config.alpha.short_quantile,
        )
        self._last_target = target
        self._last_rebalance_minute = minute
        return target

    @staticmethod
    def _minute(ts):
        try:
            return ts.replace(second=0, microsecond=0)
        except AttributeError:
            return ts

    @staticmethod
    def _elapsed_minutes(start, end) -> float:
        delta = end - start
        if isinstance(delta, timedelta):
            return delta.total_seconds() / 60.0
        return float(delta)

    def _regime(self, btc_daily) -> Tuple[str, float]:
        """BTC daily regime gate (features.compute_btc_regime), config-parameterized.

        btc_daily is a pandas DataFrame with a 'close' column in the UI, or None.
        """
        rc = self.config.regime
        if not getattr(rc, "enabled", True):
            return "AllOn", 1.0          # regime gate ablated -> always deploy
        if btc_daily is None:
            return "Unknown", 0.0
        try:
            if getattr(btc_daily, "empty", False):
                return "Unknown", 0.0
            closes = np.asarray(btc_daily["close"], dtype=float)
        except Exception:
            return "Unknown", 0.0

        short = rc.momentum_short_days
        long = rc.momentum_long_days
        vw = rc.volatility_window_days
        lookback = rc.volatility_threshold_days

        if len(closes) < lookback + long + 2:
            return "Unknown", 0.0

        c = closes[:-1]  # drop the (possibly incomplete) current candle
        if len(c) < lookback:
            return "Unknown", 0.0

        r_short = c[-1] / c[-(short + 1)] - 1.0
        r_long = c[-1] / c[-(long + 1)] - 1.0

        daily_rets = np.diff(np.log(c))
        vol_now = float(np.std(daily_rets[-vw:], ddof=1) * np.sqrt(365))
        past_vols = [
            float(np.std(daily_rets[i:i + vw], ddof=1) * np.sqrt(365))
            for i in range(len(daily_rets) - lookback, len(daily_rets) - vw)
            if i >= 0
        ]
        vol_threshold = (
            float(np.percentile(past_vols, rc.volatility_percentile)) if past_vols else 999.0
        )

        if r_long > 0 and r_short > 0 and vol_now < vol_threshold:
            return "Strong", 1.0
        if r_long > 0 and r_short <= 0 and vol_now < vol_threshold:
            return "Moderate", 0.5
        return "RiskOff", 0.0


__all__ = ["CombinedRegimePcaStrategy", "Portfolio", "SignalSnapshot", "Target"]
