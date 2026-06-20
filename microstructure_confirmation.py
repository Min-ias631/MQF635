"""microstructure.py — order-book confirmation + liquidity gate (canonical engine).

Implements Layer 3 of Strategy(PCA+BTC Attention).pdf:
  - weighted OBI (w_k = 1/k), microprice signal, and the
    MicroScore = 0.4*OBI + 0.3*MicroSignal + 0.3*TFI_short blend;
  - direction confirmation: MicroConfirm = 1(TradeDirection x MicroScore > 0);
  - liquidity/tradability filters per the PDF:
      * block if SpreadBps > 1.5 x RollingMedian(SpreadBps, 60),
      * exclude assets in the bottom 20% of cross-sectional ADV20
        (20-minute rolling average dollar volume, close x volume).

Spread and dollar-volume histories are bucketed per minute bar (intra-minute
re-evaluations replace the live bucket) so both adapters — the Yumi backtest,
which calls per market event, and the UI, which calls per snapshot — accumulate
one observation per bar.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np


class MicrostructureConfirmation:
    def __init__(self, trade_symbols, config) -> None:
        self.trade_symbols: Tuple[str, ...] = tuple(s.upper() for s in trade_symbols)
        self.config = config
        self._spread_hist: Dict[str, Deque[float]] = {
            s: deque(maxlen=config.spread_median_window) for s in self.trade_symbols
        }
        self._dv_hist: Dict[str, Deque[float]] = {
            s: deque(maxlen=config.adv_window) for s in self.trade_symbols
        }
        self._last_minute = None

    def evaluate(
        self, *, snapshot, adjusted_alpha: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, str]]:
        """Return (micro_scores, confirmed, eligible, reasons) per trade symbol."""
        micro_scores: Dict[str, float] = {}
        confirmed: Dict[str, bool] = {}
        eligible: Dict[str, bool] = {}
        reasons: Dict[str, str] = {}

        minute = self._minute_key(snapshot.timestamp)
        new_bar = minute != self._last_minute
        self._last_minute = minute

        c = self.config
        spreads: Dict[str, Optional[float]] = {}
        for sym in self.trade_symbols:
            book = snapshot.books.get(sym)
            bar = snapshot.bars.get(sym)

            obi = self._obi(book)
            micro = self._microprice(book)
            tfi_short = float(bar.tfi) if bar is not None else 0.0

            score = c.w_obi * obi + c.w_micro * micro["micro_signal"] + c.w_tfi * tfi_short
            micro_scores[sym] = float(score)

            alpha = float(adjusted_alpha.get(sym, 0.0))
            direction = 1 if alpha > 0 else (-1 if alpha < 0 else 0)
            if not getattr(c, "confirm_gate", True):
                # Ablation switch off: EffectiveMicroConfirm = 1 for all assets.
                confirmed[sym] = True
            else:
                # MicroConfirm: order book supports the intended trade direction.
                confirmed[sym] = bool(direction != 0 and (direction * score) > 0)

            spreads[sym] = micro.get("spread_bps")
            self._observe(self._spread_hist[sym], spreads[sym], new_bar)
            dollar_volume = (
                float(bar.close) * float(bar.volume) if bar is not None else None
            )
            self._observe(self._dv_hist[sym], dollar_volume, new_bar)

        adv_cutoff = self._adv_cutoff()
        for sym in self.trade_symbols:
            eligible[sym], reasons[sym] = self._eligible(sym, snapshot.books.get(sym), spreads[sym], adv_cutoff)

        return micro_scores, confirmed, eligible, reasons

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _minute_key(ts):
        try:
            return ts.replace(second=0, microsecond=0)
        except AttributeError:
            return ts

    @staticmethod
    def _observe(hist: Deque[float], value: Optional[float], new_bar: bool) -> None:
        """Append one observation per minute bar; replace the live bar's value."""
        if value is None:
            return
        if new_bar or not hist:
            hist.append(float(value))
        else:
            hist[-1] = float(value)

    def _adv_cutoff(self) -> Optional[float]:
        """Cross-sectional bottom-percentile cutoff of rolling avg dollar volume."""
        advs = [
            float(np.mean(hist)) for hist in self._dv_hist.values() if len(hist) > 0
        ]
        if len(advs) < 2:
            return None
        return float(np.percentile(advs, self.config.adv_exclude_pct))

    def _eligible(
        self, sym: str, book, spread_bps: Optional[float], adv_cutoff: Optional[float]
    ) -> Tuple[bool, str]:
        """Liquidity gate: two-sided book, spread vs rolling median, ADV rank."""
        if book is None or not book.bids or not book.asks:
            return False, "no_book"

        hist = self._spread_hist[sym]
        if spread_bps is not None and len(hist) > 0:
            median = float(np.median(hist))
            if median > 0 and spread_bps > self.config.spread_median_mult * median:
                return False, "wide_spread"

        dv = self._dv_hist[sym]
        if adv_cutoff is not None and len(dv) > 0:
            if float(np.mean(dv)) < adv_cutoff:
                return False, "low_adv"

        return True, ""

    def _obi(self, book) -> float:
        """Weighted order-book imbalance, w_k = 1/k (1-indexed depth levels)."""
        if book is None or not book.bids or not book.asks:
            return 0.0
        n = self.config.obi_levels
        bid_w = sum((1.0 / (k + 1)) * q for k, (_p, q) in enumerate(book.bids[:n]))
        ask_w = sum((1.0 / (k + 1)) * q for k, (_p, q) in enumerate(book.asks[:n]))
        denom = bid_w + ask_w
        if denom == 0:
            return 0.0
        return float((bid_w - ask_w) / denom)

    def _microprice(self, book) -> Dict[str, Optional[float]]:
        """Mid / microprice signal / spread_bps with a small spread floor."""
        if book is None or not book.bids or not book.asks:
            return {"mid": None, "micro_signal": 0.0, "spread_bps": None}

        best_bid_p, best_bid_q = book.bids[0]
        best_ask_p, best_ask_q = book.asks[0]

        mid = (best_bid_p + best_ask_p) / 2.0
        spread = best_ask_p - best_bid_p
        spread_bps = (spread / mid) * 10000 if mid > 0 else 0.0

        denom = best_bid_q + best_ask_q
        if denom == 0:
            return {"mid": mid, "micro_signal": 0.0, "spread_bps": spread_bps}

        microprice = (best_ask_p * best_bid_q + best_bid_p * best_ask_q) / denom
        epsilon = max(spread, 0.5 * mid / 10000) if mid > 0 else 1e-8
        micro_signal = (microprice - mid) / epsilon
        return {"mid": mid, "micro_signal": float(micro_signal), "spread_bps": spread_bps}


__all__ = ["MicrostructureConfirmation"]
