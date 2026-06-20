"""
risk_manager.py
Enforces all risk limits and circuit breakers.
"""

import numpy as np
from datetime import datetime, timezone, date


DAILY_LOSS_LIMIT   = -0.03    # -3%
MAX_SPREAD_MULT    = 1.5      # block if spread > 1.5x median
MIN_ELIGIBLE_ASSETS = 4


class RiskManager:
    def __init__(self):
        self.daily_pnl      = 0.0
        self.trading_halted = False
        self.halt_reason    = ""
        self.last_reset_day = None
        self.circuit_breaker_log: list[dict] = []

    # ── Daily reset ─────────────────────────────────────────────────────────
    def check_daily_reset(self):
        today = date.today()
        if self.last_reset_day != today:
            self.daily_pnl      = 0.0
            self.trading_halted = False
            self.halt_reason    = ""
            self.last_reset_day = today

    # ── Update PnL ───────────────────────────────────────────────────────────
    def update_pnl(self, pnl_increment: float):
        self.daily_pnl += pnl_increment
        if self.daily_pnl < DAILY_LOSS_LIMIT:
            self._trigger("Daily loss limit breached: "
                          f"{self.daily_pnl*100:.2f}%")

    def mark_daily_pnl(self, pnl_fraction: float):
        """Set today's PnL as a fraction of capital and fire the CB if breached.

        Use this from a rerun-driven UI (which recomputes current PnL each cycle)
        instead of update_pnl, which accumulates increments and would double-count
        across reruns.
        """
        self.daily_pnl = float(pnl_fraction)
        if self.daily_pnl < DAILY_LOSS_LIMIT:
            self._trigger("Daily loss limit breached: "
                          f"{self.daily_pnl*100:.2f}%")

    # ── Circuit breaker trigger ───────────────────────────────────────────────
    def _trigger(self, reason: str):
        if not self.trading_halted:
            self.trading_halted = True
            self.halt_reason    = reason
            self.circuit_breaker_log.append({
                "time": datetime.now(tz=timezone.utc).isoformat(),
                "reason": reason,
            })

    # ── Individual checks ────────────────────────────────────────────────────
    def check_regime(self, regime: str) -> bool:
        if regime == "RiskOff":
            self._trigger("BTC regime is Risk-Off")
            return False
        return True

    def check_spread(self, spread_bps: float, median_spread_bps: float,
                     symbol: str) -> bool:
        if median_spread_bps <= 0:
            return True
        if spread_bps > MAX_SPREAD_MULT * median_spread_bps:
            return False   # not a hard halt, just block this asset
        return True

    def check_depth(self, bids: list, asks: list, min_depth: float = 0) -> bool:
        return bool(bids and asks)

    def check_stale_data(self, last_update_ts: float,
                         max_stale_seconds: float = 60) -> bool:
        age = datetime.now(tz=timezone.utc).timestamp() - last_update_ts
        if age > max_stale_seconds:
            self._trigger(f"Stale data: {age:.0f}s since last update")
            return False
        return True

    # ── Full pre-trade check ─────────────────────────────────────────────────
    def is_tradeable(self, symbol: str,
                     spread_bps: float,
                     median_spread: float,
                     orderbook: dict,
                     data_fresh: bool) -> bool:
        """
        Returns True only if all checks pass for this symbol.
        Does NOT trigger a hard halt (those are separate calls).
        """
        if self.trading_halted:
            return False
        if not data_fresh:
            return False
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not self.check_depth(bids, asks):
            return False
        if not self.check_spread(spread_bps, median_spread, symbol):
            return False
        return True

    # ── Status summary ────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "trading_halted": self.trading_halted,
            "halt_reason":    self.halt_reason,
            "daily_pnl_pct":  round(self.daily_pnl * 100, 3),
            "circuit_breaker_count": len(self.circuit_breaker_log),
        }

    def resume(self):
        """Manually resume trading (e.g. new UTC day)."""
        self.trading_halted = False
        self.halt_reason    = ""
