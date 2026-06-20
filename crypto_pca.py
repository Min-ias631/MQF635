"""Trading-framework adapter for the shared PCA flow alpha engine.

Now that the whole project is a single flat package, the engine modules
(engine_config, events, pca_flow, microstructure_confirmation) are imported
directly — no sys.path bridge needed.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque

import numpy as np

from book import Book
from event import (
    MarketEvent,
    PriceLevel,
    SignalAction,
    SignalEvent,
    TradeEvent,
    freeze_meta,
)


def _canonical_classes():
    from engine_config import AlphaConfig, MicrostructureConfig
    from events import Bar, BookSnapshot, FundingRate, MarketSnapshot
    from microstructure_confirmation import MicrostructureConfirmation
    from pca_flow import PCAFlowAlpha

    return {
        "AlphaConfig": AlphaConfig,
        "MicrostructureConfig": MicrostructureConfig,
        "Bar": Bar,
        "BookSnapshot": BookSnapshot,
        "FundingRate": FundingRate,
        "MarketSnapshot": MarketSnapshot,
        "MicrostructureConfirmation": MicrostructureConfirmation,
        "PCAFlowAlpha": PCAFlowAlpha,
    }


@dataclass(slots=True)
class _MinuteBarState:
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float
    taker_sell_volume: float


@dataclass(slots=True)
class CryptoPcaStrategy:
    """Adapter that preserves Yumi's Strategy interface and architecture boundary."""

    strategy_id: str = "crypto_pca"
    pca_symbols: tuple[str, ...] = (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "DOTUSDT",
    )
    trade_symbols: tuple[str, ...] = (
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "DOTUSDT",
    )
    pca_window: int = 120
    beta_window: int = 60
    direction_sign: int = 1      # +1 continuation, -1 reversal (set from IC validation)
    entry_alpha_threshold: float = 0.0
    # Per-symbol re-signal cooldown. Default matches the strategy note's baseline
    # 5-minute portfolio refresh (alpha still updates every minute internally).
    cooldown_ns: int = 300_000_000_000
    _classes: dict[str, Any] = field(init=False, repr=False)
    _flow_alpha: Any = field(init=False, repr=False)
    _microstructure: Any = field(init=False, repr=False)
    _bars: dict[str, _MinuteBarState] = field(default_factory=dict, init=False, repr=False)
    _last_action: dict[str, SignalAction] = field(default_factory=dict, init=False, repr=False)
    _last_signal_ts_ns: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_minute_ns: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    latest_signals: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.pca_symbols = tuple(symbol.upper() for symbol in self.pca_symbols)
        self.trade_symbols = tuple(symbol.upper() for symbol in self.trade_symbols)
        self._classes = _canonical_classes()
        alpha_config = self._classes["AlphaConfig"](
            pca_window=self.pca_window,
            beta_window=self.beta_window,
            direction_sign=self.direction_sign,
        )
        micro_config = self._classes["MicrostructureConfig"]()
        self._flow_alpha = self._classes["PCAFlowAlpha"](self.pca_symbols, alpha_config)
        self._microstructure = self._classes["MicrostructureConfirmation"](
            self.trade_symbols,
            micro_config,
        )

    def on_market(self, event: MarketEvent, book: Book) -> list[SignalEvent]:
        self._update_bar_state(event, book)
        if not self._has_required_state(book):
            return []

        snapshot = self._build_snapshot(event, book)
        adjusted_alpha = self._flow_alpha.update(snapshot)
        residual_z, funding_signal = self._flow_alpha.raw_components(snapshot)
        micro_scores, confirmed, eligible, reasons = self._microstructure.evaluate(
            snapshot=snapshot,
            adjusted_alpha=adjusted_alpha,
        )

        signals: list[SignalEvent] = []
        self.latest_signals = {}
        for symbol in self.trade_symbols:
            alpha = float(adjusted_alpha.get(symbol, 0.0))
            action = self._action_from_alpha(alpha, confirmed.get(symbol, False), eligible.get(symbol, False))
            self.latest_signals[symbol] = {
                "adjusted_alpha": alpha,
                "residual_z": float(residual_z.get(symbol, 0.0)),
                "funding_signal": float(funding_signal.get(symbol, 0.0)),
                "micro_score": float(micro_scores.get(symbol, 0.0)),
                "micro_confirmed": bool(confirmed.get(symbol, False)),
                "eligible": bool(eligible.get(symbol, False)),
                "reason": reasons.get(symbol, ""),
                "action": int(action),
            }
            if not self._should_emit(symbol, action, event.ts_ns):
                continue
            signals.append(
                SignalEvent(
                    ts_ns=event.ts_ns,
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    action=action,
                    strength=min(1.0, abs(alpha)),
                    meta=freeze_meta(self.latest_signals[symbol]),
                )
            )
            self._last_action[symbol] = action
            self._last_signal_ts_ns[symbol] = event.ts_ns
        return signals

    def _update_bar_state(self, event: MarketEvent, book: Book) -> None:
        symbol = event.symbol.upper()
        snap = book.snapshot(symbol)
        if snap is None:
            return
        price = snap.last_price or snap.mid
        if price is None:
            return

        minute_ns = event.ts_ns - (event.ts_ns % 60_000_000_000)
        state = self._bars.get(symbol)
        if state is None or self._last_minute_ns.get(symbol) != minute_ns:
            state = _MinuteBarState(
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
                taker_buy_volume=0.0,
                taker_sell_volume=0.0,
            )
            self._bars[symbol] = state
            self._last_minute_ns[symbol] = minute_ns
        else:
            state.high = max(state.high, price)
            state.low = min(state.low, price)
            state.close = price

        if isinstance(event, TradeEvent):
            state.volume += event.qty
            if event.is_buyer_maker:
                state.taker_sell_volume += event.qty
            else:
                state.taker_buy_volume += event.qty
        else:
            state.volume += 1.0
            imbalance = book.imbalance(symbol, levels=3) or 0.0
            if imbalance >= 0:
                state.taker_buy_volume += max(imbalance, 0.01)
            else:
                state.taker_sell_volume += max(-imbalance, 0.01)

    def _has_required_state(self, book: Book) -> bool:
        for symbol in self.pca_symbols:
            if symbol not in self._bars:
                return False
        for symbol in self.trade_symbols:
            snap = book.snapshot(symbol)
            if snap is None or not snap.is_valid:
                return False
        return True

    def _build_snapshot(self, event: MarketEvent, book: Book):
        timestamp = datetime.fromtimestamp(event.ts_ns / 1_000_000_000, tz=timezone.utc)
        bars = {}
        for symbol in self.pca_symbols:
            state = self._bars[symbol]
            bars[symbol] = self._classes["Bar"](
                timestamp=timestamp,
                symbol=symbol,
                open=state.open,
                high=state.high,
                low=state.low,
                close=state.close,
                volume=max(state.volume, 1e-12),
                taker_buy_volume=state.taker_buy_volume,
                taker_sell_volume=state.taker_sell_volume,
            )

        books = {}
        for symbol in self.trade_symbols:
            snap = book.snapshot(symbol)
            assert snap is not None
            books[symbol] = self._classes["BookSnapshot"](
                timestamp=timestamp,
                symbol=symbol,
                bids=tuple((level.price, level.qty) for level in snap.bids),
                asks=tuple((level.price, level.qty) for level in snap.asks),
            )

        # Real funding rate from the book (populated by the runner from
        # FundingEvents); defaults to 0.0 when no funding feed is present (spot
        # data or mock). Previously hardcoded 0.0, which silently zeroed the
        # funding term of the blend on the futures path.
        funding = {
            symbol: self._classes["FundingRate"](
                timestamp=timestamp,
                symbol=symbol,
                rate=book.funding_rate(symbol),
            )
            for symbol in self.trade_symbols
        }
        return self._classes["MarketSnapshot"](
            timestamp=timestamp,
            bars=bars,
            books=books,
            funding=funding,
            btc_daily=None,
        )

    def _action_from_alpha(self, alpha: float, confirmed: bool, eligible: bool) -> SignalAction:
        if not eligible or not confirmed or abs(alpha) <= self.entry_alpha_threshold:
            return SignalAction.FLAT
        return SignalAction.LONG if alpha > 0 else SignalAction.SHORT

    def _should_emit(self, symbol: str, action: SignalAction, ts_ns: int) -> bool:
        if self._last_action.get(symbol, SignalAction.FLAT) == action:
            return False
        if ts_ns - self._last_signal_ts_ns.get(symbol, 0) < self.cooldown_ns:
            return False
        return True


__all__ = ["CryptoPcaStrategy"]
