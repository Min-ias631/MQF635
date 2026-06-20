"""
microstructure.py —— 简单盘口不平衡策略

用途：
- 作为正式策略前的系统测试策略
- 验证 feed -> book -> strategy -> risk -> broker -> portfolio 链路

信号：
- imbalance >= entry_threshold: LONG
- imbalance <= -entry_threshold: SHORT
- abs(imbalance) <= exit_threshold: FLAT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from book import Book
from event import MarketEvent, SignalAction, SignalEvent, freeze_meta


@dataclass(slots=True)
class _SymbolSignalState:
    last_action: SignalAction = SignalAction.FLAT
    last_signal_ts_ns: int = 0


@dataclass(slots=True)
class MicrostructureImbalanceStrategy:
    """
    Top-N order book imbalance strategy.

    这是一个测试策略，不代表稳定 alpha。它故意保持简单，方便观察系统行为。
    """

    strategy_id: str = "microstructure_imbalance"
    levels: int = 3
    entry_threshold: float = 0.35
    exit_threshold: float = 0.10
    max_spread_bps: float = 8.0
    cooldown_ns: int = 1_000_000_000
    strength: float = 1.0
    _state: dict[str, _SymbolSignalState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.levels <= 0:
            raise ValueError("levels must be positive")
        if not 0.0 <= self.exit_threshold <= self.entry_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= exit <= entry <= 1")
        if self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive")
        if self.cooldown_ns < 0:
            raise ValueError("cooldown_ns must be non-negative")
        self.strength = max(0.0, min(1.0, float(self.strength)))

    def on_market(self, event: MarketEvent, book: Book) -> list[SignalEvent]:
        sym = event.symbol.upper()
        snap = book.snapshot(sym)
        if snap is None or not snap.is_valid:
            return []
        if snap.mid is None or snap.mid <= 0 or snap.spread is None:
            return []

        spread_bps = snap.spread / snap.mid * 10_000.0
        if spread_bps > self.max_spread_bps:
            return []

        imbalance = book.imbalance(sym, self.levels)
        if imbalance is None:
            return []

        action = self._decide_action(imbalance)
        if action is None:
            return []

        st = self._state.setdefault(sym, _SymbolSignalState())
        if action == st.last_action:
            return []
        if event.ts_ns - st.last_signal_ts_ns < self.cooldown_ns:
            return []

        st.last_action = action
        st.last_signal_ts_ns = event.ts_ns

        return [
            SignalEvent(
                ts_ns=event.ts_ns,
                symbol=sym,
                strategy_id=self.strategy_id,
                action=action,
                strength=self.strength,
                meta=freeze_meta(
                    {
                        "imbalance": imbalance,
                        "spread_bps": spread_bps,
                        "levels": self.levels,
                    }
                ),
            )
        ]

    def _decide_action(self, imbalance: float) -> SignalAction | None:
        if imbalance >= self.entry_threshold:
            return SignalAction.LONG
        if imbalance <= -self.entry_threshold:
            return SignalAction.SHORT
        if abs(imbalance) <= self.exit_threshold:
            return SignalAction.FLAT
        return None


__all__ = ["MicrostructureImbalanceStrategy"]
