"""
risk.py —— 风险管理（Pre-Trade Gate + Circuit Breaker）

职责：
- 拦截策略信号 SignalEvent，决定是否放行、下单数量、是否熔断
- 唯一通道：SignalEvent -> risk.check() -> OrderEvent | None

检查项（来自 config.settings）：
- 单笔下单量上限
- 总持仓 / 总敞口上限
- 日内最大亏损（触发 HALT）
- 频率限制（每 symbol 每秒下单次数）
- 连续下单失败（触发 HALT，由 broker/runner 回调 on_order_failure）

说明：
- 不直接调用 broker；只产出 OrderEvent 或 RiskEvent（由 runner 消费）
"""

from __future__ import annotations

import secrets
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Protocol

from book import Book
from config import Settings, settings
from event import (
    OrderEvent,
    OrderType,
    RiskAction,
    RiskEvent,
    Side,
    SignalAction,
    SignalEvent,
    TimeInForce,
)


# ---------------------------------------------------------------------------
# Portfolio 只读接口（portfolio.py 实现后对接）
# ---------------------------------------------------------------------------


class PortfolioReader(Protocol):
    """风控需要的持仓/盈亏只读视图。"""

    def position_qty(self, symbol: str) -> float:
        """当前持仓数量；正=多，负=空。"""
        ...

    def total_exposure_qty(self) -> float:
        """总敞口（通常用各品种 |position| 之和）。"""
        ...

    def daily_pnl(self) -> float:
        """当日盈亏（已实现+未实现）；亏损为负。"""
        ...


@dataclass(slots=True)
class PortfolioStub:
    """测试/开发用简易持仓簿记。"""

    positions: dict[str, float]
    _daily_pnl: float = 0.0

    def position_qty(self, symbol: str) -> float:
        return float(self.positions.get(symbol.upper(), 0.0))

    def total_exposure_qty(self) -> float:
        return sum(abs(q) for q in self.positions.values())

    def daily_pnl(self) -> float:
        return self._daily_pnl


# ---------------------------------------------------------------------------
# 风控结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    order: Optional[OrderEvent]
    risk_event: Optional[RiskEvent] = None
    reject_reason: Optional[str] = None


class RiskState:
    NORMAL = "NORMAL"
    HALTED = "HALTED"
    KILL = "KILL"


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """
    Pre-trade gate + circuit breaker.

    用法（runner）::
        risk = RiskManager()
        result = risk.check(signal, portfolio, book=book)
        if result.risk_event:
            recorder.log(result.risk_event)
        if result.order:
            broker.send(result.order)
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        *,
        account_id: str = "default",
    ) -> None:
        self._cfg = cfg or settings
        self._account_id = account_id

        self._state = RiskState.NORMAL
        self._consecutive_failures = 0

        # per-symbol 下单时间戳（秒），用于频率限制
        self._order_times: dict[str, Deque[float]] = {}

        # runner 可批量取走的风控事件
        self._pending_risk_events: list[RiskEvent] = []

    @property
    def state(self) -> str:
        return self._state

    def pop_risk_events(self) -> list[RiskEvent]:
        events = self._pending_risk_events
        self._pending_risk_events = []
        return events

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def check(
        self,
        signal: SignalEvent,
        portfolio: PortfolioReader,
        *,
        book: Book | None = None,
    ) -> RiskCheckResult:
        """SignalEvent -> OrderEvent | None（策略到执行的唯一通道）。"""
        if self._state == RiskState.KILL:
            return RiskCheckResult(
                order=None,
                reject_reason="kill_switch_active",
            )
        if self._state == RiskState.HALTED:
            return RiskCheckResult(
                order=None,
                reject_reason="risk_halted",
            )

        sym = signal.symbol.upper()

        # 日内最大亏损 -> 立即 HALT
        if portfolio.daily_pnl() <= -abs(self._cfg.daily_max_loss):
            ev = self._emit_halt(
                signal.ts_ns,
                reason=f"daily_max_loss breached: pnl={portfolio.daily_pnl():.4f}",
                source="daily_max_loss",
            )
            return RiskCheckResult(order=None, risk_event=ev, reject_reason=ev.reason)

        # 盘口不可用时不交易（避免 stale book）
        if book is not None:
            snap = book.snapshot(sym)
            if snap is None or not snap.is_valid:
                return RiskCheckResult(
                    order=None,
                    reject_reason="book_not_valid",
                )

        # P1 TODO：频率限制建议放在算出 side/qty 并判定 reduce_only 之后；
        # 仅对开仓/加仓限频，平仓/减仓/kill 清仓应有更高优先级。
        if not self._allow_rate(sym, signal.ts_ns):
            return RiskCheckResult(
                order=None,
                reject_reason="order_rate_limited",
            )

        pos = portfolio.position_qty(sym)
        side, qty = self._signal_to_order(signal, pos)
        if side is None or qty <= 0:
            return RiskCheckResult(order=None, reject_reason="no_order_needed")

        reduce_only = self._is_reduce_only(signal, side, pos, qty)

        if reduce_only:
            # P0：减仓/平仓必须能执行，不能被开仓限额误拦。
            # - 不应用总敞口 cap（否则超限时无法逐步降风险）
            # - 不应用普通 max_single_order_qty（否则 FLAT 可能被截断无法清仓）
            # 仅保证不超过当前可平数量
            qty = min(qty, abs(pos))
        else:
            # 开仓/加仓：应用单笔上限与总敞口上限
            max_single = abs(self._cfg.max_single_order_qty)
            if qty > max_single:
                qty = max_single
            qty = self._cap_qty_by_exposure(sym, side, qty, pos, portfolio)
            if qty <= 0:
                return RiskCheckResult(order=None, reject_reason="max_exposure_reached")

        order = OrderEvent(
            ts_ns=signal.ts_ns,
            client_order_id=self._new_client_order_id(signal),
            symbol=sym,
            side=side,
            order_type=OrderType.MARKET,
            qty=qty,
            price=None,
            tif=TimeInForce.GTC,
            strategy_id=signal.strategy_id,
            account_id=self._account_id,
        )

        self._record_order(sym, signal.ts_ns)
        return RiskCheckResult(order=order)

    # ------------------------------------------------------------------
    # Circuit breaker 回调（broker/runner）
    # ------------------------------------------------------------------

    def on_order_success(self) -> None:
        self._consecutive_failures = 0

    def on_order_failure(self, ts_ns: int, *, reason: str = "order_failed") -> Optional[RiskEvent]:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._cfg.max_consecutive_order_failures:
            return self._emit_halt(
                ts_ns,
                reason=f"{reason} (consecutive={self._consecutive_failures})",
                source="consecutive_order_failures",
            )
        ev = RiskEvent(
            ts_ns=ts_ns,
            action=RiskAction.WARN,
            reason=reason,
            source="order_failure",
        )
        self._pending_risk_events.append(ev)
        return ev

    def resume(self, ts_ns: int) -> RiskEvent:
        """人工恢复交易（从 HALTED -> NORMAL）。"""
        # P2 TODO：生产语义上 KILL 比 HALTED 更重，通常不允许普通 resume 直接恢复；
        # 应要求 force=True 或系统重启，避免误把 kill switch 清掉。
        self._state = RiskState.NORMAL
        self._consecutive_failures = 0
        ev = RiskEvent(
            ts_ns=ts_ns,
            action=RiskAction.RESUME,
            reason="manual_resume",
            source="risk_manager",
        )
        self._pending_risk_events.append(ev)
        return ev

    def kill(self, ts_ns: int, *, reason: str = "manual_kill") -> RiskEvent:
        """Kill switch：拒绝一切新单（runner 应撤单+平仓）。"""
        self._state = RiskState.KILL
        ev = RiskEvent(
            ts_ns=ts_ns,
            action=RiskAction.KILL,
            reason=reason,
            source="risk_manager",
        )
        self._pending_risk_events.append(ev)
        return ev

    # ------------------------------------------------------------------
    # 内部：信号 -> 订单方向/数量
    # ------------------------------------------------------------------

    def _signal_to_order(
        self,
        signal: SignalEvent,
        position_qty: float,
    ) -> tuple[Optional[Side], float]:
        strength = max(0.0, min(1.0, float(signal.strength)))
        base_qty = abs(self._cfg.max_single_order_qty) * strength

        if signal.action == SignalAction.FLAT:
            if abs(position_qty) < 1e-12:
                return None, 0.0
            side = Side.SELL if position_qty > 0 else Side.BUY
            return side, abs(position_qty)

        if signal.action == SignalAction.LONG:
            if position_qty < 0:
                # 先平空
                return Side.BUY, min(abs(position_qty), base_qty)
            return Side.BUY, base_qty

        if signal.action == SignalAction.SHORT:
            if position_qty > 0:
                return Side.SELL, min(position_qty, base_qty)
            return Side.SELL, base_qty

        return None, 0.0

    @staticmethod
    def _is_reduce_only(
        signal: SignalEvent,
        side: Side,
        position_qty: float,
        qty: float,
    ) -> bool:
        """判断订单是否降低当前 symbol 绝对持仓（含 FLAT 平仓）。"""
        if qty <= 0 or abs(position_qty) < 1e-12:
            return False
        if signal.action == SignalAction.FLAT:
            return True
        if position_qty > 0 and side == Side.SELL:
            return True
        if position_qty < 0 and side == Side.BUY:
            return True
        return False

    def _cap_qty_by_exposure(
        self,
        symbol: str,
        side: Side,
        qty: float,
        position_qty: float,
        portfolio: PortfolioReader,
    ) -> float:
        max_total = abs(self._cfg.max_total_position_qty)
        if max_total <= 0:
            return 0.0

        def projected_total(order_qty: float) -> float:
            if side == Side.BUY:
                new_pos = position_qty + order_qty
            else:
                new_pos = position_qty - order_qty
            # 只替换该 symbol 的贡献
            others = portfolio.total_exposure_qty() - abs(position_qty)
            return others + abs(new_pos)

        if projected_total(qty) <= max_total:
            return qty

        # 线性收紧：二分或逐步减小（简单线性扫描足够）
        lo, hi = 0.0, qty
        for _ in range(32):
            mid = (lo + hi) / 2.0
            if projected_total(mid) <= max_total:
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------
    # 内部：频率限制 / 熔断 / id
    # ------------------------------------------------------------------

    def _allow_rate(self, symbol: str, ts_ns: int) -> bool:
        limit = int(self._cfg.max_order_rate_per_symbol)
        if limit <= 0:
            return False
        now = ts_ns / 1e9
        dq = self._order_times.setdefault(symbol, deque())
        window = 1.0
        while dq and (now - dq[0]) > window:
            dq.popleft()
        if len(dq) >= limit:
            return False
        return True

    def _record_order(self, symbol: str, ts_ns: int) -> None:
        now = ts_ns / 1e9
        self._order_times.setdefault(symbol, deque()).append(now)

    def _emit_halt(self, ts_ns: int, *, reason: str, source: str) -> RiskEvent:
        # P2 TODO：RiskEvent 会同时进入 _pending_risk_events，且 check() 也可能通过
        # RiskCheckResult.risk_event 返回；runner 应二选一记录，避免重复落盘。
        self._state = RiskState.HALTED
        ev = RiskEvent(
            ts_ns=ts_ns,
            action=RiskAction.HALT,
            reason=reason,
            source=source,
        )
        self._pending_risk_events.append(ev)
        return ev

    @staticmethod
    def _new_client_order_id(signal: SignalEvent) -> str:
        # P1 TODO：随机后缀并非真正幂等；同一 signal 重入 check() 会生成不同 ID。
        # 更稳方案：ID 含 strategy/symbol/ts/action，且 runner 保证同一 signal 只生成一次 OrderEvent；
        # broker 重试必须复用同一 OrderEvent，而不是重新调用 check()。
        suffix = secrets.token_hex(4)
        return f"{signal.strategy_id}-{signal.ts_ns}-{suffix}"


__all__ = [
    "PortfolioReader",
    "PortfolioStub",
    "RiskCheckResult",
    "RiskManager",
    "RiskState",
]
