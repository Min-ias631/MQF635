"""
portfolio.py —— 持仓与盈亏簿记（Booking）

职责：
- 订阅 FillEvent，维护每个 symbol 的持仓、均价、已实现/未实现盈亏
- 为 risk.py 提供 PortfolioReader 只读视图（position / exposure / daily_pnl）
- 通过 Book 做 mark-to-market 计算浮动盈亏

约定：
- 只有 on_fill() 能改变持仓；策略/风控不能直接改仓位
- runner 在收到 FillEvent 后调用 on_fill()，必要时再 mark_to_market(book)
"""

from __future__ import annotations

from dataclasses import dataclass

from book import Book
from event import FillEvent, Side


@dataclass(slots=True)
class _SymbolPosition:
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl_gross: float = 0.0
    realized_pnl_net: float = 0.0
    fees: float = 0.0
    daily_realized_pnl_gross: float = 0.0
    daily_realized_pnl_net: float = 0.0
    daily_fees: float = 0.0


@dataclass(frozen=True, slots=True)
class PositionView:
    """某个 symbol 的只读持仓视图。"""
    symbol: str
    qty: float
    avg_price: float
    realized_pnl_gross: float
    realized_pnl_net: float
    daily_realized_pnl_gross: float
    daily_realized_pnl_net: float
    unrealized_pnl: float
    fees: float
    daily_fees: float
    mark_price: float | None

    @property
    def realized_pnl(self) -> float:
        """Backward-compatible alias: realized PnL after fees."""
        return self.realized_pnl_net


class Portfolio:
    """
    组合簿记。

    用法（runner）::
        portfolio = Portfolio()
        portfolio.on_fill(fill_event)
        portfolio.mark_to_market(book)
        pnl = portfolio.daily_pnl()
    """

    def __init__(self, *, account_id: str = "default") -> None:
        self._account_id = account_id
        self._positions: dict[str, _SymbolPosition] = {}
        self._marks: dict[str, float] = {}

        # Daily/session totals are separate from lifetime per-symbol fields.
        self._daily_realized_pnl_gross: float = 0.0
        self._daily_realized_pnl_net: float = 0.0
        self._daily_fees: float = 0.0

        # Perp funding cashflows (FundingCost_t in the strategy note): applied at
        # funding settlement events; long pays positive funding, short receives.
        self._daily_funding_pnl: float = 0.0
        self._funding_pnl_total: float = 0.0

        # Fill 去重（exchange_trade_id 优先）
        # P1: this set is bounded by trading-day/session lifecycle via reset_daily(clear_seen=True).
        # For 24/7 production, persist recent trade ids and restore them after restart.
        self._seen_trade_ids: set[str] = set()

    # ------------------------------------------------------------------
    # 写入：仅 FillEvent
    # ------------------------------------------------------------------

    def on_fill(self, fill: FillEvent) -> bool:
        """
        应用成交回报，更新持仓与已实现盈亏。
        返回 False 表示重复 fill（已忽略）。
        """
        if fill.account_id != self._account_id:
            # 多账户预留：当前只处理本 portfolio 的 account
            # P2 TODO: return a richer status enum so runner/record can distinguish
            # account mismatch from duplicate fill without inspecting logs.
            return False

        trade_key = self._trade_dedup_key(fill)
        if trade_key in self._seen_trade_ids:
            return False
        self._seen_trade_ids.add(trade_key)

        sym = fill.symbol.upper()
        st = self._positions.setdefault(sym, _SymbolPosition())

        # P2 TODO: portfolio intentionally allows negative positions for futures/margin.
        # If this system runs spot-only, risk/broker should reject naked shorts before fills.
        signed_qty = fill.fill_qty if fill.side == Side.BUY else -fill.fill_qty
        st.qty, st.avg_price, realized_delta = self._apply_fill(
            st.qty,
            st.avg_price,
            signed_qty,
            fill.fill_price,
        )

        fee = abs(fill.fee)
        net_delta = realized_delta - fee

        st.realized_pnl_gross += realized_delta
        st.realized_pnl_net += net_delta
        st.fees += fee
        st.daily_realized_pnl_gross += realized_delta
        st.daily_realized_pnl_net += net_delta
        st.daily_fees += fee

        self._daily_realized_pnl_gross += realized_delta
        self._daily_realized_pnl_net += net_delta
        self._daily_fees += fee

        # 清理接近零的持仓
        if abs(st.qty) < 1e-12:
            st.qty = 0.0
            st.avg_price = 0.0

        return True

    def apply_funding(self, symbol: str, rate: float, mark_price: float | None) -> float:
        """Apply a perp funding settlement to the current position.

        FundingPnL = -qty * mark * rate: a long position pays positive funding
        (negative PnL) and receives negative funding; shorts are the mirror.
        Returns the PnL delta applied (0.0 when flat or unmarked).
        """
        sym = symbol.upper()
        st = self._positions.get(sym)
        if st is None or abs(st.qty) < 1e-12:
            return 0.0
        mark = mark_price if mark_price is not None else self._marks.get(sym)
        if mark is None:
            return 0.0
        delta = -st.qty * float(mark) * float(rate)
        self._daily_funding_pnl += delta
        self._funding_pnl_total += delta
        return delta

    @property
    def funding_pnl_total(self) -> float:
        return self._funding_pnl_total

    @property
    def daily_funding_pnl(self) -> float:
        return self._daily_funding_pnl

    def mark_to_market(self, book: Book) -> None:
        """用 book 最新价刷新各 symbol 的 mark price（供未实现盈亏计算）。"""
        for sym in list(self._positions.keys()):
            px = book.last_price(sym)
            if px is not None:
                self._marks[sym] = px

    def set_mark_price(self, symbol: str, price: float) -> None:
        """手动设置 mark（测试/无 book 场景）。"""
        self._marks[symbol.upper()] = price

    def reset_daily(self, *, clear_seen: bool = True) -> None:
        """
        日切：重置 daily/session 统计，但保留 lifetime realized PnL/fees。

        clear_seen=True 会清理本地成交去重集合，避免长时间实盘无界增长。
        若 runner 会跨日补单/重放成交，可先落盘最近 trade ids，再延后清理。
        """
        self._daily_realized_pnl_gross = 0.0
        self._daily_realized_pnl_net = 0.0
        self._daily_fees = 0.0
        self._daily_funding_pnl = 0.0
        for st in self._positions.values():
            st.daily_realized_pnl_gross = 0.0
            st.daily_realized_pnl_net = 0.0
            st.daily_fees = 0.0
        if clear_seen:
            self._seen_trade_ids.clear()

    # ------------------------------------------------------------------
    # PortfolioReader（供 risk 使用）
    # ------------------------------------------------------------------

    def position_qty(self, symbol: str) -> float:
        st = self._positions.get(symbol.upper())
        return st.qty if st else 0.0

    def total_exposure_qty(self) -> float:
        # P2 TODO: quantity exposure is only a coarse risk proxy; BTC qty and ETH qty
        # are not directly comparable. Add total_exposure_notional(book) before live trading.
        return sum(abs(st.qty) for st in self._positions.values())

    def daily_pnl(self) -> float:
        """当日盈亏 ≈ 会话已实现 + 资金费 + 当前未实现（未实现依赖 mark_to_market）。"""
        return self._daily_realized_pnl_net + self._daily_funding_pnl + self.unrealized_pnl_total

    # ------------------------------------------------------------------
    # 只读查询
    # ------------------------------------------------------------------

    @property
    def unrealized_pnl_total(self) -> float:
        total = 0.0
        for sym, st in self._positions.items():
            if abs(st.qty) < 1e-12:
                continue
            mark = self._marks.get(sym)
            if mark is None:
                continue
            total += (mark - st.avg_price) * st.qty
        return total

    def get_position(self, symbol: str) -> PositionView | None:
        sym = symbol.upper()
        st = self._positions.get(sym)
        if st is None:
            return None
        mark = self._marks.get(sym)
        unreal = 0.0
        if mark is not None and abs(st.qty) >= 1e-12:
            unreal = (mark - st.avg_price) * st.qty
        return PositionView(
            symbol=sym,
            qty=st.qty,
            avg_price=st.avg_price,
            realized_pnl_gross=st.realized_pnl_gross,
            realized_pnl_net=st.realized_pnl_net,
            daily_realized_pnl_gross=st.daily_realized_pnl_gross,
            daily_realized_pnl_net=st.daily_realized_pnl_net,
            unrealized_pnl=unreal,
            fees=st.fees,
            daily_fees=st.daily_fees,
            mark_price=mark,
        )

    def symbols(self) -> tuple[str, ...]:
        return tuple(self._positions.keys())

    # ------------------------------------------------------------------
    # 内部：成交入账与去重
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_dedup_key(fill: FillEvent) -> str:
        if fill.exchange_trade_id:
            return f"{fill.exchange_order_id}:{fill.exchange_trade_id}"
        # 兜底：无 trade id 时用订单+价格+数量+时间（弱去重）
        return (
            f"{fill.client_order_id}:{fill.fill_price}:{fill.fill_qty}:"
            f"{fill.cum_qty}:{fill.ts_ns}"
        )

    @staticmethod
    def _apply_fill(
        pos: float,
        avg: float,
        signed_qty: float,
        price: float,
    ) -> tuple[float, float, float]:
        """
        根据 signed_qty（买为正、卖为负）更新持仓与均价，返回 (new_pos, new_avg, realized_delta)。
        """
        if abs(signed_qty) < 1e-12:
            return pos, avg, 0.0

        if abs(pos) < 1e-12:
            return signed_qty, price, 0.0

        # 加仓：方向相同
        if pos * signed_qty > 0:
            new_pos = pos + signed_qty
            new_avg = (abs(pos) * avg + abs(signed_qty) * price) / abs(new_pos)
            return new_pos, new_avg, 0.0

        # 减仓 / 平仓 / 反向
        close_qty = min(abs(pos), abs(signed_qty))
        if pos > 0:
            realized = (price - avg) * close_qty
        else:
            realized = (avg - price) * close_qty

        new_pos = pos + signed_qty
        if abs(new_pos) < 1e-12:
            return 0.0, 0.0, realized

        # 反向开仓：新方向用本次成交价作为均价
        if (pos > 0 > new_pos) or (pos < 0 < new_pos):
            return new_pos, price, realized

        return new_pos, avg, realized


__all__ = ["Portfolio", "PositionView"]
