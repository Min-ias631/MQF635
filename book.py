"""
book.py —— 盘口/行情快照维护（纯内存，无 IO）

职责：
- 接收 feed/runner 推送的 MarketEvent，维护每个 symbol 的最新盘口与成交价
- 为 strategies / risk / portfolio 提供只读查询接口

约定：
- 默认由 runner 单线程顺序调用 update()；若未来 feed 独立线程写入，需外加锁
"""

from __future__ import annotations

from dataclasses import dataclass

from event import BookDeltaEvent, BookTickEvent, MarketEvent, PriceLevel, TradeEvent


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """某个 symbol 的只读快照（供策略/风控读取）"""
    ts_ns: int
    symbol: str
    best_bid: float | None
    best_bid_qty: float | None
    best_ask: float | None
    best_ask_qty: float | None
    mid: float | None
    spread: float | None
    last_price: float | None
    last_qty: float | None
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    # 一致性状态：runner/risk 可据此判断盘口是否可用于交易
    last_update_id: int | None = None
    need_resync: bool = False
    is_valid: bool = False


@dataclass(slots=True)
class _SymbolState:
    ts_ns: int = 0
    bids: tuple[PriceLevel, ...] = ()
    asks: tuple[PriceLevel, ...] = ()
    last_price: float | None = None
    last_qty: float | None = None
    # 增量簿一致性相关（Binance depthUpdate 风格）
    last_update_id: int | None = None
    need_resync: bool = False


class Book:
    """
    多品种盘口状态容器。

    用法（runner 主循环）::
        book = Book(symbols=settings.symbols)
        ...
        book.update(market_event)
        snap = book.snapshot("BTCUSDT")
    """

    def __init__(self, symbols: tuple[str, ...] | None = None, *, depth_limit: int = 50) -> None:
        self._symbols = tuple(s.upper() for s in symbols) if symbols else ()
        self._depth_limit = max(1, int(depth_limit))
        self._state: dict[str, _SymbolState] = {
            s: _SymbolState() for s in self._symbols
        }
        # Latest perp funding rate per symbol. Funding is read-only market state a
        # futures flow strategy legitimately needs; Book is the strategy's only
        # read surface, so it is surfaced here (populated by runner from
        # FundingEvent) rather than passed through a side channel.
        self._funding: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 写入（feed / runner）
    # ------------------------------------------------------------------

    def update(self, event: MarketEvent) -> None:
        """根据 MarketEvent 更新内存状态。"""
        if isinstance(event, BookTickEvent):
            self._update_book_tick(event)
        elif isinstance(event, BookDeltaEvent):
            self._update_book_delta(event)
        elif isinstance(event, TradeEvent):
            self._update_trade(event)
        else:
            raise TypeError(f"unsupported market event: {type(event)!r}")

    def _update_book_tick(self, event: BookTickEvent) -> None:
        sym = event.symbol.upper()
        st = self._state.setdefault(sym, _SymbolState())
        st.ts_ns = event.ts_ns
        st.bids = event.bids
        st.asks = event.asks
        if event.last_update_id is not None:
            st.last_update_id = event.last_update_id
            st.need_resync = False

    def _update_trade(self, event: TradeEvent) -> None:
        sym = event.symbol.upper()
        st = self._state.setdefault(sym, _SymbolState())
        st.ts_ns = event.ts_ns
        st.last_price = event.price
        st.last_qty = event.qty

    def _update_book_delta(self, event: BookDeltaEvent) -> None:
        sym = event.symbol.upper()
        st = self._state.setdefault(sym, _SymbolState())

        # 没有快照序列号/还在等待重建时，直接丢弃 delta
        if st.need_resync or st.last_update_id is None:
            return

        # 旧消息 / 重复消息：直接忽略，不能误判为断档
        if event.final_update_id <= st.last_update_id:
            return

        expected = st.last_update_id + 1
        if not (event.first_update_id <= expected <= event.final_update_id):
            # 序列断档：要求 feed 在下一次提供快照前不要依赖这些增量
            st.need_resync = True
            return

        # 应用增量：qty==0 删除档位，否则替换/新增
        bids_map = {lv.price: lv.qty for lv in st.bids}
        asks_map = {lv.price: lv.qty for lv in st.asks}

        for lv in event.bids:
            if lv.qty == 0:
                bids_map.pop(lv.price, None)
            else:
                bids_map[lv.price] = lv.qty
        for lv in event.asks:
            if lv.qty == 0:
                asks_map.pop(lv.price, None)
            else:
                asks_map[lv.price] = lv.qty

        # 恢复排序并裁剪到 depth_limit
        sorted_bids = sorted(bids_map.items(), key=lambda x: x[0], reverse=True)[: self._depth_limit]
        sorted_asks = sorted(asks_map.items(), key=lambda x: x[0])[: self._depth_limit]

        st.bids = tuple(PriceLevel(price=p, qty=q) for p, q in sorted_bids)
        st.asks = tuple(PriceLevel(price=p, qty=q) for p, q in sorted_asks)

        st.ts_ns = event.ts_ns
        st.last_update_id = event.final_update_id

    # ------------------------------------------------------------------
    # 读取（strategy / risk / portfolio）
    # ------------------------------------------------------------------

    def snapshot(self, symbol: str) -> BookSnapshot | None:
        sym = symbol.upper()
        st = self._state.get(sym)
        if st is None or st.ts_ns == 0:
            return None

        best_bid = st.bids[0].price if st.bids else None
        best_bid_qty = st.bids[0].qty if st.bids else None
        best_ask = st.asks[0].price if st.asks else None
        best_ask_qty = st.asks[0].qty if st.asks else None

        mid: float | None = None
        spread: float | None = None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

        has_book = bool(st.bids or st.asks)
        is_valid = (not st.need_resync) and has_book

        return BookSnapshot(
            ts_ns=st.ts_ns,
            symbol=sym,
            best_bid=best_bid,
            best_bid_qty=best_bid_qty,
            best_ask=best_ask,
            best_ask_qty=best_ask_qty,
            mid=mid,
            spread=spread,
            last_price=st.last_price,
            last_qty=st.last_qty,
            bids=st.bids,
            asks=st.asks,
            last_update_id=st.last_update_id,
            need_resync=st.need_resync,
            is_valid=is_valid,
        )

    def mid_price(self, symbol: str) -> float | None:
        snap = self.snapshot(symbol)
        return snap.mid if snap and snap.is_valid else None

    def last_price(self, symbol: str) -> float | None:
        snap = self.snapshot(symbol)
        if not snap or not snap.is_valid:
            return None
        if snap.last_price is not None:
            return snap.last_price
        # 没有逐笔成交时，用 mid 作为估值价
        return snap.mid

    def spread(self, symbol: str) -> float | None:
        snap = self.snapshot(symbol)
        return snap.spread if snap and snap.is_valid else None

    def imbalance(self, symbol: str, levels: int = 1) -> float | None:
        """
        盘口不平衡：(bid_qty - ask_qty) / (bid_qty + ask_qty)，范围约 [-1, 1]。
        levels 表示累加前 N 档深度。
        """
        snap = self.snapshot(symbol)
        if snap is None or not snap.is_valid:
            return None

        n = max(1, levels)
        bid_qty = sum(lv.qty for lv in snap.bids[:n])
        ask_qty = sum(lv.qty for lv in snap.asks[:n])
        denom = bid_qty + ask_qty
        if denom <= 0:
            return None
        return (bid_qty - ask_qty) / denom

    def set_funding(self, symbol: str, rate: float) -> None:
        """Record the latest funding rate for a symbol (runner <- FundingEvent)."""
        self._funding[symbol.upper()] = float(rate)

    def funding_rate(self, symbol: str) -> float:
        """Latest funding rate; 0.0 if none seen (e.g. spot or no funding feed)."""
        return self._funding.get(symbol.upper(), 0.0)

    def symbols(self) -> tuple[str, ...]:
        return tuple(self._state.keys())


__all__ = ["Book", "BookSnapshot"]
