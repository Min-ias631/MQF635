"""
runner.py —— 主事件循环（系统心脏）

职责：
- 从事件队列消费 MarketEvent / FillEvent / RiskEvent 等
- 串联 feed -> book -> strategy -> risk -> broker -> portfolio -> record
- 只有 runner 知道所有模块；其他模块保持松耦合
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

from book import Book
from broker import Broker, BrokerEvent
from event import (
    BookDeltaEvent,
    BookTickEvent,
    CancelOrderEvent,
    FillEvent,
    FundingEvent,
    MarketEvent,
    ModifyOrderEvent,
    OrderAckEvent,
    OrderRejectEvent,
    RiskEvent,
    TradeEvent,
)
from feed import MarketDataFeed
from portfolio import Portfolio
from record import EventRecorder
from risk import RiskManager
from status import StatusStore
from strategy_base import Strategy


@dataclass(frozen=True, slots=True)
class RunnerStats:
    market_events: int = 0
    signals: int = 0
    orders: int = 0
    fills: int = 0
    rejects: int = 0
    risk_events: int = 0
    broker_events: int = 0


@dataclass(slots=True)
class _MutableStats:
    market_events: int = 0
    signals: int = 0
    orders: int = 0
    fills: int = 0
    rejects: int = 0
    risk_events: int = 0
    broker_events: int = 0

    def snapshot(self) -> RunnerStats:
        return RunnerStats(
            market_events=self.market_events,
            signals=self.signals,
            orders=self.orders,
            fills=self.fills,
            rejects=self.rejects,
            risk_events=self.risk_events,
            broker_events=self.broker_events,
        )


class Runner:
    """
    同步事件循环。

    对于单元测试/回测，可以不断调用 run_once()。
    对于 paper/live，可由 main.py 启动 feed 后调用 run_forever()。
    """

    def __init__(
        self,
        *,
        event_queue: "queue.Queue[object]",
        book: Book,
        strategy: Strategy,
        risk: RiskManager,
        broker: Broker,
        portfolio: Portfolio,
        recorder: EventRecorder | None = None,
        feed: MarketDataFeed | None = None,
        status_store: StatusStore | None = None,
    ) -> None:
        self._queue = event_queue
        self._book = book
        self._strategy = strategy
        self._risk = risk
        self._broker = broker
        self._portfolio = portfolio
        self._recorder = recorder
        self._feed = feed
        self._status_store = status_store
        self._running = False
        self._stats = _MutableStats()

    @property
    def stats(self) -> RunnerStats:
        return self._stats.snapshot()

    def start(self) -> None:
        self._running = True
        if self._recorder is not None:
            self._recorder.start()
        # Live brokers run an async user-data stream that injects fills into the
        # queue; Sim/Paper brokers have no start() and are skipped.
        if hasattr(self._broker, "start"):
            self._broker.start()
        if self._feed is not None:
            self._feed.start()

    def stop(self) -> None:
        self._running = False
        if self._feed is not None:
            self._feed.stop()
        if hasattr(self._broker, "stop"):
            self._broker.stop()
        if self._recorder is not None:
            self._recorder.stop()

    def run_forever(self, *, timeout: float = 1.0) -> None:
        self.start()
        try:
            while self._running:
                self.run_once(timeout=timeout)
        finally:
            self.stop()

    def run_once(self, *, timeout: float = 0.0) -> bool:
        """
        处理一个队列事件。

        返回 True 表示处理了事件；False 表示队列为空。
        """
        try:
            event = self._queue.get(timeout=timeout)
        except queue.Empty:
            return False

        self.handle_event(event)
        return True

    def drain(self, *, max_events: int | None = None) -> int:
        """处理当前队列里的事件，适合测试/同步回测。"""
        processed = 0
        while max_events is None or processed < max_events:
            if not self.run_once(timeout=0.0):
                break
            processed += 1
        return processed

    def handle_event(self, event: object) -> None:
        self._record(event)
        self._status_event(event)

        if self._is_market_event(event):
            self._handle_market(event)
            self._status_runtime()
            return
        if isinstance(event, FundingEvent):
            # Funding is not a tradable market event; it updates the read-only
            # funding state the strategy consumes via Book (no strategy call),
            # and settles the funding cashflow on any open position (PDF cost
            # model: FundingCost is charged at actual settlement timestamps,
            # which is when the exchange publishes FundingEvents).
            self._book.set_funding(event.symbol, event.funding_rate)
            self._portfolio.apply_funding(
                event.symbol,
                event.funding_rate,
                self._book.last_price(event.symbol),
            )
            self._status_runtime()
            return
        if isinstance(event, (CancelOrderEvent, ModifyOrderEvent)):
            self._handle_order_amendment(event)
            self._status_runtime()
            return
        if isinstance(event, FillEvent):
            self._handle_fill(event)
            self._status_runtime()
            return
        if isinstance(event, OrderRejectEvent):
            self._handle_reject(event)
            self._status_runtime()
            return
        if isinstance(event, RiskEvent):
            self._stats.risk_events += 1
            self._status_runtime()
            return
        if isinstance(event, OrderAckEvent):
            self._stats.broker_events += 1
            self._status_runtime()
            return

    def _handle_market(self, event: MarketEvent) -> None:
        self._stats.market_events += 1
        self._book.update(event)
        self._portfolio.mark_to_market(self._book)

        signals = self._strategy.on_market(event, self._book)
        for signal in signals:
            self._record(signal)
            self._status_event(signal)
            self._stats.signals += 1
            result = self._risk.check(signal, self._portfolio, book=self._book)
            if result.risk_event is not None:
                self._record(result.risk_event)
                self._status_event(result.risk_event)
                self._stats.risk_events += 1
            if result.order is None:
                continue

            self._record(result.order)
            self._status_event(result.order)
            self._stats.orders += 1
            broker_result = self._broker.send(result.order, book=self._book)
            for broker_event in broker_result.events:
                self._dispatch_broker_event(broker_event)

    def _handle_order_amendment(self, event: CancelOrderEvent | ModifyOrderEvent) -> None:
        # All order flow stays inside the broker boundary: cancel/replace intents
        # are routed to the broker, which returns ack/reject/fill events.
        self._record(event)
        self._status_event(event)
        if isinstance(event, CancelOrderEvent):
            result = self._broker.cancel(event)
        else:
            result = self._broker.modify(event)
        for broker_event in result.events:
            self._dispatch_broker_event(broker_event)

    def _handle_fill(self, event: FillEvent) -> None:
        if self._portfolio.on_fill(event):
            self._portfolio.mark_to_market(self._book)
            self._risk.on_order_success()
            self._stats.fills += 1

    def _handle_reject(self, event: OrderRejectEvent) -> None:
        self._stats.rejects += 1
        ev = self._risk.on_order_failure(event.ts_ns, reason=event.reason)
        if ev is not None:
            self._record(ev)
            self._stats.risk_events += 1

    def _dispatch_broker_event(self, event: BrokerEvent) -> None:
        self._record(event)
        self._status_event(event)
        self._stats.broker_events += 1
        if isinstance(event, FillEvent):
            self._handle_fill(event)
        elif isinstance(event, OrderRejectEvent):
            self._handle_reject(event)

    def _record(self, event: object) -> None:
        if self._recorder is not None:
            self._recorder.log(event)

    def _status_event(self, event: object) -> None:
        if self._status_store is not None:
            self._status_store.on_event(event)

    def _status_runtime(self) -> None:
        if self._status_store is not None:
            self._status_store.update_runtime(
                book=self._book,
                portfolio=self._portfolio,
                risk=self._risk,
                stats=self.stats,
            )

    @staticmethod
    def _is_market_event(event: object) -> bool:
        return isinstance(event, (BookTickEvent, BookDeltaEvent, TradeEvent))


__all__ = ["Runner", "RunnerStats", "Strategy"]
