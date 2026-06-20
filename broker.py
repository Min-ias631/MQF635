"""
broker.py —— 执行接口（Sim/Paper）

职责：
- 接收已通过风控的 OrderEvent
- 产出 OrderAckEvent / FillEvent / OrderRejectEvent
- 不直接修改 Portfolio；成交回报由 runner 分发给 portfolio

说明：
- SimBroker: 用当前 Book 立即撮合，适合回测/单元测试
- PaperBroker: 只做本地确认，不产生真实成交，适合实盘行情 dry-run
- LiveBroker: 后续对接 Binance REST + User Data WS 时再实现
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Optional

from book import Book
from event import (
    CancelOrderEvent,
    ExecutionType,
    FillEvent,
    ModifyOrderEvent,
    OrderAckEvent,
    OrderEvent,
    OrderRejectEvent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    now_ns,
)

logger = logging.getLogger(__name__)


_BINANCE_STATUS = {
    "NEW": OrderStatus.NEW,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
}


def parse_user_data_message(message: dict, *, account_id: str = "default") -> list:
    """Convert a Binance futures ORDER_TRADE_UPDATE into our execution events.

    Pure / network-free (testable): returns FillEvent on TRADE, OrderAckEvent on
    NEW, OrderRejectEvent on REJECTED/EXPIRED; [] for anything else.
    """
    if message.get("e") != "ORDER_TRADE_UPDATE":
        return []
    o = message.get("o", {})
    ts_ns = int(message.get("T") or message.get("E") or 0) * 1_000_000
    client_id = o.get("c", "")
    exchange_order_id = str(o.get("i", ""))
    symbol = o.get("s", "")
    side = Side.BUY if o.get("S") == "BUY" else Side.SELL
    exec_type = o.get("x")
    order_status = o.get("X")

    if exec_type == "TRADE":
        last_qty = float(o.get("l", 0) or 0)
        cum_qty = float(o.get("z", 0) or 0)
        order_qty = float(o.get("q", 0) or 0)
        return [FillEvent(
            ts_ns=ts_ns,
            client_order_id=client_id,
            exchange_order_id=exchange_order_id,
            symbol=symbol,
            side=side,
            fill_price=float(o.get("L", 0) or 0),
            fill_qty=last_qty,
            cum_qty=cum_qty,
            leaves_qty=max(order_qty - cum_qty, 0.0),
            fee=float(o.get("n", 0) or 0),
            fee_asset=o.get("N") or "",
            status=_BINANCE_STATUS.get(order_status, OrderStatus.PARTIALLY_FILLED),
            exec_type=ExecutionType.TRADE,
            exchange_trade_id=str(o.get("t", "")) or None,
            account_id=account_id,
        )]
    if exec_type == "NEW":
        return [OrderAckEvent(
            ts_ns=ts_ns,
            client_order_id=client_id,
            exchange_order_id=exchange_order_id,
            status=OrderStatus.NEW,
            account_id=account_id,
        )]
    if exec_type == "CANCELED" or order_status == "CANCELED":
        return [OrderAckEvent(
            ts_ns=ts_ns,
            client_order_id=client_id,
            exchange_order_id=exchange_order_id,
            status=OrderStatus.CANCELED,
            account_id=account_id,
        )]
    if order_status in ("REJECTED", "EXPIRED"):
        return [OrderRejectEvent(
            ts_ns=ts_ns,
            client_order_id=client_id,
            reason=f"exchange_{str(order_status).lower()}",
            account_id=account_id,
        )]
    return []


BrokerEvent = OrderAckEvent | FillEvent | OrderRejectEvent


@dataclass(frozen=True, slots=True)
class BrokerResult:
    """一次 broker 操作产生的回报事件。"""

    events: tuple[BrokerEvent, ...]

    @property
    def accepted(self) -> bool:
        return any(isinstance(ev, OrderAckEvent) for ev in self.events) and not any(
            isinstance(ev, OrderRejectEvent) for ev in self.events
        )

    @property
    def fills(self) -> tuple[FillEvent, ...]:
        return tuple(ev for ev in self.events if isinstance(ev, FillEvent))


class Broker(ABC):
    """执行层抽象。runner 只依赖这个接口。"""

    @abstractmethod
    def send(self, order: OrderEvent, *, book: Book | None = None) -> BrokerResult:
        ...

    def cancel(self, event: CancelOrderEvent) -> BrokerResult:
        reject = OrderRejectEvent(
            ts_ns=event.ts_ns,
            client_order_id=event.orig_client_order_id,
            reason="cancel_not_supported",
            account_id=event.account_id,
        )
        return BrokerResult((reject,))

    def modify(self, event: ModifyOrderEvent) -> BrokerResult:
        reject = OrderRejectEvent(
            ts_ns=event.ts_ns,
            client_order_id=event.orig_client_order_id,
            reason="modify_not_supported",
            account_id=event.account_id,
        )
        return BrokerResult((reject,))


class SimBroker(Broker):
    """
    简单撮合 broker。

    MARKET 单：
    - BUY 使用 best ask；若无 ask 则用 last/mid
    - SELL 使用 best bid；若无 bid 则用 last/mid

    LIMIT 单：
    - BUY price >= best ask 时成交
    - SELL price <= best bid 时成交
    - 否则只 ack，不挂单管理（后续 OrderManager 再接）
    """

    def __init__(self, *, fee_rate: float = 0.0, quote_asset: str = "USDT") -> None:
        self._fee_rate = max(0.0, float(fee_rate))
        self._quote_asset = quote_asset
        self._seq = itertools.count(1)
        self._seen_client_order_ids: set[str] = set()

    def send(self, order: OrderEvent, *, book: Book | None = None) -> BrokerResult:
        if order.client_order_id in self._seen_client_order_ids:
            return BrokerResult(
                (
                    OrderRejectEvent(
                        ts_ns=now_ns(),
                        client_order_id=order.client_order_id,
                        reason="duplicate_client_order_id",
                        account_id=order.account_id,
                    ),
                )
            )
        self._seen_client_order_ids.add(order.client_order_id)

        reject_reason = self._validate_order(order)
        if reject_reason:
            return BrokerResult(
                (
                    OrderRejectEvent(
                        ts_ns=now_ns(),
                        client_order_id=order.client_order_id,
                        reason=reject_reason,
                        account_id=order.account_id,
                    ),
                )
            )

        exchange_order_id = self._new_exchange_order_id()
        ack = OrderAckEvent(
            ts_ns=now_ns(),
            client_order_id=order.client_order_id,
            exchange_order_id=exchange_order_id,
            status=OrderStatus.NEW,
            account_id=order.account_id,
        )

        if book is None:
            reject = OrderRejectEvent(
                ts_ns=now_ns(),
                client_order_id=order.client_order_id,
                reason="book_required_for_sim_fill",
                account_id=order.account_id,
            )
            return BrokerResult((ack, reject))

        fill_price = self._match_price(order, book)
        if fill_price is None:
            return BrokerResult((ack,))

        notional = abs(order.qty * fill_price)
        fill = FillEvent(
            ts_ns=now_ns(),
            client_order_id=order.client_order_id,
            exchange_order_id=exchange_order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=order.qty,
            cum_qty=order.qty,
            leaves_qty=0.0,
            fee=notional * self._fee_rate,
            fee_asset=self._quote_asset,
            status=OrderStatus.FILLED,
            exec_type=ExecutionType.TRADE,
            exchange_trade_id=self._new_trade_id(exchange_order_id),
            account_id=order.account_id,
        )
        return BrokerResult((ack, fill))

    @staticmethod
    def _validate_order(order: OrderEvent) -> str | None:
        if order.qty <= 0:
            return "invalid_qty"
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.price is None:
            return "limit_price_required"
        if order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT) and order.stop_price is None:
            return "stop_price_required"
        if order.order_type == OrderType.MARKET and order.tif != TimeInForce.GTC:
            # Binance ignores tif for MARKET; keep this permissive but explicit for future review.
            return None
        return None

    def _match_price(self, order: OrderEvent, book: Book) -> float | None:
        snap = book.snapshot(order.symbol)
        if snap is None or not snap.is_valid:
            return None

        if order.order_type == OrderType.MARKET:
            if order.side == Side.BUY:
                return snap.best_ask or snap.last_price or snap.mid
            return snap.best_bid or snap.last_price or snap.mid

        if order.order_type == OrderType.LIMIT:
            if order.price is None:
                return None
            if order.side == Side.BUY and snap.best_ask is not None and order.price >= snap.best_ask:
                return snap.best_ask
            if order.side == Side.SELL and snap.best_bid is not None and order.price <= snap.best_bid:
                return snap.best_bid
            return None

        return None

    def _new_exchange_order_id(self) -> str:
        return f"SIM-O-{next(self._seq)}"

    def _new_trade_id(self, exchange_order_id: str) -> str:
        return f"{exchange_order_id}-T"


class PaperBroker(Broker):
    """
    Paper broker: 确认订单，但不产生成交。

    适合“实盘行情 + 不真实发单”的 dry-run。后续可以扩展为继承 SimBroker，
    用盘口模拟成交；当前版本保持不改仓位的保守语义。
    """

    def __init__(self) -> None:
        self._seq = itertools.count(1)
        self._seen_client_order_ids: set[str] = set()

    def send(self, order: OrderEvent, *, book: Book | None = None) -> BrokerResult:
        if order.client_order_id in self._seen_client_order_ids:
            return BrokerResult(
                (
                    OrderRejectEvent(
                        ts_ns=now_ns(),
                        client_order_id=order.client_order_id,
                        reason="duplicate_client_order_id",
                        account_id=order.account_id,
                    ),
                )
            )
        self._seen_client_order_ids.add(order.client_order_id)

        if order.qty <= 0:
            return BrokerResult(
                (
                    OrderRejectEvent(
                        ts_ns=now_ns(),
                        client_order_id=order.client_order_id,
                        reason="invalid_qty",
                        account_id=order.account_id,
                    ),
                )
            )

        ack = OrderAckEvent(
            ts_ns=now_ns(),
            client_order_id=order.client_order_id,
            exchange_order_id=f"PAPER-O-{next(self._seq)}",
            status=OrderStatus.NEW,
            account_id=order.account_id,
        )
        return BrokerResult((ack,))


class LiveBroker(Broker):
    """Binance USD-M Futures live broker.

    Mirrors the reference gateway: orders are submitted via signed REST with
    per-symbol tick/step rounding (and GTX post-only for maker limits), while
    fills arrive asynchronously from the user-data WebSocket stream and are
    injected into the runner's event queue as FillEvent/OrderAck/OrderReject.
    ``send`` returns only the immediate ack/reject; fills are NOT synchronous.

    The REST client is injected (duck-typed) so this stays import-light and the
    order/parse logic is unit-testable without network.
    """

    def __init__(
        self,
        rest: Any,
        event_queue: "Any | None" = None,
        *,
        account_id: str = "default",
        symbols: Iterable[str] = (),
        post_only: bool = False,
    ) -> None:
        self._rest = rest
        self._queue = event_queue
        self._account_id = account_id
        self._symbols = tuple(s.upper() for s in symbols)
        self._post_only = post_only
        self._seen_client_order_ids: set[str] = set()
        self._filters_loaded = False
        self._listen_key: Optional[str] = None
        self._ws = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # Lightweight order state machine: client_order_id -> live order record.
        self._orders: dict[str, dict] = {}
        self._orders_lock = threading.Lock()

    # ----------------------------------------------------------- order submit

    def send(self, order: OrderEvent, *, book: Book | None = None) -> BrokerResult:
        if order.client_order_id in self._seen_client_order_ids:
            return self._reject(order, "duplicate_client_order_id")
        self._seen_client_order_ids.add(order.client_order_id)

        if not self._filters_loaded:
            self._rest.load_exchange_filters(self._symbols or (order.symbol,))
            self._filters_loaded = True

        quantity = self._rest.format_quantity(order.symbol, order.qty)
        if Decimal(quantity) <= 0:
            return self._reject(order, "qty_rounded_to_zero")

        kwargs: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": quantity,
            "client_order_id": order.client_order_id,
        }
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if order.price is None:
                return self._reject(order, "limit_price_required")
            kwargs["price"] = self._rest.format_price(order.symbol, order.price)
            kwargs["time_in_force"] = "GTX" if self._post_only else (
                order.tif.value if order.tif else "GTC"
            )

        try:
            response = self._rest.place_order(**kwargs)
        except Exception as exc:  # network / exchange rejection
            logger.exception("[LiveBroker] order failed")
            return self._reject(order, f"exchange_error: {exc}")

        exchange_order_id = str(response.get("orderId", ""))
        with self._orders_lock:
            self._orders[order.client_order_id] = {
                "exchange_order_id": exchange_order_id,
                "symbol": order.symbol.upper(),
                "side": order.side,
                "order_type": order.order_type,
                "price": order.price,
                "qty": float(order.qty),
                "filled": 0.0,
                "status": OrderStatus.NEW,
            }
        ack = OrderAckEvent(
            ts_ns=now_ns(),
            client_order_id=order.client_order_id,
            exchange_order_id=exchange_order_id,
            status=OrderStatus.NEW,
            account_id=order.account_id,
        )
        return BrokerResult((ack,))  # fills arrive async via the user-data stream

    # ------------------------------------------------------------ cancel / replace

    def cancel(self, event: CancelOrderEvent) -> BrokerResult:
        """Cancel a resting order by its client_order_id. The CANCELED confirmation
        arrives asynchronously via the user-data stream, so this returns empty on
        success and a reject on failure."""
        client_id = event.orig_client_order_id
        symbol = (event.symbol or self._order_symbol(client_id) or "").upper()
        if not client_id or not symbol:
            return BrokerResult((self._cancel_reject(event, "missing_order_reference"),))
        try:
            self._rest.cancel_order(symbol=symbol, orig_client_order_id=client_id)
        except Exception as exc:
            logger.exception("[LiveBroker] cancel failed")
            return BrokerResult((self._cancel_reject(event, f"cancel_error: {exc}"),))
        with self._orders_lock:
            if client_id in self._orders:
                self._orders[client_id]["status"] = OrderStatus.CANCELED
        return BrokerResult(())

    def modify(self, event: ModifyOrderEvent) -> BrokerResult:
        """Replace a resting order: cancel the original, then place a new order.

        Fields left None on the ModifyOrderEvent inherit from the original order
        (tracked from the prior submission), matching exchange replace semantics.
        """
        original = self._order_record(event.orig_client_order_id)
        cancel_result = self.cancel(CancelOrderEvent(
            ts_ns=event.ts_ns,
            orig_client_order_id=event.orig_client_order_id,
            symbol=event.symbol or (original or {}).get("symbol", ""),
            account_id=event.account_id,
        ))
        if any(isinstance(ev, OrderRejectEvent) for ev in cancel_result.events):
            return cancel_result  # could not cancel -> do not place replacement

        side = event.side or (original or {}).get("side")
        qty = event.qty if event.qty is not None else (original or {}).get("qty")
        price = event.price if event.price is not None else (original or {}).get("price")
        if side is None or qty is None:
            return BrokerResult(cancel_result.events + (self._cancel_reject(
                event, "replace_missing_side_or_qty"),))

        new_order = OrderEvent(
            ts_ns=event.ts_ns,
            client_order_id=event.modify_client_order_id or f"replace-{now_ns()}",
            symbol=event.symbol or (original or {}).get("symbol", ""),
            side=side,
            order_type=OrderType.LIMIT if price is not None else OrderType.MARKET,
            qty=qty,
            price=price,
            stop_price=event.stop_price,
            tif=event.tif or TimeInForce.GTC,
            strategy_id=event.strategy_id,
            account_id=event.account_id,
        )
        send_result = self.send(new_order)
        return BrokerResult(tuple(cancel_result.events) + tuple(send_result.events))

    def open_orders(self) -> dict[str, dict]:
        with self._orders_lock:
            return {
                cid: dict(rec) for cid, rec in self._orders.items()
                if rec["status"] in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
            }

    def reconcile_open_orders(self) -> None:
        """Rebuild the local order book from the exchange (call after (re)connect)."""
        try:
            remote = self._rest.get_open_orders()
        except Exception:
            logger.exception("[LiveBroker] reconcile_open_orders failed")
            return
        with self._orders_lock:
            for o in remote:
                client_id = o.get("clientOrderId", "")
                if not client_id:
                    continue
                self._orders[client_id] = {
                    "exchange_order_id": str(o.get("orderId", "")),
                    "symbol": o.get("symbol", ""),
                    "side": Side.BUY if o.get("side") == "BUY" else Side.SELL,
                    "order_type": o.get("type", ""),
                    "price": float(o.get("price", 0) or 0),
                    "qty": float(o.get("origQty", 0) or 0),
                    "filled": float(o.get("executedQty", 0) or 0),
                    "status": _BINANCE_STATUS.get(o.get("status", ""), OrderStatus.NEW),
                }

    def _order_record(self, client_id: Optional[str]) -> Optional[dict]:
        if not client_id:
            return None
        with self._orders_lock:
            rec = self._orders.get(client_id)
            return dict(rec) if rec else None

    def _order_symbol(self, client_id: Optional[str]) -> Optional[str]:
        rec = self._order_record(client_id)
        return rec.get("symbol") if rec else None

    def _track_from_event(self, event) -> None:
        """Update the local order state machine from a user-data execution event."""
        client_id = getattr(event, "client_order_id", "")
        if not client_id:
            return
        with self._orders_lock:
            rec = self._orders.get(client_id)
            if rec is None:
                return
            if isinstance(event, FillEvent):
                rec["filled"] = event.cum_qty
                rec["status"] = event.status
                if event.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
                    rec["status"] = event.status
            elif isinstance(event, OrderAckEvent):
                rec["status"] = event.status
            elif isinstance(event, OrderRejectEvent):
                rec["status"] = OrderStatus.REJECTED

    def _reject(self, order: OrderEvent, reason: str) -> BrokerResult:
        return BrokerResult((OrderRejectEvent(
            ts_ns=now_ns(),
            client_order_id=order.client_order_id,
            reason=reason,
            account_id=order.account_id,
        ),))

    def _cancel_reject(self, event, reason: str) -> OrderRejectEvent:
        return OrderRejectEvent(
            ts_ns=now_ns(),
            client_order_id=getattr(event, "orig_client_order_id", "") or "",
            reason=reason,
            account_id=getattr(event, "account_id", "default"),
        )

    # ---------------------------------------------------- user-data lifecycle

    def start(self) -> None:
        if self._queue is None:
            logger.warning("[LiveBroker] no event_queue set; fills will be dropped")
        if hasattr(self._rest, "validate_credentials"):
            self._rest.validate_credentials()   # fail fast with a clear auth message
        self._listen_key = self._rest.create_listen_key()
        self.reconcile_open_orders()      # rebuild local order book from exchange
        self._stop.clear()
        for target, name in ((self._run_ws, "user-data-ws"),
                             (self._keepalive_loop, "listenkey-keepalive")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info("[LiveBroker] user-data stream started")

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._listen_key is not None:
            self._rest.close_listen_key()

    def _keepalive_loop(self) -> None:
        while not self._stop.wait(1800):  # Binance listenKey expires after 60 min
            try:
                self._rest.keepalive_listen_key()
            except Exception:
                logger.exception("[LiveBroker] listenKey keepalive failed")

    def _run_ws(self) -> None:
        websocket = _load_websocket_module()
        url = f"{self._rest.ws_base}/{self._listen_key}"
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=lambda *_a: None,
                    on_close=lambda *_a: None,
                )
                self._ws.run_forever(ping_interval=180, ping_timeout=10)
            except Exception:
                logger.exception("[LiveBroker] user-data stream crashed; reconnecting")
            if self._stop.is_set():
                break
            time.sleep(3)

    def _on_message(self, _ws, raw: str) -> None:
        try:
            message = json.loads(raw)
        except Exception:
            return
        for event in parse_user_data_message(message, account_id=self._account_id):
            self._track_from_event(event)      # update local order state machine
            if self._queue is not None:
                try:
                    self._queue.put_nowait(event)
                except Exception:
                    logger.warning("[LiveBroker] event queue full; dropped %s", type(event).__name__)


def _load_websocket_module():
    try:
        import websocket  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "LiveBroker user-data stream requires websocket-client. "
            "Install with: python -m pip install websocket-client"
        ) from e
    return websocket


def iter_broker_events(result: BrokerResult) -> Iterable[BrokerEvent]:
    return iter(result.events)


__all__ = [
    "Broker",
    "BrokerEvent",
    "BrokerResult",
    "LiveBroker",
    "PaperBroker",
    "SimBroker",
    "iter_broker_events",
    "parse_user_data_message",
]
