"""LiveBroker tests — order submission + user-data fill parsing (no network)."""

from __future__ import annotations

import queue
import unittest

from binance_rest import format_to_increment
from decimal import Decimal

from broker import LiveBroker, parse_user_data_message
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
)


class _FakeRest:
    """Stand-in for BinanceFuturesRest: records calls, no network."""

    ws_base = "wss://test/ws"

    def __init__(self, *, reject: bool = False, reject_cancel: bool = False,
                 step="0.001", tick="0.01", open_orders=None):
        self.reject = reject
        self.reject_cancel = reject_cancel
        self._step = Decimal(step)
        self._tick = Decimal(tick)
        self.placed: list[dict] = []
        self.canceled: list[dict] = []
        self._open_orders = open_orders or []

    def load_exchange_filters(self, symbols):
        self.loaded = list(symbols)

    def format_quantity(self, symbol, qty):
        return format_to_increment(qty, self._step)

    def format_price(self, symbol, price):
        return format_to_increment(price, self._tick)

    def place_order(self, **kwargs):
        if self.reject:
            raise RuntimeError("insufficient margin")
        self.placed.append(kwargs)
        return {"orderId": 12345, "status": "NEW"}

    def cancel_order(self, **kwargs):
        if self.reject_cancel:
            raise RuntimeError("Unknown order sent")
        self.canceled.append(kwargs)
        return {"status": "CANCELED"}

    def get_open_orders(self, symbol=None):
        return self._open_orders


def _order(qty=0.0123, order_type=OrderType.MARKET, price=None, coid="c1"):
    return OrderEvent(
        ts_ns=1, client_order_id=coid, symbol="BTCUSDT", side=Side.BUY,
        order_type=order_type, qty=qty, price=price, tif=TimeInForce.GTC,
        strategy_id="t", account_id="default",
    )


class LiveBrokerSubmitTests(unittest.TestCase):
    def test_market_order_formats_quantity_and_acks(self):
        rest = _FakeRest(step="0.001")
        broker = LiveBroker(rest, queue.Queue(), symbols=("BTCUSDT",))
        result = broker.send(_order(qty=0.0123456))

        self.assertTrue(result.accepted)
        self.assertEqual(rest.placed[0]["quantity"], "0.012")     # rounded DOWN to step
        self.assertEqual(rest.placed[0]["order_type"], "MARKET")
        ack = result.events[0]
        self.assertIsInstance(ack, OrderAckEvent)
        self.assertEqual(ack.exchange_order_id, "12345")

    def test_limit_order_formats_price_and_post_only(self):
        rest = _FakeRest(tick="0.01")
        broker = LiveBroker(rest, queue.Queue(), symbols=("BTCUSDT",), post_only=True)
        result = broker.send(_order(qty=1.0, order_type=OrderType.LIMIT, price=100.12345))

        self.assertTrue(result.accepted)
        self.assertEqual(rest.placed[0]["price"], "100.12")       # rounded to tick
        self.assertEqual(rest.placed[0]["time_in_force"], "GTX")  # post-only

    def test_quantity_rounded_to_zero_is_rejected(self):
        rest = _FakeRest(step="1")            # whole-unit step
        broker = LiveBroker(rest, queue.Queue(), symbols=("BTCUSDT",))
        result = broker.send(_order(qty=0.5))

        self.assertFalse(result.accepted)
        self.assertIsInstance(result.events[0], OrderRejectEvent)
        self.assertEqual(result.events[0].reason, "qty_rounded_to_zero")
        self.assertEqual(rest.placed, [])      # never sent

    def test_exchange_error_is_rejected_not_raised(self):
        broker = LiveBroker(_FakeRest(reject=True), queue.Queue(), symbols=("BTCUSDT",))
        result = broker.send(_order())
        self.assertFalse(result.accepted)
        self.assertIn("exchange_error", result.events[0].reason)

    def test_duplicate_client_order_id_rejected(self):
        broker = LiveBroker(_FakeRest(), queue.Queue(), symbols=("BTCUSDT",))
        broker.send(_order(coid="dup"))
        result = broker.send(_order(coid="dup"))
        self.assertFalse(result.accepted)
        self.assertEqual(result.events[0].reason, "duplicate_client_order_id")


class CancelReplaceTests(unittest.TestCase):
    def _broker(self, **kw):
        return LiveBroker(_FakeRest(**kw), queue.Queue(), symbols=("BTCUSDT",))

    def test_cancel_success_is_empty_and_marks_canceled(self):
        rest = _FakeRest()
        broker = LiveBroker(rest, queue.Queue(), symbols=("BTCUSDT",))
        broker.send(_order(qty=1.0, order_type=OrderType.LIMIT, price=100.0, coid="c1"))
        self.assertEqual(len(broker.open_orders()), 1)

        result = broker.cancel(CancelOrderEvent(
            ts_ns=1, orig_client_order_id="c1", symbol="BTCUSDT"))

        self.assertEqual(result.events, ())            # confirmation comes via stream
        self.assertEqual(rest.canceled[0]["orig_client_order_id"], "c1")
        self.assertEqual(broker.open_orders(), {})     # no longer resting

    def test_cancel_failure_rejects(self):
        broker = self._broker(reject_cancel=True)
        broker.send(_order(qty=1.0, order_type=OrderType.LIMIT, price=100.0, coid="c1"))
        result = broker.cancel(CancelOrderEvent(
            ts_ns=1, orig_client_order_id="c1", symbol="BTCUSDT"))
        self.assertIsInstance(result.events[0], OrderRejectEvent)
        self.assertIn("cancel_error", result.events[0].reason)

    def test_modify_cancels_then_places_replacement(self):
        rest = _FakeRest()
        broker = LiveBroker(rest, queue.Queue(), symbols=("BTCUSDT",))
        broker.send(_order(qty=2.0, order_type=OrderType.LIMIT, price=100.0, coid="c1"))

        result = broker.modify(ModifyOrderEvent(
            ts_ns=1, orig_client_order_id="c1", modify_client_order_id="c2",
            symbol="BTCUSDT", price=101.12345))   # side/qty inherited from original

        self.assertEqual(len(rest.canceled), 1)              # cancelled original
        self.assertEqual(len(rest.placed), 2)                # original + replacement
        self.assertEqual(rest.placed[1]["price"], "101.12")  # new price, tick-rounded
        self.assertEqual(rest.placed[1]["quantity"], "2")    # inherited qty (step 0.001)
        self.assertTrue(any(isinstance(e, OrderAckEvent) for e in result.events))

    def test_modify_aborts_if_cancel_fails(self):
        rest = _FakeRest(reject_cancel=True)
        broker = LiveBroker(rest, queue.Queue(), symbols=("BTCUSDT",))
        broker.send(_order(qty=2.0, order_type=OrderType.LIMIT, price=100.0, coid="c1"))
        result = broker.modify(ModifyOrderEvent(
            ts_ns=1, orig_client_order_id="c1", modify_client_order_id="c2",
            symbol="BTCUSDT", price=101.0))
        self.assertEqual(len(rest.placed), 1)                # replacement NOT placed
        self.assertIsInstance(result.events[0], OrderRejectEvent)

    def test_reconcile_rebuilds_open_orders_from_exchange(self):
        rest = _FakeRest(open_orders=[{
            "clientOrderId": "x1", "orderId": 7, "symbol": "BTCUSDT", "side": "SELL",
            "type": "LIMIT", "price": "100.5", "origQty": "1.0", "executedQty": "0.0",
            "status": "NEW",
        }])
        broker = LiveBroker(rest, queue.Queue(), symbols=("BTCUSDT",))
        broker.reconcile_open_orders()
        self.assertIn("x1", broker.open_orders())
        self.assertEqual(broker.open_orders()["x1"]["side"], Side.SELL)

    def test_stream_fill_updates_tracker(self):
        broker = LiveBroker(_FakeRest(), queue.Queue(), symbols=("BTCUSDT",))
        broker.send(_order(qty=1.0, order_type=OrderType.LIMIT, price=100.0, coid="c1"))
        broker._track_from_event(FillEvent(
            ts_ns=1, client_order_id="c1", exchange_order_id="12345", symbol="BTCUSDT",
            side=Side.BUY, fill_price=100.0, fill_qty=1.0, cum_qty=1.0, leaves_qty=0.0,
            fee=0.0, fee_asset="USDT", status=OrderStatus.FILLED,
            exec_type=ExecutionType.TRADE))
        self.assertEqual(broker.open_orders(), {})           # fully filled -> not resting


class UserDataParseTests(unittest.TestCase):
    def _trade_msg(self, status="FILLED"):
        return {"e": "ORDER_TRADE_UPDATE", "T": 1700, "o": {
            "s": "BTCUSDT", "c": "c1", "i": 999, "S": "BUY", "x": "TRADE", "X": status,
            "l": "0.5", "z": "0.5", "q": "0.5", "L": "30000.0", "n": "0.6", "N": "USDT", "t": 77,
        }}

    def test_trade_message_becomes_fill_event(self):
        events = parse_user_data_message(self._trade_msg())
        self.assertEqual(len(events), 1)
        fill = events[0]
        self.assertIsInstance(fill, FillEvent)
        self.assertEqual(fill.symbol, "BTCUSDT")
        self.assertEqual(fill.side, Side.BUY)
        self.assertAlmostEqual(fill.fill_price, 30000.0)
        self.assertAlmostEqual(fill.fill_qty, 0.5)
        self.assertAlmostEqual(fill.fee, 0.6)
        self.assertEqual(fill.exchange_trade_id, "77")
        self.assertEqual(fill.exec_type, ExecutionType.TRADE)

    def test_new_message_becomes_ack(self):
        msg = {"e": "ORDER_TRADE_UPDATE", "T": 1, "o": {"c": "c1", "i": 1, "S": "BUY", "x": "NEW", "X": "NEW"}}
        events = parse_user_data_message(msg)
        self.assertIsInstance(events[0], OrderAckEvent)

    def test_rejected_message_becomes_reject(self):
        msg = {"e": "ORDER_TRADE_UPDATE", "T": 1, "o": {"c": "c1", "i": 1, "S": "SELL", "x": "REJECTED", "X": "REJECTED"}}
        events = parse_user_data_message(msg)
        self.assertIsInstance(events[0], OrderRejectEvent)
        self.assertIn("rejected", events[0].reason)

    def test_unrelated_message_ignored(self):
        self.assertEqual(parse_user_data_message({"e": "ACCOUNT_UPDATE"}), [])


class FormatIncrementTests(unittest.TestCase):
    def test_rounds_down_to_increment(self):
        self.assertEqual(format_to_increment(0.0123456, Decimal("0.001")), "0.012")
        self.assertEqual(format_to_increment(100.127, Decimal("0.01")), "100.12")
        self.assertEqual(format_to_increment(3.0, Decimal("1")), "3")
        self.assertEqual(format_to_increment(1.5, None), "1.5")   # no increment -> as-is


if __name__ == "__main__":
    unittest.main()
