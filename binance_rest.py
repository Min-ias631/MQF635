"""binance_rest.py —— Binance USD-M Futures 签名 REST 客户端（供 LiveBroker 使用）

职责：
- 加载交易规则（LOT_SIZE stepSize / PRICE_FILTER tickSize），按交易所精度格式化数量/价格
- 下单（MARKET / LIMIT，支持 GTX post-only）
- user-data listenKey 的创建 / 续期 / 关闭

说明：
- 纯 REST + HMAC-SHA256 签名；格式化函数不依赖网络，便于单测
- 凭证从外部注入（main.py 从 settings 读取）；不在本文件硬编码
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from decimal import Decimal, ROUND_DOWN
from typing import Iterable, Optional

import requests

logger = logging.getLogger(__name__)

LIVE_BASE = "https://fapi.binance.com"
TESTNET_BASE = "https://testnet.binancefuture.com"
LIVE_WS = "wss://fstream.binance.com/ws"
TESTNET_WS = "wss://stream.binancefuture.com/ws"


def format_to_increment(value: float, increment: Optional[Decimal]) -> str:
    """Round value DOWN to the exchange increment (tick/step). Pure + testable."""
    decimal_value = Decimal(str(value))
    if increment is not None and increment > 0:
        decimal_value = (decimal_value / increment).to_integral_value(
            rounding=ROUND_DOWN
        ) * increment
    return format(decimal_value.normalize(), "f")


class BinanceFuturesRest:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = True,
        recv_window: int = 5000,
    ) -> None:
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.base = TESTNET_BASE if testnet else LIVE_BASE
        self.ws_base = TESTNET_WS if testnet else LIVE_WS
        self.recv_window = recv_window
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        self._step: dict[str, Decimal] = {}
        self._tick: dict[str, Decimal] = {}

    # ------------------------------------------------------------------ signing

    def _sign(self, params: dict) -> dict:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.recv_window
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    # ----------------------------------------------------------- exchange rules

    def load_exchange_filters(self, symbols: Iterable[str]) -> None:
        """Cache per-symbol stepSize/tickSize from /fapi/v1/exchangeInfo (public)."""
        try:
            r = self.session.get(self.base + "/fapi/v1/exchangeInfo", timeout=10)
            r.raise_for_status()
            info = r.json()
        except Exception:
            logger.exception("Failed to load Binance Futures exchange rules")
            return

        wanted = {s.upper() for s in symbols}
        for symbol_info in info.get("symbols", []):
            symbol = symbol_info["symbol"]
            if wanted and symbol not in wanted:
                continue
            for f in symbol_info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    self._step[symbol] = Decimal(f["stepSize"])
                elif f["filterType"] == "PRICE_FILTER":
                    self._tick[symbol] = Decimal(f["tickSize"])

    def format_quantity(self, symbol: str, quantity: float) -> str:
        return format_to_increment(quantity, self._step.get(symbol.upper()))

    def format_price(self, symbol: str, price: float) -> str:
        return format_to_increment(price, self._tick.get(symbol.upper()))

    # --------------------------------------------------------------- orders

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: str,
        price: Optional[str] = None,
        time_in_force: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        params: dict[str, object] = {
            "symbol": symbol.upper(),
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        r = self.session.post(self.base + "/fapi/v1/order", params=self._sign(params), timeout=10)
        data = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"order rejected ({r.status_code}): {data}")
        return data

    def cancel_order(
        self,
        *,
        symbol: str,
        orig_client_order_id: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> dict:
        params: dict[str, object] = {"symbol": symbol.upper()}
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        elif order_id:
            params["orderId"] = order_id
        else:
            raise ValueError("cancel_order requires orig_client_order_id or order_id")

        r = self.session.delete(self.base + "/fapi/v1/order", params=self._sign(params), timeout=10)
        data = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"cancel rejected ({r.status_code}): {data}")
        return data

    def query_order(
        self,
        *,
        symbol: str,
        orig_client_order_id: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> dict:
        params: dict[str, object] = {"symbol": symbol.upper()}
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        elif order_id:
            params["orderId"] = order_id
        r = self.session.get(self.base + "/fapi/v1/order", params=self._sign(params), timeout=10)
        r.raise_for_status()
        return r.json()

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        params: dict[str, object] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        r = self.session.get(self.base + "/fapi/v1/openOrders", params=self._sign(params), timeout=10)
        r.raise_for_status()
        return r.json()

    # ----------------------------------------------------------- user data key

    def _auth_hint(self, status_code: int) -> str:
        env = "Testnet" if "testnet" in self.base else "Live (mainnet)"
        portal = (
            "https://testnet.binancefuture.com"
            if "testnet" in self.base
            else "https://www.binance.com (Futures API Management)"
        )
        return (
            f"Binance rejected the API key ({status_code}) at {self.base}. "
            f"You are pointed at {env}, which needs keys generated on {portal}. "
            "Testnet and live keys are NOT interchangeable. Check that "
            "BINANCE_API_KEY / BINANCE_SECRET match BINANCE_TESTNET "
            "(1=testnet, 0=mainnet), have Futures enabled, and have no stray quotes/whitespace."
        )

    def validate_credentials(self) -> None:
        """Signed preflight so live startup fails fast with a clear message."""
        r = self.session.get(self.base + "/fapi/v2/balance", params=self._sign({}), timeout=10)
        if r.status_code in (401, 403):
            raise RuntimeError(self._auth_hint(r.status_code))
        if r.status_code != 200:
            raise RuntimeError(f"Binance credential check failed ({r.status_code}): {r.text}")

    def create_listen_key(self) -> str:
        r = self.session.post(self.base + "/fapi/v1/listenKey", timeout=10)
        if r.status_code in (401, 403):
            raise RuntimeError(self._auth_hint(r.status_code))
        r.raise_for_status()
        return r.json()["listenKey"]

    def keepalive_listen_key(self) -> None:
        self.session.put(self.base + "/fapi/v1/listenKey", timeout=10)

    def close_listen_key(self) -> None:
        try:
            self.session.delete(self.base + "/fapi/v1/listenKey", timeout=10)
        except Exception:
            logger.exception("Failed to close listenKey")


__all__ = ["BinanceFuturesRest", "format_to_increment"]
