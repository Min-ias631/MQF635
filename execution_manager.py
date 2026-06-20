"""
execution_manager.py
Handles real order placement on Binance Futures Testnet.
Uses MARKET orders for reliable fills on Testnet.
"""

import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone

TESTNET_BASE = "https://testnet.binancefuture.com"
LIVE_BASE    = "https://fapi.binance.com"


class ExecutionManager:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.base       = TESTNET_BASE if testnet else LIVE_BASE
        self.api_key    = api_key
        self.api_secret = api_secret
        self.session    = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self.order_log: list[dict] = []
        self._qty_precision: dict[str, int] | None = None  # symbol -> quantityPrecision

    # ── Signature ─────────────────────────────────────────────────────────
    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        sig   = hmac.new(self.api_secret.encode(),
                         query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _post(self, endpoint: str, params: dict) -> dict:
        params = self._sign(params)
        try:
            r = self.session.post(self.base + endpoint, data=params, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _get_signed(self, endpoint: str, params: dict) -> dict:
        params = self._sign(params)
        try:
            r = self.session.get(self.base + endpoint, params=params, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _delete(self, endpoint: str, params: dict) -> dict:
        params = self._sign(params)
        try:
            r = self.session.delete(self.base + endpoint, params=params, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Symbol quantity precision (per-symbol step size) ──────────────────
    def _load_qty_precision(self) -> dict[str, int]:
        """Fetch per-symbol quantityPrecision from exchangeInfo (cached).

        Binance rejects quantities that violate a symbol's step size. A fixed
        round(qty, 3) is wrong for symbols with a coarser step (e.g. integer
        lots), so round to each symbol's own precision instead.
        """
        if self._qty_precision is not None:
            return self._qty_precision
        cache: dict[str, int] = {}
        try:
            r = self.session.get(self.base + "/fapi/v1/exchangeInfo", timeout=10)
            for s in r.json().get("symbols", []):
                cache[s["symbol"]] = int(s.get("quantityPrecision", 3))
        except Exception:
            pass  # fall back to default rounding below
        self._qty_precision = cache
        return cache

    def _round_qty(self, symbol: str, quantity: float) -> float:
        precision = self._load_qty_precision().get(symbol, 3)
        return round(quantity, precision)

    # ── Set leverage ──────────────────────────────────────────────────────
    def set_leverage(self, symbol: str, leverage: int = 1) -> dict:
        return self._post("/fapi/v1/leverage", {
            "symbol": symbol, "leverage": leverage
        })

    # ── Market order (guaranteed fill) ───────────────────────────────────
    def place_market_order(self, symbol: str, side: str,
                           quantity: float) -> dict:
        qty = self._round_qty(symbol, quantity)
        params = {
            "symbol":   symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": qty,
        }
        result = self._post("/fapi/v1/order", params)
        self._log_order(symbol, side, qty, "MARKET", result)
        return result

    # ── Cancel all orders ─────────────────────────────────────────────────
    def cancel_all_orders(self, symbol: str) -> dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    # ── Positions & PnL ───────────────────────────────────────────────────
    def get_positions(self) -> list[dict]:
        data = self._get_signed("/fapi/v2/positionRisk", {})
        if isinstance(data, list):
            return [p for p in data if float(p.get("positionAmt", 0)) != 0]
        return []

    def get_account(self) -> dict:
        return self._get_signed("/fapi/v2/account", {})

    def get_balance(self) -> float:
        acc = self.get_account()
        if "assets" in acc:
            for a in acc["assets"]:
                if a["asset"] == "USDT":
                    return float(a["walletBalance"])
        return 0.0

    def get_unrealized_pnl(self) -> float:
        return sum(float(p.get("unRealizedProfit", 0))
                   for p in self.get_positions())

    # ── Execute target weights → market orders ────────────────────────────
    def execute_weights(self, target_weights: dict,
                        current_prices: dict,
                        total_capital: float = 100.0,
                        min_notional: float = 5.0) -> list[dict]:
        """
        Convert target weights → MARKET orders.
        Guaranteed to fill on both Testnet and Live.
        """
        results = []
        for sym, weight in target_weights.items():
            if abs(weight) < 0.01:
                continue
            price = current_prices.get(sym, 0)
            if price <= 0:
                continue

            notional = abs(weight) * total_capital
            if notional < min_notional:
                continue

            quantity = notional / price
            side     = "BUY" if weight > 0 else "SELL"

            # Set 1x leverage for safety
            self.set_leverage(sym, leverage=1)
            time.sleep(0.05)

            result = self.place_market_order(sym, side, quantity)
            status = result.get("status", result.get("error", "unknown"))

            results.append({
                "symbol":   sym,
                "side":     side,
                "weight":   weight,
                "notional": round(notional, 2),
                "quantity": round(quantity, 4),
                "result":   result,
            })
            time.sleep(0.1)

        return results

    # ── Flatten all positions ─────────────────────────────────────────────
    def flatten_all(self) -> list[dict]:
        positions = self.get_positions()
        results   = []
        for p in positions:
            sym = p["symbol"]
            amt = float(p["positionAmt"])
            if amt == 0:
                continue
            side   = "SELL" if amt > 0 else "BUY"
            result = self.place_market_order(sym, side, abs(amt))
            results.append({"symbol": sym, "side": side,
                            "qty": abs(amt), "result": result})
            time.sleep(0.1)
        return results

    # ── Order log ─────────────────────────────────────────────────────────
    def _log_order(self, symbol, side, qty, order_type, result):
        status = result.get("status", result.get("error", "unknown"))
        self.order_log.append({
            "time":    datetime.now(tz=timezone.utc).strftime("%H:%M:%S"),
            "symbol":  symbol,
            "side":    side,
            "qty":     round(qty, 4),
            "type":    order_type,
            "status":  status,
            "orderId": result.get("orderId", "—"),
        })
        if len(self.order_log) > 100:
            self.order_log = self.order_log[-100:]