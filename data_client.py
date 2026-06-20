"""
data_client.py
Handles all Binance Futures API connections.
Uses Testnet by default.
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

TESTNET_BASE = "https://testnet.binancefuture.com"
LIVE_BASE    = "https://fapi.binance.com"

SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","DOTUSDT"]

TRADE_SYMBOLS = [s for s in SYMBOLS if s != "BTCUSDT"]


class BinanceClient:
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.base = TESTNET_BASE if testnet else LIVE_BASE
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _get(self, endpoint: str, params: dict = None) -> dict | list:
        url = self.base + endpoint
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Klines ──────────────────────────────────────────────────────────────
    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
        """
        Returns OHLCV + taker buy volume for one symbol.
        Columns: open_time, open, high, low, close, volume,
                 taker_buy_vol, taker_sell_vol, tfi
        """
        data = self._get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "limit": limit
        })
        if isinstance(data, dict) and "error" in data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_vol",
            "taker_buy_quote","ignore"
        ])
        for c in ["open","high","low","close","volume","taker_buy_vol"]:
            df[c] = pd.to_numeric(df[c])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["taker_sell_vol"] = df["volume"] - df["taker_buy_vol"]
        denom = df["taker_buy_vol"] + df["taker_sell_vol"]
        df["tfi"] = np.where(denom > 0,
                             (df["taker_buy_vol"] - df["taker_sell_vol"]) / denom,
                             0.0)
        return df[["open_time","open","high","low","close","volume",
                   "taker_buy_vol","taker_sell_vol","tfi"]].copy()

    def get_all_klines(self, interval: str = "1m", limit: int = 200) -> dict[str, pd.DataFrame]:
        """Fetch klines for all 10 symbols."""
        return {sym: self.get_klines(sym, interval, limit) for sym in SYMBOLS}

    # ── Daily BTC klines for regime ─────────────────────────────────────────
    def get_btc_daily(self, limit: int = 300) -> pd.DataFrame:
        return self.get_klines("BTCUSDT", "1d", limit)

    # ── Funding rates ────────────────────────────────────────────────────────
    def get_funding_rates(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        data = self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
        if not data or isinstance(data, dict):
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"])
        return df[["fundingTime","fundingRate"]].sort_values("fundingTime")

    def get_all_funding(self) -> dict[str, float]:
        """Returns smoothed 3-period average funding rate per symbol."""
        rates = {}
        for sym in TRADE_SYMBOLS:
            df = self.get_funding_rates(sym, limit=6)
            if df.empty:
                rates[sym] = 0.0
            else:
                rates[sym] = float(df["fundingRate"].tail(3).mean())
        return rates

    # ── Order book snapshot ──────────────────────────────────────────────────
    def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Returns top-N bids and asks for microstructure features."""
        data = self._get("/fapi/v1/depth", {"symbol": symbol, "limit": limit})
        if isinstance(data, dict) and "error" in data:
            return {}
        bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
        return {"bids": bids, "asks": asks}

    def get_all_orderbooks(self) -> dict[str, dict]:
        return {sym: self.get_orderbook(sym) for sym in TRADE_SYMBOLS}

    # ── Server time check ────────────────────────────────────────────────────
    def server_time(self) -> datetime:
        data = self._get("/fapi/v1/time")
        if "serverTime" in data:
            return datetime.fromtimestamp(data["serverTime"] / 1000, tz=timezone.utc)
        return datetime.now(tz=timezone.utc)

    def ping(self) -> bool:
        data = self._get("/fapi/v1/ping")
        return data == {}
