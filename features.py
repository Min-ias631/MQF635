"""
features.py
Builds all features from raw Binance data:
  - BTC daily regime (Strong / Moderate / RiskOff)
  - Cross-sectional TFI matrix
  - Smoothed funding signal
  - Order-book imbalance (OBI)
  - Microprice signal
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore as scipy_zscore


TRADE_SYMBOLS = ["ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
                 "ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","DOTUSDT"]

# ── BTC Regime ───────────────────────────────────────────────────────────────

# Regime windows per Strategy(PCA+BTC Attention).pdf Layer 1: 5-day (short) and
# 20-day (long) momentum, 20-day realized vol, threshold = Q80 over past 252 days.
MOM_SHORT_DAYS = 5
MOM_LONG_DAYS  = 20
VOL_WINDOW_DAYS = 20
VOL_LOOKBACK_DAYS = 252


def compute_btc_regime(btc_daily: pd.DataFrame) -> dict:
    """
    Input: BTC daily klines, strictly backward-looking.
    Returns dict with regime label and multiplier.
    """
    if btc_daily is None or len(btc_daily) < VOL_LOOKBACK_DAYS + 18:
        return {"regime": "Unknown", "multiplier": 0.0,
                "r5": None, "r20": None, "vol20": None, "vol_threshold": None}

    closes = btc_daily["close"].values.astype(float)
    # Use only completed candles: last row may be incomplete
    c = closes[:-1]  # drop current incomplete candle

    if len(c) < VOL_LOOKBACK_DAYS:
        return {"regime": "Unknown", "multiplier": 0.0,
                "r5": None, "r20": None, "vol20": None, "vol_threshold": None}

    r5  = c[-1] / c[-(MOM_SHORT_DAYS + 1)] - 1
    r20 = c[-1] / c[-(MOM_LONG_DAYS + 1)]  - 1

    # Daily log returns for vol
    daily_rets = np.diff(np.log(c))

    vol20 = float(np.std(daily_rets[-VOL_WINDOW_DAYS:], ddof=1) * np.sqrt(365))

    # Rolling 20-day vol over past 252 trading days
    past_vols = [
        float(np.std(daily_rets[i:i + VOL_WINDOW_DAYS], ddof=1) * np.sqrt(365))
        for i in range(len(daily_rets) - VOL_LOOKBACK_DAYS, len(daily_rets) - VOL_WINDOW_DAYS)
        if i >= 0
    ]
    vol_threshold = float(np.percentile(past_vols, 80)) if past_vols else 999.0

    if r20 > 0 and r5 > 0 and vol20 < vol_threshold:
        regime, mult = "Strong", 1.0
    elif r20 > 0 and r5 <= 0 and vol20 < vol_threshold:
        regime, mult = "Moderate", 0.5
    else:
        regime, mult = "RiskOff", 0.0

    return {
        "regime": regime,
        "multiplier": mult,
        "r5": round(r5 * 100, 2),
        "r20": round(r20 * 100, 2),
        "vol20": round(vol20 * 100, 2),
        "vol_threshold": round(vol_threshold * 100, 2),
    }


# ── TFI Matrix ───────────────────────────────────────────────────────────────

def build_tfi_matrix(klines_dict: dict, symbols: list, n_bars: int = 200) -> pd.DataFrame:
    """
    Returns a DataFrame of shape (n_bars, len(symbols))
    where each column is the TFI series for that symbol,
    aligned on open_time.
    """
    frames = {}
    for sym in symbols:
        df = klines_dict.get(sym, pd.DataFrame())
        if df.empty or "tfi" not in df.columns:
            continue
        s = df.set_index("open_time")["tfi"].tail(n_bars)
        frames[sym] = s

    if not frames:
        return pd.DataFrame()

    matrix = pd.DataFrame(frames).dropna()
    return matrix


# ── Funding Signal ───────────────────────────────────────────────────────────

def compute_funding_signal(funding_rates: dict) -> dict:
    """
    Input: {symbol: smoothed_3period_avg_funding_rate}
    Returns: {symbol: FundingSignal = -tanh(z-score)}
    """
    syms = list(funding_rates.keys())
    vals = np.array([funding_rates[s] for s in syms], dtype=float)

    if len(vals) < 2 or np.std(vals) == 0:
        return {s: 0.0 for s in syms}

    zvals = scipy_zscore(vals)
    signals = {s: float(-np.tanh(z)) for s, z in zip(syms, zvals)}
    return signals


# ── Order Book Microstructure ─────────────────────────────────────────────────

def compute_obi(orderbook: dict, n_levels: int = 10) -> float:
    """
    Weighted order book imbalance: w_k = 1/k
    Returns OBI in [-1, +1]
    """
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    if not bids or not asks:
        return 0.0

    bid_w = sum((1.0 / (k+1)) * q for k, (p, q) in enumerate(bids[:n_levels]))
    ask_w = sum((1.0 / (k+1)) * q for k, (p, q) in enumerate(asks[:n_levels]))

    denom = bid_w + ask_w
    if denom == 0:
        return 0.0
    return float((bid_w - ask_w) / denom)


def compute_microprice(orderbook: dict) -> dict:
    """
    Returns mid, microprice, spread_bps, micro_signal.
    """
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        return {"mid": None, "microprice": None,
                "spread_bps": None, "micro_signal": 0.0}

    best_bid_p, best_bid_q = bids[0]
    best_ask_p, best_ask_q = asks[0]

    mid = (best_bid_p + best_ask_p) / 2.0
    spread = best_ask_p - best_bid_p
    spread_bps = (spread / mid) * 10000 if mid > 0 else 0.0

    denom = best_bid_q + best_ask_q
    if denom == 0:
        return {"mid": mid, "microprice": mid,
                "spread_bps": spread_bps, "micro_signal": 0.0}

    microprice = (best_ask_p * best_bid_q + best_bid_p * best_ask_q) / denom
    epsilon = max(spread, 0.5 * mid / 10000) if mid > 0 else 1e-8
    micro_signal = float((microprice - mid) / epsilon)

    return {
        "mid": round(mid, 6),
        "microprice": round(microprice, 6),
        "spread_bps": round(spread_bps, 4),
        "micro_signal": round(micro_signal, 4),
    }


def compute_micro_score(obi: float, micro_signal: float, tfi_short: float) -> float:
    """MicroScore = 0.4*OBI + 0.3*MicroSignal + 0.3*TFI_short"""
    return 0.4 * obi + 0.3 * micro_signal + 0.3 * tfi_short


def micro_confirm(trade_direction: int, micro_score: float) -> bool:
    """Returns True if micro_score aligns with intended trade direction."""
    if trade_direction == 0:
        return False
    return (trade_direction * micro_score) > 0
