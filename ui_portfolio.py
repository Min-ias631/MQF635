"""
portfolio.py
Constructs the long-short portfolio from adjusted alpha.
  - Q80/Q20 selection on eligible universe
  - Volatility-scaled weights
  - Regime-gated gross exposure
  - Single-asset cap
  - Market neutrality preservation
"""

import numpy as np
import pandas as pd

SINGLE_ASSET_CAP = 0.20   # max weight per asset
MIN_ELIGIBLE     = 4      # fallback: skip if fewer eligible assets
Q_LONG           = 0.80
Q_SHORT          = 0.20


def compute_realized_vol(klines_dict: dict, symbol: str,
                         window: int = 20) -> float:
    """20-bar rolling vol of 1-min returns, annualized."""
    df = klines_dict.get(symbol, pd.DataFrame())
    if df is None or len(df) < window + 1:
        return 1.0
    closes = df["close"].astype(float).values
    rets = np.diff(np.log(closes[-(window+1):]))
    vol = np.std(rets, ddof=1) * np.sqrt(525600)   # minutes in a year
    return max(vol, 1e-6)


def check_spread_filter(spread_bps: float, median_spread_bps: float) -> bool:
    """Block if spread > 1.5x rolling median."""
    if median_spread_bps <= 0:
        return True
    return spread_bps <= 1.5 * median_spread_bps


def construct_portfolio(adjusted_alpha: dict,
                        micro_confirm: dict,
                        liquidity_ok: dict,
                        klines_dict: dict,
                        regime_multiplier: float,
                        gross_exposure: float = 1.0) -> dict:
    """
    Returns target weights {symbol: weight}.
    Positive = long, negative = short, zero = flat.

    adjusted_alpha:    {sym: float}
    micro_confirm:     {sym: bool}  — does order book support direction?
    liquidity_ok:      {sym: bool}  — passes spread + ADV filter?
    klines_dict:       raw klines for vol estimation
    regime_multiplier: 0, 0.5, or 1.0
    gross_exposure:    baseline gross (default 1.0 = 100%)
    """
    weights = {}

    if regime_multiplier == 0:
        return {sym: 0.0 for sym in adjusted_alpha}

    Gt = regime_multiplier * gross_exposure

    # Build eligible universe
    eligible = {
        sym: alpha
        for sym, alpha in adjusted_alpha.items()
        if liquidity_ok.get(sym, False) and micro_confirm.get(sym, False)
    }

    if len(eligible) < MIN_ELIGIBLE:
        # Fallback: too few eligible assets, skip this rebalance
        return {sym: 0.0 for sym in adjusted_alpha}

    syms   = list(eligible.keys())
    alphas = np.array([eligible[s] for s in syms])

    q80 = np.quantile(alphas, Q_LONG)
    q20 = np.quantile(alphas, Q_SHORT)

    long_syms  = [s for s, a in zip(syms, alphas) if a > q80]
    short_syms = [s for s, a in zip(syms, alphas) if a < q20]

    def _vol_scale_leg(leg_syms, leg_alphas_raw, leg_sign):
        """Returns {sym: weight} for one leg."""
        vols = np.array([compute_realized_vol(klines_dict, s) for s in leg_syms])
        scores = np.abs(leg_alphas_raw) / vols
        total = scores.sum()
        if total < 1e-10:
            return {}
        raw_w = {s: leg_sign * (sc / total) * 0.5 * Gt
                 for s, sc in zip(leg_syms, scores)}
        return raw_w

    long_alphas  = np.array([eligible[s] for s in long_syms])
    short_alphas = np.array([eligible[s] for s in short_syms])

    long_weights  = _vol_scale_leg(long_syms,  long_alphas,  +1) if long_syms  else {}
    short_weights = _vol_scale_leg(short_syms, short_alphas, -1) if short_syms else {}

    combined = {**long_weights, **short_weights}

    # Single-asset cap + re-normalize
    combined = _apply_cap_and_renormalize(combined, SINGLE_ASSET_CAP, Gt)

    # Zero out all non-selected
    for sym in adjusted_alpha:
        weights[sym] = combined.get(sym, 0.0)

    return weights


def _apply_cap_and_renormalize(weights: dict, cap: float, Gt: float) -> dict:
    """
    Cap any single asset at `cap` gross weight.
    Re-normalize each leg separately to preserve market neutrality.
    """
    long_w  = {s: w for s, w in weights.items() if w > 0}
    short_w = {s: w for s, w in weights.items() if w < 0}

    def _cap_leg(leg):
        capped = {s: min(abs(w), cap) * np.sign(w) for s, w in leg.items()}
        total_abs = sum(abs(v) for v in capped.values())
        target = 0.5 * Gt
        if total_abs > 1e-10:
            scale = target / total_abs
            capped = {s: v * scale for s, v in capped.items()}
        return capped

    return {**_cap_leg(long_w), **_cap_leg(short_w)}


def compute_portfolio_metrics(weights: dict) -> dict:
    """Summary stats for the current portfolio."""
    w = np.array(list(weights.values()))
    gross = np.sum(np.abs(w))
    net   = np.sum(w)
    n_long  = int((w > 0).sum())
    n_short = int((w < 0).sum())
    return {
        "gross_exposure": round(float(gross), 4),
        "net_exposure":   round(float(net), 6),
        "n_long":  n_long,
        "n_short": n_short,
    }
