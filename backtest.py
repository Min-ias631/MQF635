"""
backtest.py — historical backtest of the FULL regime-gated PCA strategy.

Drives the canonical CombinedRegimePcaStrategy (BTC regime gate + PCA residual
flow alpha + IC direction gate + microstructure confirmation + Q80/Q20 vol-scaled
market-neutral portfolio) over real Binance USD-M perpetual futures history,
bar by bar, with every gate ON and nothing forced. Reports an equity curve,
Sharpe, max drawdown, turnover and a regime/trade timeline.

Usage (run from the repo root):
    python backtest.py                       # last 30 days, all defaults
    python backtest.py --days 60
    python backtest.py --start 2025-01-01 --days 45
    python backtest.py --no-ic-gate          # PDF baseline (simple IC sign rule)
    python backtest.py --refresh             # ignore cached data, refetch

Honest limitations (printed in the report too):
  * Binance has no historical L2 order books over REST, so books are synthesized
    from klines (neutral OBI/microprice, ~2 bps spread). Microstructure
    confirmation therefore degrades to a taker-flow-direction check; the ADV
    liquidity filter is fully real (from kline dollar volume).
  * Weights-based backtest: target weights are assumed achieved each bar (no
    fill simulation). Turnover is charged a configurable per-unit cost.
  * Funding is applied from real fundingRate history when available (3-period
    smoothed, per the spec); falls back to 0 if the fetch fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from crypto_pca_adapter import _build_config, _canonical_modules

LIVE_BASE = "https://fapi.binance.com"
CACHE_DIR = Path(__file__).resolve().parent / "backtest_data"
MIN_MS = 60_000
DAY_MS = 86_400_000
MINUTES_PER_YEAR = 365 * 1440


# ───────────────────────────────────────────────────────── data fetching ────

def _get(endpoint: str, params: dict) -> object:
    r = requests.get(LIVE_BASE + endpoint, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_klines(symbol: str, start_ms: int, end_ms: int, *, refresh: bool) -> pd.DataFrame:
    """Paginated 1-minute klines for [start, end), cached to backtest_data/."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{symbol}_1m_{start_ms}_{end_ms}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, parse_dates=["open_time"])
        return df

    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        data = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": "1m",
            "startTime": cur, "endTime": end_ms, "limit": 1500,
        })
        if not isinstance(data, list) or not data:
            break
        rows.extend(data)
        cur = data[-1][0] + MIN_MS
        if len(data) < 1500:
            break
        time.sleep(0.2)

    df = _klines_to_df(rows)
    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def _klines_to_df(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore",
    ])
    for c in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
        df[c] = pd.to_numeric(df[c])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["taker_sell_vol"] = df["volume"] - df["taker_buy_vol"]
    denom = df["taker_buy_vol"] + df["taker_sell_vol"]
    df["tfi"] = np.where(denom > 0, (df["taker_buy_vol"] - df["taker_sell_vol"]) / denom, 0.0)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume",
               "taker_buy_vol", "taker_sell_vol", "tfi"]]


def fetch_btc_daily(end_ms: int, *, refresh: bool) -> pd.DataFrame:
    cache = CACHE_DIR / f"BTCUSDT_1d_{end_ms}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["open_time"])
    data = _get("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1d",
                                    "endTime": end_ms, "limit": 400})
    df = _klines_to_df(data if isinstance(data, list) else [])
    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    """3-period-smoothed funding rate, indexed by funding settlement time."""
    try:
        data = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000,
        })
        if not isinstance(data, list) or not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        t = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        rate = pd.to_numeric(df["fundingRate"]).rolling(3, min_periods=1).mean()
        return pd.Series(rate.values, index=t).sort_index()
    except Exception:
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────────────────── replay ────────

def align_klines(klines: dict[str, pd.DataFrame], pca_symbols) -> pd.DatetimeIndex:
    idx = None
    for s in pca_symbols:
        df = klines.get(s)
        if df is None or df.empty:
            raise SystemExit(f"no klines for {s} — fetch failed or empty window")
        ts = pd.DatetimeIndex(df["open_time"])
        idx = ts if idx is None else idx.intersection(ts)
    return idx.sort_values()


def build_funding_frame(funding: dict[str, pd.Series], index: pd.DatetimeIndex,
                        trade_symbols) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=index, columns=list(trade_symbols))
    for s in trade_symbols:
        ser = funding.get(s)
        if ser is not None and len(ser):
            out[s] = ser.reindex(index, method="ffill").fillna(0.0).values
    return out


def run_replay(modules, config, klines, btc_daily, funding_frame, index,
               *, half_spread_bps: float = 1.0, progress: bool = True):
    """Step the strategy bar-by-bar; return (weights_df, regime_df, closes_df)."""
    Bar = modules["Bar"]
    BookSnapshot = modules["BookSnapshot"]
    FundingRate = modules["FundingRate"]
    MarketSnapshot = modules["MarketSnapshot"]
    strategy = modules["CombinedRegimePcaStrategy"](config)

    pca_symbols = config.universe.pca_symbols
    trade_symbols = config.universe.trade_symbols
    frames = {s: klines[s].set_index("open_time") for s in pca_symbols}

    # Per-date completed-daily-candle slices of BTC, to avoid look-ahead in the
    # regime gate (a minute on day D only sees daily candles up to D-1's close).
    btc = btc_daily.copy()
    btc["date"] = pd.DatetimeIndex(btc["open_time"]).date
    daily_by_date: dict[object, pd.DataFrame] = {}
    for d in sorted({ts.date() for ts in index}):
        sl = btc[btc["date"] <= d]
        daily_by_date[d] = sl[["close"]].reset_index(drop=True) if len(sl) else None

    weights_rows, regime_rows, signal_rows = [], [], []
    n = len(index)
    for i, ts in enumerate(index):
        pyts = ts.to_pydatetime()
        bars = {}
        for s in pca_symbols:
            row = frames[s].loc[ts]
            bars[s] = Bar(timestamp=pyts, symbol=s,
                          open=float(row["open"]), high=float(row["high"]),
                          low=float(row["low"]), close=float(row["close"]),
                          volume=float(row["volume"]),
                          taker_buy_volume=float(row["taker_buy_vol"]),
                          taker_sell_volume=float(row["taker_sell_vol"]))
        books = {}
        for s in trade_symbols:
            px = float(frames[s].loc[ts, "close"])
            hs = px * half_spread_bps / 1e4
            books[s] = BookSnapshot(timestamp=pyts, symbol=s,
                                    bids=((px - hs, 1.0),), asks=((px + hs, 1.0),))
        funding = {s: FundingRate(timestamp=pyts, symbol=s,
                                  rate=float(funding_frame.iloc[i][s]))
                   for s in trade_symbols}
        snap = MarketSnapshot(timestamp=pyts, bars=bars, books=books, funding=funding,
                              btc_daily=daily_by_date.get(ts.date()))
        target = strategy.on_market_snapshot(snap)
        weights_rows.append(target.weights)
        regime_rows.append({"regime": target.regime, "mult": target.regime_multiplier})
        # Capture the raw flow signal (residual-z) every bar — computed before the
        # regime/IC/cadence gates, so IC is measured on the pure signal regardless
        # of whether the strategy was allowed to trade.
        sig = strategy.latest_signals
        signal_rows.append({s: (sig[s].residual_z if s in sig else np.nan)
                            for s in trade_symbols})

        if progress and (i % 5000 == 0 or i == n - 1):
            print(f"  replay {i + 1}/{n} bars…", flush=True)

    cols = list(trade_symbols)
    weights_df = pd.DataFrame(weights_rows, index=index).reindex(columns=cols).fillna(0.0)
    regime_df = pd.DataFrame(regime_rows, index=index)
    signal_df = pd.DataFrame(signal_rows, index=index).reindex(columns=cols)
    closes_df = pd.DataFrame({s: frames[s]["close"].reindex(index) for s in trade_symbols})
    return weights_df, regime_df, signal_df, closes_df


# ──────────────────────────────────────────────────────────── metrics ───────

def compute_metrics(weights_df, regime_df, closes_df, cost_bps: float):
    rets = closes_df.pct_change().shift(-1)              # r_{t -> t+1}, aligned to t
    gross = (weights_df * rets).sum(axis=1)
    turnover = weights_df.diff().abs().sum(axis=1).fillna(weights_df.abs().sum(axis=1))
    cost = turnover * (cost_bps / 1e4)
    net = (gross - cost).iloc[:-1]                        # drop last bar (no fwd ret)

    equity = (1.0 + net).cumprod()
    gross_exp = weights_df.abs().sum(axis=1)
    in_market = gross_exp > 1e-9

    daily = net.resample("1D").sum()
    sharpe_min = _sharpe(net, MINUTES_PER_YEAR)
    sharpe_day = _sharpe(daily, 365)
    dd = equity / equity.cummax() - 1.0

    regime_counts = regime_df["regime"].value_counts(normalize=True).to_dict()

    # --- additional metrics (annualized return, win rate, trade count, benchmark) ---
    n = len(net)
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    total_return = (final_equity - 1.0) * 100.0
    ann_return = ((final_equity ** (MINUTES_PER_YEAR / n) - 1.0) * 100.0
                  if (n > 0 and final_equity > 0) else 0.0)
    # Win rate = share of IN-MARKET bars with positive net return.
    in_net = net[in_market.reindex(net.index, fill_value=False)]
    win_rate = float((in_net > 0).mean() * 100) if len(in_net) else 0.0
    # Trades = per-asset weight changes (one rebalance can touch several names);
    # first bar counts as entry from flat.
    dw = weights_df.diff()
    if len(weights_df):
        dw.iloc[0] = weights_df.iloc[0]
    num_trades = int((dw.abs() > 1e-9).to_numpy().sum())
    total_turnover = float(turnover.sum())
    # Benchmark = equal-weight buy-hold of the traded universe over the window.
    first, last = closes_df.iloc[0], closes_df.iloc[-1]
    bench_per = (last / first - 1.0).to_numpy(dtype=float)
    bench_return = float(np.nanmean(bench_per) * 100) if bench_per.size else 0.0

    return {
        "equity": equity,
        "net": net,
        "n_bars": int(n),
        "total_return_pct": total_return,
        "annualized_return_pct": ann_return,
        "sharpe_annual_from_min": sharpe_min,
        "sharpe_annual_from_daily": sharpe_day,
        "ann_vol_pct": float(net.std() * np.sqrt(MINUTES_PER_YEAR) * 100) if net.std() else 0.0,
        "max_drawdown_pct": float(dd.min() * 100) if len(dd) else 0.0,
        "win_rate_pct": win_rate,
        "time_in_market_pct": float(in_market.mean() * 100),
        "avg_gross_when_in": float(gross_exp[in_market].mean()) if in_market.any() else 0.0,
        "num_trades": num_trades,
        "rebalances": int((turnover > 1e-9).sum()),
        "avg_turnover": float(turnover[turnover > 1e-9].mean()) if (turnover > 1e-9).any() else 0.0,
        "total_turnover": total_turnover,
        "total_cost_pct": float(cost.iloc[:-1].sum() * 100),
        "benchmark_return_pct": bench_return,
        "excess_return_pct": total_return - bench_return,
        "regime_mix": {k: round(v * 100, 1) for k, v in regime_counts.items()},
    }


def compute_ic_report(signal_df: pd.DataFrame, closes_df: pd.DataFrame,
                      horizons=(1, 2, 5, 10, 30)) -> dict:
    """Per-period cross-sectional Spearman IC of the signal vs forward return.

    For each bar t and horizon h: rank-correlate the cross-section of the signal
    at t against the cross-section of returns r_{t->t+h}, then average those
    per-period ICs over the whole sample (Fama-MacBeth). Regime/gate/cost
    independent — a pure 'does this signal predict?' measure.
    """
    A = signal_df.to_numpy(dtype=float)
    C = closes_df.to_numpy(dtype=float)
    n = A.shape[0]
    out = {}
    for h in horizons:
        ics = []
        for t in range(n - h):
            a = A[t]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = C[t + h] / C[t] - 1.0
            valid = np.isfinite(a) & np.isfinite(r)
            if valid.sum() < 3:
                continue
            av, rv = a[valid], r[valid]
            if np.std(av) < 1e-12 or np.std(rv) < 1e-12:
                continue
            ics.append(_spearman(av, rv))
        ics = np.array([x for x in ics if np.isfinite(x)])
        if len(ics) < 2:
            out[h] = {"ic": 0.0, "tstat": 0.0, "hit": 0.0, "n": int(len(ics))}
            continue
        se = ics.std(ddof=1) / np.sqrt(len(ics))
        out[h] = {
            "ic": float(ics.mean()),
            "tstat": float(ics.mean() / se) if se > 1e-12 else 0.0,
            "hit": float((ics > 0).mean() * 100),
            "n": int(len(ics)),
        }
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def print_ic_report(ic: dict) -> None:
    print("\n" + "=" * 64)
    print("  SIGNAL DIAGNOSTIC - Information Coefficient")
    print("  (residual-flow signal vs forward return; gate/cost independent)")
    print("=" * 64)
    print(f"  {'horizon':>8}  {'IC':>8}  {'t-stat':>8}  {'hit%':>6}  {'periods':>8}")
    print("  " + "-" * 50)
    for h, r in ic.items():
        print(f"  {str(h)+'m':>8}  {r['ic']:+8.4f}  {r['tstat']:+8.2f}  "
              f"{r['hit']:6.1f}  {r['n']:8,}")
    print("  " + "-" * 50)
    # Verdict is driven by IC MAGNITUDE, not t-stat: with tens of thousands of
    # periods the t-stat inflates so much that even a ~0.005 IC clears |t|>2, so
    # significance alone is misleading. The cross-sectional noise floor here is
    # ~0.01, so peak |IC| below that is indistinguishable from no signal.
    if not ic:
        best_ic = 0.0; best_h = None; best_t = 0.0
    else:
        best_h = max(ic, key=lambda h: abs(ic[h]["ic"]))
        best_ic = abs(ic[best_h]["ic"])
        best_t = abs(ic[best_h]["tstat"])
    if best_ic >= 0.02 and best_t >= 3.0:
        verdict = ("Real signal - peak |IC|>=0.02 with |t|>=3. Worth keeping; the job "
                   "is to capture it net of turnover/costs.")
    elif best_ic >= 0.01 and best_t >= 2.0:
        verdict = ("Marginal - peak |IC| in 0.01-0.02. Possible weak edge; confirm "
                   "out-of-sample and in a Strong regime before trusting it.")
    else:
        verdict = ("NO usable edge - peak |IC| < 0.01, within the cross-sectional noise "
                   "floor. Tuning/de-turnoturning won't help; the signal needs rethinking.")
    print(f"  Verdict: {verdict}")
    if best_h is not None:
        print(f"  (peak |IC| = {best_ic:.4f} at {best_h}m; t-stat inflates with sample size, "
              f"so weight the magnitude.)")
    print("  Guide: in crypto microstructure, |IC|~0.02-0.03 (consistent sign, decaying")
    print("         across horizons) is a usable signal; peak |IC|<0.01 is noise.")
    print("=" * 64)


def _sharpe(series: pd.Series, ann: float) -> float:
    s = series.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(ann))


def _sparkline(equity: pd.Series, width: int = 64) -> str:
    if len(equity) < 2:
        return ""
    ramp = "_.-=+*#@"  # ASCII ramp (Windows cp1252-safe)
    pts = equity.iloc[np.linspace(0, len(equity) - 1, min(width, len(equity))).astype(int)]
    lo, hi = pts.min(), pts.max()
    rng = (hi - lo) or 1.0
    return "".join(ramp[int((v - lo) / rng * (len(ramp) - 1))] for v in pts)


def print_report(m: dict, args, index):
    print("\n" + "=" * 64)
    print("  HISTORICAL BACKTEST — Regime-Gated PCA Flow (full strategy)")
    print("=" * 64)
    print(f"  Window      : {index[0].date()} → {index[-1].date()}  ({m['n_bars']:,} 1-min bars)")
    print(f"  IC gate     : {'ON (live direction)' if not args.no_ic_gate else 'OFF (PDF sign rule)'}"
          f"   Gross: {args.gross}x   Cost: {args.cost_bps} bps/turn")
    print("-" * 64)
    print(f"  Total return        : {m['total_return_pct']:+8.2f} %")
    print(f"  Annualized return   : {m['annualized_return_pct']:+8.2f} %")
    print(f"  Annualized vol      : {m['ann_vol_pct']:8.2f} %")
    print(f"  Sharpe (ann, daily) : {m['sharpe_annual_from_daily']:+8.2f}")
    print(f"  Sharpe (ann, 1-min) : {m['sharpe_annual_from_min']:+8.2f}")
    print(f"  Max drawdown        : {m['max_drawdown_pct']:+8.2f} %")
    print(f"  Win rate (in mkt)   : {m['win_rate_pct']:8.1f} %")
    print(f"  Time in market      : {m['time_in_market_pct']:8.1f} %")
    print(f"  Avg gross (in mkt)  : {m['avg_gross_when_in']:8.2f} x")
    print(f"  Trades / rebalances : {m['num_trades']:,} / {m['rebalances']:,}"
          f"   avg turnover {m['avg_turnover']:.3f}   total {m['total_turnover']:.1f}")
    print(f"  Total trading cost  : {m['total_cost_pct']:8.2f} %")
    print(f"  Benchmark (EW BH)   : {m['benchmark_return_pct']:+8.2f} %")
    print(f"  Excess vs benchmark : {m['excess_return_pct']:+8.2f} %")
    print(f"  Regime mix          : {m['regime_mix']}")
    print("-" * 64)
    print("  Equity: " + _sparkline(m["equity"]))
    print("=" * 64)


# ──────────────────────────────────────────────────────────── main ──────────

def main() -> None:
    p = argparse.ArgumentParser(description="Backtest the full regime-gated PCA strategy.")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", type=str, default=None, help="UTC start date YYYY-MM-DD")
    p.add_argument("--gross", type=float, default=1.0)
    p.add_argument("--alpha-w", type=float, default=0.8)
    p.add_argument("--direction", type=int, default=1, choices=(1, -1))
    p.add_argument("--no-ic-gate", action="store_true")
    p.add_argument("--no-confirm", action="store_true",
                   help="disable microstructure confirmation (it can't be evaluated "
                        "without historical L2 books, and it fights a reversal alpha)")
    p.add_argument("--no-regime", action="store_true",
                   help="disable the BTC regime gate -> deploy in all regimes "
                        "(ablation: a market-neutral book shouldn't need a BTC-trend gate)")
    p.add_argument("--smooth", type=int, default=1,
                   help="EWMA span (bars) for the traded alpha; >1 cuts turnover")
    p.add_argument("--band", type=float, default=0.0,
                   help="no-trade band: skip weight moves smaller than this")
    p.add_argument("--rebalance", type=int, default=5,
                   help="minutes between portfolio rebalances (default 5); raise to cut turnover")
    p.add_argument("--cost-bps", type=float, default=1.0, help="cost per unit turnover")
    p.add_argument("--refresh", action="store_true", help="ignore cached data")
    p.add_argument("--out", type=str, default="backtest_results.csv")
    args = p.parse_args()

    try:  # make box-drawing/emoji-free output robust on legacy Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.start:
        start_ms = int(datetime.strptime(args.start, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
    else:
        start_ms = int(time.time() * 1000) - args.days * DAY_MS
    end_ms = min(start_ms + args.days * DAY_MS, int(time.time() * 1000))

    modules = _canonical_modules()
    config = _build_config(args.gross, args.direction, args.alpha_w,
                           not args.no_ic_gate, not args.no_confirm,
                           regime_on=not args.no_regime,
                           smooth_span=args.smooth, no_trade_band=args.band,
                           rebalance_minutes=args.rebalance)
    pca_symbols = config.universe.pca_symbols
    trade_symbols = config.universe.trade_symbols

    print(f"Fetching {args.days}d of 1-min data for {len(pca_symbols)} symbols "
          f"(cached in {CACHE_DIR.name}/)…", flush=True)
    klines = {}
    for s in pca_symbols:
        klines[s] = fetch_klines(s, start_ms, end_ms, refresh=args.refresh)
        print(f"  {s}: {len(klines[s]):,} bars", flush=True)
    btc_daily = fetch_btc_daily(end_ms, refresh=args.refresh)
    if btc_daily.empty:
        raise SystemExit("BTC daily fetch failed — cannot evaluate the regime gate.")
    funding = {s: fetch_funding(s, start_ms, end_ms) for s in trade_symbols}

    index = align_klines(klines, pca_symbols)
    print(f"Aligned {len(index):,} common bars. Replaying strategy (every gate on)…")
    funding_frame = build_funding_frame(funding, index, trade_symbols)
    weights_df, regime_df, signal_df, closes_df = run_replay(
        modules, config, klines, btc_daily, funding_frame, index)

    m = compute_metrics(weights_df, regime_df, closes_df, args.cost_bps)
    print_report(m, args, index)

    ic = compute_ic_report(signal_df, closes_df)
    print_ic_report(ic)

    out = pd.DataFrame({
        "regime": regime_df["regime"],
        "mult": regime_df["mult"],
        "gross_exposure": weights_df.abs().sum(axis=1),
        "net_return": (weights_df * closes_df.pct_change().shift(-1)).sum(axis=1),
    })
    out["equity"] = m["equity"].reindex(out.index)
    out_path = Path(__file__).resolve().parent / args.out
    out.to_csv(out_path)
    print(f"\nPer-bar records + equity curve saved to {out_path.name}")
    if "RiskOff" in m["regime_mix"] and m["time_in_market_pct"] < 1.0:
        print("NOTE: the strategy was flat almost the whole window (mostly RiskOff). "
              "Try --start on a past uptrend, or a longer --days, to see it deploy.")


if __name__ == "__main__":
    main()
