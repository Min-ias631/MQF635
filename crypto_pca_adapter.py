"""Adapter from the shared PCA flow engine to the Streamlit UI / backtester.

Flat package: engine modules are imported directly (no sys.path bridge).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _canonical_modules():
    from combined import CombinedRegimePcaStrategy
    from engine_config import (
        AlphaConfig,
        BacktestDataConfig,
        CostConfig,
        MicrostructureConfig,
        PortfolioConfig,
        RegimeConfig,
        RiskConfig,
        StrategyConfig,
        UniverseConfig,
    )
    from events import Bar, BookSnapshot, FundingRate, MarketSnapshot

    return {
        "CombinedRegimePcaStrategy": CombinedRegimePcaStrategy,
        "AlphaConfig": AlphaConfig,
        "BacktestDataConfig": BacktestDataConfig,
        "CostConfig": CostConfig,
        "MicrostructureConfig": MicrostructureConfig,
        "PortfolioConfig": PortfolioConfig,
        "RegimeConfig": RegimeConfig,
        "RiskConfig": RiskConfig,
        "StrategyConfig": StrategyConfig,
        "UniverseConfig": UniverseConfig,
        "Bar": Bar,
        "BookSnapshot": BookSnapshot,
        "FundingRate": FundingRate,
        "MarketSnapshot": MarketSnapshot,
    }


def _build_config(gross_exposure: float, direction_sign: int, alpha_w: float,
                  ic_gate: bool = True, confirm_gate: bool = True,
                  regime_on: bool = True, smooth_span: int = 1,
                  no_trade_band: float = 0.0, rebalance_minutes: int = 5):
    m = _canonical_modules()
    return m["StrategyConfig"](
        universe=m["UniverseConfig"](),
        regime=m["RegimeConfig"](enabled=regime_on),
        alpha=m["AlphaConfig"](
            direction_sign=direction_sign,
            flow_weight=alpha_w,
            funding_weight=1.0 - alpha_w,
            ic_stability_gate=ic_gate,
            smooth_span=smooth_span,
        ),
        microstructure=m["MicrostructureConfig"](confirm_gate=confirm_gate),
        portfolio=m["PortfolioConfig"](base_gross_exposure=gross_exposure,
                                       no_trade_band=no_trade_band,
                                       rebalance_minutes=rebalance_minutes),
        cost=m["CostConfig"](commission_bps=0.0, impact_eta=0.1),
        risk=m["RiskConfig"](),
        backtest_data=m["BacktestDataConfig"](
            minute_bars_path=Path("minute_bars.csv"),
            order_book_path=Path("order_book.csv"),
            funding_path=Path("funding.csv"),
            btc_daily_path=Path("btc_daily.csv"),
        ),
    )


def compute_crypto_pca_dashboard(
    *,
    klines_all: Dict[str, pd.DataFrame],
    orderbooks: Dict[str, dict],
    funding: Dict[str, float],
    btc_daily: pd.DataFrame,
    direction_sign: int,
    alpha_w: float,
    gross_exposure: float,
    force_trade: bool = False,
    ic_gate: bool = True,
) -> dict[str, Any]:
    """Run the canonical combined strategy on the current UI data batch."""

    modules = _canonical_modules()
    config = _build_config(gross_exposure, direction_sign, alpha_w, ic_gate)
    strategy = modules["CombinedRegimePcaStrategy"](config)
    snapshots = _build_snapshots(modules, config, klines_all, orderbooks, funding, btc_daily)

    if not snapshots:
        zero_weights = {symbol: 0.0 for symbol in config.universe.trade_symbols}
        return {
            "regime_info": {"regime": "Unknown", "multiplier": 0.0},
            "adjusted_alpha": {},
            "residual_z": {},
            "funding_signal": {},
            "micro_scores": {},
            "micro_confirms": {},
            "liquidity_ok": {},
            "weights": zero_weights,
            "port_metrics": _portfolio_metrics(zero_weights),
            "pc1_var": 0.0,
            "direction_info": {},
        }

    target = None
    for snapshot in snapshots:
        target = strategy.on_market_snapshot(snapshot)

    assert target is not None
    signals = strategy.latest_signals
    regime_info = _regime_display(config, btc_daily)
    regime_info["regime"] = "Forced" if force_trade else target.regime
    regime_info["multiplier"] = 1.0 if force_trade else target.regime_multiplier

    weights = dict(target.weights)
    if force_trade and target.regime_multiplier == 0.0 and signals:
        weights = strategy.portfolio.construct(
            timestamp=snapshots[-1].timestamp,
            signals=signals,
            regime="Forced",
            regime_multiplier=1.0,
            long_quantile=config.alpha.long_quantile,
            short_quantile=config.alpha.short_quantile,
        ).weights

    adjusted_alpha = {symbol: signal.adjusted_alpha for symbol, signal in signals.items()}
    residual_z = {symbol: signal.residual_z for symbol, signal in signals.items()}
    funding_signal = {symbol: signal.funding_signal for symbol, signal in signals.items()}
    micro_scores = {symbol: signal.micro_score for symbol, signal in signals.items()}
    micro_confirms = {symbol: signal.micro_confirmed for symbol, signal in signals.items()}
    liquidity_ok = {symbol: signal.eligible for symbol, signal in signals.items()}

    return {
        "regime_info": regime_info,
        "adjusted_alpha": adjusted_alpha,
        "residual_z": residual_z,
        "funding_signal": funding_signal,
        "micro_scores": micro_scores,
        "micro_confirms": micro_confirms,
        "liquidity_ok": liquidity_ok,
        "weights": weights,
        "port_metrics": _portfolio_metrics(weights),
        "pc1_var": _pc1_explained_variance(config, klines_all),
        "direction_info": dict(getattr(strategy.flow_alpha, "direction_info", {}) or {}),
    }


def _build_snapshots(modules, config, klines_all, orderbooks, funding, btc_daily):
    aligned = []
    for symbol in config.universe.pca_symbols:
        frame = klines_all.get(symbol)
        if frame is None or frame.empty:
            return []
        aligned.append(frame.set_index("open_time"))

    common_index = aligned[0].index
    for frame in aligned[1:]:
        common_index = common_index.intersection(frame.index)
    common_index = common_index.sort_values()

    min_rows = config.alpha.pca_window + config.alpha.beta_window + 2
    if len(common_index) < min_rows:
        return []

    # Replay the full available history (not just the minimum window) so the IC
    # direction gate accumulates realized alpha-vs-return periods; capped to keep
    # the per-refresh compute bounded.
    max_rows = max(min_rows, 600)
    snapshots = []
    for timestamp in common_index[-max_rows:]:
        bars = {}
        for symbol, frame in zip(config.universe.pca_symbols, aligned):
            row = frame.loc[timestamp]
            bars[symbol] = modules["Bar"](
                timestamp=timestamp.to_pydatetime(),
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                taker_buy_volume=float(row["taker_buy_vol"]),
                taker_sell_volume=float(row["taker_sell_vol"]),
            )

        books = {
            symbol: modules["BookSnapshot"](
                timestamp=timestamp.to_pydatetime(),
                symbol=symbol,
                bids=tuple(orderbooks.get(symbol, {}).get("bids", ())),
                asks=tuple(orderbooks.get(symbol, {}).get("asks", ())),
            )
            for symbol in config.universe.trade_symbols
        }
        rates = {
            symbol: modules["FundingRate"](
                timestamp=timestamp.to_pydatetime(),
                symbol=symbol,
                rate=float(funding.get(symbol, 0.0)),
            )
            for symbol in config.universe.trade_symbols
        }
        snapshots.append(
            modules["MarketSnapshot"](
                timestamp=timestamp.to_pydatetime(),
                bars=bars,
                books=books,
                funding=rates,
                btc_daily=btc_daily,
            )
        )
    return snapshots


def _portfolio_metrics(weights: dict[str, float]) -> dict[str, float | int]:
    values = np.array(list(weights.values()), dtype=float) if weights else np.array([])
    return {
        "gross_exposure": round(float(np.abs(values).sum()), 4) if len(values) else 0.0,
        "net_exposure": round(float(values.sum()), 6) if len(values) else 0.0,
        "n_long": int((values > 0).sum()) if len(values) else 0,
        "n_short": int((values < 0).sum()) if len(values) else 0,
    }


def _pc1_explained_variance(config, klines_all) -> float:
    rows = []
    for symbol in config.universe.pca_symbols:
        frame = klines_all.get(symbol)
        if frame is None or len(frame) < config.alpha.pca_window:
            return 0.0
        rows.append(frame["tfi"].tail(config.alpha.pca_window).to_numpy(dtype=float))
    matrix = np.column_stack(rows)
    matrix = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) < 1e-12, 1.0, matrix.std(axis=0))
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    variances = s**2
    return float(variances[0] / variances.sum()) if variances.sum() > 0 else 0.0


def _regime_display(config, btc_daily: pd.DataFrame) -> dict[str, Any]:
    if btc_daily is None or btc_daily.empty or len(btc_daily) < config.regime.momentum_long_days + 2:
        return {"regime": "Unknown", "multiplier": 0.0, "r5": None, "r20": None, "vol20": None, "vol_threshold": None}

    closes = btc_daily["close"].astype(float)
    short = config.regime.momentum_short_days
    long = config.regime.momentum_long_days
    vol_window = config.regime.volatility_window_days
    threshold_days = config.regime.volatility_threshold_days
    completed = closes.iloc[:-1] if len(closes) > threshold_days else closes
    if len(completed) < threshold_days:
        return {"regime": "Unknown", "multiplier": 0.0, "r5": None, "r20": None, "vol20": None, "vol_threshold": None}

    returns = completed.pct_change()
    short_mom = completed.iloc[-1] / completed.iloc[-short - 1] - 1.0
    long_mom = completed.iloc[-1] / completed.iloc[-long - 1] - 1.0
    vol = returns.iloc[-vol_window:].std() * np.sqrt(365)
    historical = returns.rolling(vol_window).std().mul(np.sqrt(365)).dropna().iloc[-threshold_days - 1 : -1]
    threshold = float(np.percentile(historical, config.regime.volatility_percentile)) if not historical.empty else None
    return {
        "regime": "Unknown",
        "multiplier": 0.0,
        "r5": round(short_mom * 100, 2),
        "r20": round(long_mom * 100, 2),
        "vol20": round(float(vol) * 100, 2),
        "vol_threshold": round(threshold * 100, 2) if threshold is not None else None,
    }
