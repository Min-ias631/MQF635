"""config.py — typed configuration for the canonical crypto-PCA alpha engine.

Every constant that the ported math used as a module-level literal in
``eugene-ui/strategy/*`` is surfaced here as a dataclass field so that both
adapters (Yumi backtest, Streamlit UI) configure one engine the same way.
Defaults reproduce the original hardcoded values; no behaviour change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# Default universe mirrors data_client.SYMBOLS / TRADE_SYMBOLS.
_PCA_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
)
_TRADE_SYMBOLS = tuple(s for s in _PCA_SYMBOLS if s != "BTCUSDT")


@dataclass(frozen=True)
class UniverseConfig:
    pca_symbols: Tuple[str, ...] = _PCA_SYMBOLS
    trade_symbols: Tuple[str, ...] = _TRADE_SYMBOLS


@dataclass(frozen=True)
class AlphaConfig:
    """Rolling PCA residual-flow alpha parameters (PDF Layer 2 + research overlays)."""

    pca_window: int = 120        # PCA covariance window
    beta_window: int = 60        # rolling OLS residualization window
    n_components: int = 2        # PC1 + PC2
    direction_sign: int = 1      # +1 continuation, -1 reversal (PDF: set by IC validation)
    flow_weight: float = 0.8     # weight on residual-z in the blend
    funding_weight: float = 0.2  # weight on funding signal in the blend
    long_quantile: float = 0.80  # Q80 long cutoff
    short_quantile: float = 0.20 # Q20 short cutoff
    # IC direction stability gate (strategy.md research overlay; the PDF baseline
    # is the simple sign rule). When enabled and enough realized IC history has
    # accumulated, DirectionSign is set from live out-of-sample IC instead of the
    # configured prior: +1/-1 only if IC(1) clears ic_min with |t| >= ic_tstat_min
    # and the IC sign is stable across ic_horizons; otherwise 0 (no new positions).
    # Before ic_min_periods of history, the configured direction_sign is used.
    ic_stability_gate: bool = True    # adaptive IC-direction overlay (set False for fixed direction_sign)
    ic_min: float = 0.01
    ic_tstat_min: float = 2.0
    ic_horizons: Tuple[int, ...] = (1, 2, 5)
    ic_min_periods: int = 30
    ic_lookback: int = 240       # finalized minute bars retained for IC estimation
    # Turnover control: EWMA-smooth the traded alpha over this many bars before
    # ranking, so Q80/Q20 membership (and thus the book) is stable. 1 = off.
    # The raw signal used for the IC gate is left unsmoothed.
    smooth_span: int = 1


@dataclass(frozen=True)
class MicrostructureConfig:
    """Order-book confirmation + liquidity filters (PDF Layer 3)."""

    obi_levels: int = 10         # depth levels for weighted OBI
    w_obi: float = 0.4           # MicroScore = 0.4*OBI + 0.3*MicroSignal + 0.3*TFI_short
    w_micro: float = 0.3
    w_tfi: float = 0.3
    # Spread filter: block if SpreadBps > spread_median_mult x RollingMedian(SpreadBps, window)
    spread_median_window: int = 60
    spread_median_mult: float = 1.5
    # ADV filter: exclude assets in the bottom adv_exclude_pct of cross-sectional
    # 20-minute average dollar volume (close x volume)
    adv_window: int = 20
    adv_exclude_pct: float = 20.0
    # Ablation switch (strategy.md MicroConfirmGate): when False, selection skips
    # order-book confirmation entirely (EffectiveMicroConfirm = 1 for all assets).
    confirm_gate: bool = True


@dataclass(frozen=True)
class RegimeConfig:
    """BTC daily regime gate — Strategy(PCA+BTC Attention).pdf, Layer 1.

    5-day (short) and 20-day (long) momentum, 20-day realized vol, vol threshold =
    80th percentile of the rolling 20-day vol over the past 252 days. README.md,
    strategy.md and the PDF all use these same windows.
    """

    enabled: bool = True         # False -> always "AllOn"/mult 1.0 (ablation)
    momentum_short_days: int = 5
    momentum_long_days: int = 20
    volatility_window_days: int = 20
    volatility_threshold_days: int = 252
    volatility_percentile: float = 80.0


@dataclass(frozen=True)
class PortfolioConfig:
    """Long-short construction parameters (PDF Portfolio Construction + Execution)."""

    base_gross_exposure: float = 1.0
    single_asset_cap: float = 0.20
    min_eligible: int = 4
    vol_window: int = 20         # realized-vol window for vol scaling
    # Turnover control: don't move a name unless its target weight changes by more
    # than this (gross weight units). 0.0 = off. Cuts churn from tiny re-rankings.
    no_trade_band: float = 0.0
    # PDF Execution Algorithm: alpha updates every minute, but the baseline target
    # portfolio refreshes every 5 minutes to limit turnover. RiskOff still flattens
    # immediately regardless of cadence.
    rebalance_minutes: int = 5


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 0.0
    impact_eta: float = 0.1


@dataclass(frozen=True)
class RiskConfig:
    max_gross: float = 2.0
    max_single_weight: float = 0.25


@dataclass(frozen=True)
class BacktestDataConfig:
    minute_bars_path: Path = Path("minute_bars.csv")
    order_book_path: Path = Path("order_book.csv")
    funding_path: Path = Path("funding.csv")
    btc_daily_path: Path = Path("btc_daily.csv")


@dataclass(frozen=True)
class StrategyConfig:
    """Aggregate config consumed by CombinedRegimePcaStrategy."""

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest_data: BacktestDataConfig = field(default_factory=BacktestDataConfig)


__all__ = [
    "UniverseConfig",
    "AlphaConfig",
    "MicrostructureConfig",
    "RegimeConfig",
    "PortfolioConfig",
    "CostConfig",
    "RiskConfig",
    "BacktestDataConfig",
    "StrategyConfig",
]
