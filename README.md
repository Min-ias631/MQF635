# Regime-Gated PCA Microstructure Flow Strategy

**Final Revised Version**

---

##  Project Guide & Quickstart

Everything is **one flat package** — run all commands from this directory; no `cd`
between sub-folders. Modules import each other directly (one shared engine, used by
the live framework, the backtester, and the UI).

**Module map**
- **Shared alpha engine:** `engine_config.py`, `events.py`, `pca_flow.py`,
  `microstructure_confirmation.py`, `combined.py` (rolling PCA residual-flow alpha +
  microstructure confirmation + regime gate + Q80/Q20 portfolio — the single source of truth).
- **Event-driven framework:** `config.py` (settings), `event.py`, `book.py`, `feed.py`,
  `broker.py` (Sim/Paper/**Live**), `risk.py`, `portfolio.py`, `runner.py`, `record.py`,
  `status.py`, `status_api.py`, `binance_rest.py`, `main.py`, plus the strategies
  `crypto_pca.py`, `microstructure_strategy.py`, `strategy_base.py`.
- **UI + backtester:** `app.py` (Streamlit), `backtest.py` (historical backtest + IC
  diagnostic), `data_client.py`, `crypto_pca_adapter.py`, `features.py`, `signal_engine.py`,
  `ui_portfolio.py`, `risk_manager.py`, `execution_manager.py`.
- **Docs:** `Project Report.pdf`.	

### Install
```bash
python -m pip install -r requirements.txt
python -m pip install pytest        # to run tests
```

### Run (all from this directory)
```bash
# Backtest the full strategy (PnL / Sharpe / drawdown / IC diagnostic)
python backtest.py --start 2026-05-19 --days 30

# Streamlit live dashboard (public data; no keys needed to view)
python -m streamlit run app.py

# Event framework end-to-end (mock data, simulated fills)
MODE=paper MARKET_FEED=mock EXECUTION=sim SYMBOLS=BTCUSDT,ETHUSDT python main.py
# expect: RunnerStats(market_events=2, signals=2, orders=2, fills=2, ...)

# Live testnet trading (signed REST orders + user-data fill stream)
MODE=live MARKET_FEED=binance EXECUTION=live BINANCE_TESTNET=1 \
  BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy SYMBOLS=BTCUSDT,ETHUSDT python main.py

# Tests
python -m pytest tests/ -q          # expect: 69 passed
```

> **Credentials:** put them once in `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`).
> That single file is read by both the Streamlit UI and live trading. Environment variables override it.
> On Windows PowerShell set env vars with `$env:NAME="value"; python main.py`.


## Strategy summary

A regime-gated, market-neutral, cross-sectional **PCA residual-flow** strategy on Binance USD-M
perpetual futures (BTC + 9 liquid altcoin perps; BTC is used for the regime gate and the PCA
common-factor, the traded universe excludes BTC). Three layers:

1. **BTC regime gate** (daily) — scales the gross risk budget by a BTC momentum + realized-vol
   regime: Strong (1.0×), Moderate (0.5×), RiskOff (0×, flatten).
2. **PCA residual-flow alpha** (1-minute) — rolling PCA on the cross-section of taker-flow
   imbalance, OLS-residualized against PC1/PC2, cross-sectionally z-scored, blended 0.8/0.2 with a
   funding-crowdedness signal; trade direction set from out-of-sample IC.
3. **Microstructure confirmation + Q80/Q20 vol-scaled, market-neutral long-short portfolio.**

**Canonical specification:** `Project Report.pdf`.

### As actually implemented (authoritative — matches the code)

| Parameter | Value |
|---|---|
| BTC regime windows | **5-day & 20-day momentum, 20-day realized vol, Q80 over 252 days** |
| Alpha blend | 0.8 · ResidualZ + 0.2 · FundingSignal |
| PCA / beta windows | 120 / 60 bars, 2 components, PC signs pinned across windows |
| Selection | Q80 long / Q20 short on the liquidity-eligible non-BTC set; min 4 names |
| Direction | IC stability gate (configurable; falls back to the configured sign) |
| Rebalance | every 5 minutes (alpha updates every minute) |
| Live execution | signed REST orders (tick/step rounding, GTX post-only) + user-data fill stream + cancel/replace |

### Not implemented (design ideas in the spec only — NOT in the code)

To keep the report honest: the **probe-order liquidity test** and the **queue-aware "realistic fill"
model** described in some spec drafts are *design proposals*, not implemented. The historical
backtester also **synthesizes order books from 1-minute klines** (no historical L2 data is used), so
the microstructure-confirmation layer runs in live/paper but is only *approximated* in backtest.
See the Limitations section of the report.

### Results & evaluation

See **Project Report.pdf** for the empirical results (performance metrics, benchmark, IC
diagnostic, ablations, and limitations). Reproduce with `python backtest.py --start 2026-05-19 --days 30`.

---
## Code Structure
<img width="5640" height="2680" alt="trading_pipeline_with_binance_testnet_datasource" src="https://github.com/user-attachments/assets/067b4f6f-584d-48be-9a4d-fba033672b66" />


