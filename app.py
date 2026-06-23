"""
app.py
Regime-Gated PCA Microstructure Flow Strategy
Live Dashboard with Auto-Execute
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
from collections import deque

from data_client import BinanceClient, SYMBOLS, TRADE_SYMBOLS
from features import compute_obi, compute_microprice
from ui_risk_manager import RiskManager
from execution_manager import ExecutionManager
from crypto_pca_adapter import compute_crypto_pca_dashboard

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PCA Flow Strategy", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.metric-card { background:#0f1117; border:1px solid #2d3139; border-radius:8px; padding:16px 20px; margin-bottom:8px; }
.metric-label { font-size:11px; letter-spacing:0.12em; text-transform:uppercase; color:#6b7280; margin-bottom:4px; font-family:'IBM Plex Mono',monospace; }
.metric-value { font-size:24px; font-weight:600; font-family:'IBM Plex Mono',monospace; color:#f9fafb; }
.metric-sub   { font-size:12px; color:#9ca3af; margin-top:2px; font-family:'IBM Plex Mono',monospace; }
.regime-strong{color:#10b981;font-weight:600;} .regime-moderate{color:#f59e0b;font-weight:600;}
.regime-riskoff{color:#ef4444;font-weight:600;} .regime-forced{color:#a78bfa;font-weight:600;}
.regime-unknown{color:#9ca3af;font-weight:600;}
.status-ok{color:#10b981;} .status-bad{color:#ef4444;}
.disconnect-banner{ background:#3f1d1d; border:1px solid #ef4444; border-radius:8px; padding:10px 16px; margin-bottom:12px; color:#fca5a5; font-family:'IBM Plex Mono',monospace; font-size:13px; }
.auto-banner { background:#052e16; border:1px solid #16a34a; border-radius:8px; padding:10px 16px; margin-bottom:12px; color:#86efac; font-family:'IBM Plex Mono',monospace; font-size:13px; }
.force-banner{ background:#2d1b4e; border:1px solid #7c3aed; border-radius:8px; padding:10px 16px; margin-bottom:12px; color:#c4b5fd; font-family:'IBM Plex Mono',monospace; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    """Read a Streamlit secret, tolerating a missing secrets.toml entirely.

    st.secrets.get() parses the secrets file on access and raises
    StreamlitSecretNotFoundError when no secrets.toml exists anywhere, so it is
    not safe to call without a file present. Keys are optional here — the app
    runs on public data and only needs keys to place orders.
    """
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    api_key    = st.text_input("Binance API Key",    type="password", value=_secret("BINANCE_API_KEY"))
    api_secret = st.text_input("Binance API Secret", type="password", value=_secret("BINANCE_API_SECRET"))
    testnet    = st.toggle("Use Testnet", value=True)

    st.markdown("---")
    st.markdown("### 🚀 Execution Mode")
    paper_mode   = st.toggle("Paper Trading (no real orders)", value=True)
    force_trade  = st.toggle("⚡ Force Trade Mode", value=False,
                              help="Bypass RiskOff — trade even when BTC is bearish")
    auto_execute = st.toggle("🤖 Auto-Execute", value=False,
                              help="Automatically place orders on every refresh cycle")

    if force_trade:
        st.warning("⚡ Force Trade Mode ON")
    if auto_execute and not paper_mode:
        st.success("🤖 Auto-Execute ON — orders will be placed automatically")
    if auto_execute and paper_mode:
        st.info("🤖 Auto-Execute is ON but Paper Mode is also ON — no real orders")

    capital = st.number_input("Capital per rebalance (USDT)",
                               min_value=10, max_value=10000, value=100, step=10)

    st.markdown("---")
    st.markdown("### Strategy Parameters")
    direction_sign = st.radio("Direction Sign", ["+1 Continuation", "-1 Reversal"], index=0)
    dir_sign  = 1 if "1 C" in direction_sign else -1
    ic_gate   = st.toggle("IC Direction Gate", value=True,
                          help="Adaptive overlay: when ON, direction is set automatically from "
                               "out-of-sample IC and trades are blocked when the signal isn't "
                               "statistically reliable. OFF = use the Direction Sign above (PDF baseline).")
    alpha_w   = st.slider("Alpha weight", 0.5, 1.0, 0.8, 0.05)
    gross_exp = st.slider("Gross Exposure", 0.1, 2.0, 1.0, 0.1)
    refresh_sec = st.slider("Refresh every (seconds)", 10, 120, 30, 10)

    st.markdown("---")
    st.caption("Testnet only — no real money at risk")

# Non-blocking auto-refresh (replaces the old blocking time.sleep + st.rerun, which
# froze the whole script — and every button — for the full refresh interval).
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=int(refresh_sec * 1000), key="datarefresh")

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("risk_mgr", RiskManager()), ("exec_mgr", None),
             ("pnl_history", deque(maxlen=200)), ("alpha_history", deque(maxlen=60)),
             ("last_weights", {}), ("trade_count", 0), ("auto_log", deque(maxlen=50)),
             ("spread_history", {s: deque(maxlen=60) for s in TRADE_SYMBOLS}),
             ("last_exec_time", None), ("session_pnl", 0.0)]:
    if k not in st.session_state:
        st.session_state[k] = v

risk_mgr: RiskManager = st.session_state.risk_mgr
risk_mgr.check_daily_reset()

if api_key and api_secret and st.session_state.exec_mgr is None:
    st.session_state.exec_mgr = ExecutionManager(api_key, api_secret, testnet)
exec_mgr: ExecutionManager = st.session_state.exec_mgr

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
col_t, col_ts = st.columns([3,1])
with col_t:
    st.markdown("## Regime-Gated PCA Microstructure Flow Strategy")
    mode_label = "🤖 AUTO-EXECUTE" if (auto_execute and not paper_mode) else \
                 "📄 PAPER" if paper_mode else "🔴 LIVE TESTNET"
    st.caption(f"Binance USD-M USDT Perpetual Futures · 1-Min Alpha · {mode_label}")
with col_ts:
    st.markdown(f"**UTC** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    sh = '<span class="status-bad">⬤ HALTED</span>' if risk_mgr.trading_halted else '<span class="status-ok">⬤ LIVE</span>'
    st.markdown(sh, unsafe_allow_html=True)
st.markdown("---")

if auto_execute and not paper_mode:
    st.markdown('<div class="auto-banner">🤖 AUTO-EXECUTE ACTIVE — Strategy places orders automatically every refresh cycle</div>', unsafe_allow_html=True)
if force_trade:
    st.markdown('<div class="force-banner">⚡ FORCE TRADE MODE — BTC Regime filter bypassed</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client(k, s, tn):
    return BinanceClient(api_key=k, api_secret=s, testnet=tn)

client = get_client(api_key, api_secret, testnet)

with st.spinner("Fetching live data…"):
    connected  = client.ping()
    btc_daily  = client.get_btc_daily(limit=300) if connected else pd.DataFrame()
    klines_all = client.get_all_klines(interval="1m", limit=220) if connected else {}
    funding    = client.get_all_funding() if connected else {s:0.0 for s in TRADE_SYMBOLS}
    orderbooks = client.get_all_orderbooks() if connected else {}
    fetch_time = datetime.now(tz=timezone.utc)

if not connected:
    st.markdown(
        '<div class="disconnect-banner">⚠️ Not connected to Binance — showing empty data. '
        'Check your network, the Testnet toggle, or try again in a moment. '
        '(API keys are only needed to place orders, not to view data.)</div>',
        unsafe_allow_html=True,
    )

current_prices = {sym: float(klines_all[sym]["close"].iloc[-1])
                  for sym in TRADE_SYMBOLS
                  if sym in klines_all and not klines_all[sym].empty}

# ─────────────────────────────────────────────────────────────────────────────
# Signals from canonical crypto-pca-strategy
# ─────────────────────────────────────────────────────────────────────────────
canonical_result = compute_crypto_pca_dashboard(
    klines_all=klines_all,
    orderbooks=orderbooks,
    funding=funding,
    btc_daily=btc_daily,
    direction_sign=dir_sign,
    alpha_w=alpha_w,
    gross_exposure=gross_exp,
    force_trade=force_trade,
    ic_gate=ic_gate,
)

regime_info    = canonical_result["regime_info"]
regime         = regime_info["regime"]
effective_mult = regime_info["multiplier"]
adj_alpha      = canonical_result["adjusted_alpha"]
pc1_var        = canonical_result["pc1_var"]
resid_z        = canonical_result["residual_z"]
micro_scores   = canonical_result["micro_scores"]
micro_confirms = canonical_result["micro_confirms"]
liquidity_ok   = canonical_result["liquidity_ok"]
weights        = canonical_result["weights"]
port_metrics   = canonical_result["port_metrics"]
ob_metrics = {}

for sym in TRADE_SYMBOLS:
    ob        = orderbooks.get(sym, {})
    obi       = compute_obi(ob)
    mp        = compute_microprice(ob)
    tfi_short = float(klines_all[sym]["tfi"].tail(3).mean()
                      if sym in klines_all and not klines_all[sym].empty else 0.0)
    ob_metrics[sym] = {
        **mp,
        "obi": obi,
        "tfi_short": tfi_short,
    }
    # Track rolling spread so the spread/liquidity pre-trade check has a baseline.
    sb = mp.get("spread_bps")
    if sb is not None:
        st.session_state.spread_history[sym].append(sb)

# Pre-trade tradability gate (spread/depth/stale) applied before any order goes out.
tradeable = {}
for sym in TRADE_SYMBOLS:
    sb = ob_metrics.get(sym, {}).get("spread_bps")
    hist = st.session_state.spread_history[sym]
    median_spread = float(np.median(hist)) if len(hist) else 0.0
    tradeable[sym] = risk_mgr.is_tradeable(
        sym,
        spread_bps=sb if sb is not None else 0.0,
        median_spread=median_spread,
        orderbook=orderbooks.get(sym, {}),
        data_fresh=connected,
    )

if risk_mgr.trading_halted:
    weights = {sym: 0.0 for sym in TRADE_SYMBOLS}
    port_metrics = {
        "gross_exposure": 0.0,
        "net_exposure": 0.0,
        "n_long": 0,
        "n_short": 0,
    }
elif weights:
    st.session_state.last_weights = weights

if adj_alpha:
    st.session_state.alpha_history.append({"time": fetch_time, **adj_alpha})

# ─────────────────────────────────────────────────────────────────────────────
# Live PnL from Testnet
# ─────────────────────────────────────────────────────────────────────────────
live_positions = []; unrealized_pnl = 0.0; wallet_balance = 0.0
if exec_mgr and not paper_mode:
    try:
        live_positions = exec_mgr.get_positions()
        unrealized_pnl = exec_mgr.get_unrealized_pnl()
        wallet_balance = exec_mgr.get_balance()
    except Exception:
        pass
    # Feed PnL into the risk manager so the -3% daily-loss circuit breaker can fire.
    if wallet_balance > 0:
        risk_mgr.mark_daily_pnl(unrealized_pnl / wallet_balance)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-EXECUTE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
auto_exec_result = None
if auto_execute and not paper_mode and exec_mgr and not risk_mgr.trading_halted:
    active_weights = {s: w for s, w in weights.items()
                      if abs(w) > 0.01 and tradeable.get(s, False)}
    if active_weights:
        now = datetime.now(tz=timezone.utc)
        last = st.session_state.last_exec_time
        # Only execute if enough time has passed (avoid duplicate orders)
        if last is None or (now - last).total_seconds() > refresh_sec * 0.9:
            try:
                results = exec_mgr.execute_weights(
                    active_weights, current_prices, total_capital=float(capital)
                )
                st.session_state.trade_count += len(results)
                st.session_state.last_exec_time = now
                auto_exec_result = results
                # Log
                for r in results:
                    st.session_state.auto_log.appendleft({
                        "time":   now.strftime("%H:%M:%S"),
                        "symbol": r["symbol"],
                        "side":   r["side"],
                        "notional": f"{r['notional']:.1f}",
                        "status": r["result"].get("status", r["result"].get("error","?")),
                    })
            except Exception as e:
                st.session_state.auto_log.appendleft({
                    "time": datetime.now(tz=timezone.utc).strftime("%H:%M:%S"),
                    "symbol": "ERROR", "side": "—", "notional": "—", "status": str(e)
                })

# ─────────────────────────────────────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📡 Live Status")
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)

rc = {"Strong":"regime-strong","Moderate":"regime-moderate","RiskOff":"regime-riskoff"}.get(regime,"regime-unknown")
if force_trade: rc = "regime-forced"

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">BTC Regime</div>
        <div class="metric-value {rc}">{'FORCED' if force_trade else regime}</div>
        <div class="metric-sub">Mult: {effective_mult:.1f}× {'⚡' if force_trade else ''}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    r20=regime_info.get("r20"); r5=regime_info.get("r5")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">BTC Momentum</div>
        <div class="metric-value">{'N/A' if r20 is None else f'{r20:+.1f}%'}</div>
        <div class="metric-sub">5d: {'N/A' if r5 is None else f'{r5:+.1f}%'}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    vol20=regime_info.get("vol20"); vthr=regime_info.get("vol_threshold")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">BTC Vol</div>
        <div class="metric-value">{'N/A' if vol20 is None else f'{vol20:.1f}%'}</div>
        <div class="metric-sub">Q80: {'N/A' if vthr is None else f'{vthr:.1f}%'}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">PC1 Var</div>
        <div class="metric-value">{pc1_var*100:.1f}%</div>
        <div class="metric-sub">Common flow factor</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Gross Exposure</div>
        <div class="metric-value">{port_metrics['gross_exposure']:.2f}×</div>
        <div class="metric-sub">L:{port_metrics['n_long']} S:{port_metrics['n_short']}</div>
    </div>""", unsafe_allow_html=True)
with c6:
    uc = "status-ok" if unrealized_pnl >= 0 else "status-bad"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Unrealized PnL</div>
        <div class="metric-value {uc}">{unrealized_pnl:+.3f}</div>
        <div class="metric-sub">USDT · {'Paper' if paper_mode else 'Live'}</div>
    </div>""", unsafe_allow_html=True)
with c7:
    auto_txt = "🤖 AUTO" if (auto_execute and not paper_mode) else ("📄 PAPER" if paper_mode else "MANUAL")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Mode / Trades</div>
        <div class="metric-value" style="font-size:18px">{auto_txt}</div>
        <div class="metric-sub">Total: {st.session_state.trade_count} orders</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-execute log banner
# ─────────────────────────────────────────────────────────────────────────────
if auto_exec_result:
    filled = [r for r in auto_exec_result if "FILLED" in str(r["result"].get("status",""))]
    st.success(f"🤖 Auto-executed {len(auto_exec_result)} orders — {len(filled)} filled")

# ─────────────────────────────────────────────────────────────────────────────
# Execution Control (manual override)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Execution Control")
ex1, ex2, ex3 = st.columns(3)

with ex1:
    if st.button("▶️ Execute Now (Manual)", type="primary",
                  disabled=(paper_mode or not exec_mgr)):
        active = {s: w for s, w in weights.items()
                  if abs(w) > 0.01 and tradeable.get(s, False)}
        if active:
            with st.spinner("Placing orders…"):
                results = exec_mgr.execute_weights(active, current_prices, float(capital))
                st.session_state.trade_count += len(results)
            st.success(f"Placed {len(results)} orders")
            for r in results:
                status = r["result"].get("status", r["result"].get("error","?"))
                st.write(f"{'✅' if 'FILLED' in str(status) else '⚠️'} {r['symbol']} {r['side']} {r['quantity']:.4f} — {status}")
        else:
            st.warning("No active weights to execute")
    if paper_mode:
        st.caption("🔒 Disable Paper Mode to execute real orders")

with ex2:
    if st.button("🛑 Flatten All Positions", disabled=(paper_mode or not exec_mgr)):
        with st.spinner("Closing all…"):
            results = exec_mgr.flatten_all()
        st.success(f"Closed {len(results)} positions") if results else st.info("No open positions")

with ex3:
    if st.button("🔄 Refresh Now"):
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Auto-execute log
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.auto_log:
    st.markdown("### 🤖 Auto-Execute Log")
    df_auto = pd.DataFrame(list(st.session_state.auto_log))
    st.dataframe(df_auto, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Live Positions
# ─────────────────────────────────────────────────────────────────────────────
if not paper_mode and live_positions:
    st.markdown("### 📋 Live Open Positions")
    pos_rows = []
    for p in live_positions:
        amt  = float(p.get("positionAmt",0))
        upnl = float(p.get("unRealizedProfit",0))
        pos_rows.append({
            "Symbol": p["symbol"], "Size": amt,
            "Side":   "Long" if amt > 0 else "Short",
            "Entry":  float(p.get("entryPrice",0)),
            "Mark":   float(p.get("markPrice",0)),
            "PnL":    f"{upnl:+.4f} USDT",
        })
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Signal & Portfolio
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Signal & Portfolio")
col_a, col_w = st.columns(2)

with col_a:
    st.markdown("**Adjusted Alpha**")
    if adj_alpha:
        ss = sorted(adj_alpha.keys(), key=lambda s: adj_alpha[s], reverse=True)
        av = [adj_alpha[s] for s in ss]
        cl = ["#10b981" if v > 0 else "#ef4444" for v in av]
        mk = ["✓" if micro_confirms.get(s) else "✗" for s in ss]
        lb = [f"{s.replace('USDT','')} {m}" for s,m in zip(ss,mk)]
        fig = go.Figure(go.Bar(x=av, y=lb, orientation="h", marker_color=cl,
                               text=[f"{v:.3f}" for v in av], textposition="outside"))
        fig.update_layout(height=300, margin=dict(l=10,r=60,t=10,b=10),
                          plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                          font=dict(color="#9ca3af", family="IBM Plex Mono", size=11),
                          xaxis=dict(gridcolor="#1f2937", zeroline=True, zerolinecolor="#374151"),
                          yaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("✓ = microstructure confirmed  ✗ = blocked")
        di = canonical_result.get("direction_info") or {}
        if di.get("gate_enabled"):
            if di.get("active"):
                st.caption(
                    f"IC gate: direction {di.get('direction', 0):+d} · "
                    f"IC(1)={di.get('ic1', 0.0):.3f} (t={di.get('tstat', 0.0):.1f}, "
                    f"n={di.get('periods', 0)}) · "
                    f"{'signs stable' if di.get('signs_stable') else 'signs unstable'}"
                )
            else:
                st.caption(
                    f"IC gate: warming up ({di.get('periods', 0)} periods) — "
                    f"using sidebar prior {di.get('fallback_sign', 1):+d}"
                )
    else:
        st.info("Waiting for data (need 180+ bars)…")

with col_w:
    st.markdown("**Target Weights**")
    active_w = {s: w for s, w in weights.items() if abs(w) > 1e-6}
    if active_w:
        ws = sorted(active_w.keys(), key=lambda s: active_w[s], reverse=True)
        wv = [active_w[s] for s in ws]
        wc = ["#10b981" if v > 0 else "#ef4444" for v in wv]
        fig2 = go.Figure(go.Bar(x=wv, y=[s.replace("USDT","") for s in ws],
                                orientation="h", marker_color=wc,
                                text=[f"{v:+.3f}" for v in wv], textposition="outside"))
        fig2.update_layout(height=300, margin=dict(l=10,r=60,t=10,b=10),
                           plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                           font=dict(color="#9ca3af", family="IBM Plex Mono", size=11),
                           xaxis=dict(gridcolor="#1f2937", zeroline=True, zerolinecolor="#374151"),
                           yaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Capital: {capital} USDT · Gross: {port_metrics['gross_exposure']:.2f}×")
    else:
        reason = "Microstructure blocked all candidates" if force_trade else \
                 "Regime RiskOff — toggle Force Trade to override"
        st.warning(f"No active positions — {reason}")

# ─────────────────────────────────────────────────────────────────────────────
# Microstructure table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Microstructure Snapshot")
rows = []
for sym in TRADE_SYMBOLS:
    ob = ob_metrics.get(sym,{}); ms = micro_scores.get(sym,0.0)
    av = adj_alpha.get(sym,None); d = int(np.sign(av)) if av is not None else 0
    w  = weights.get(sym,0.0)
    rows.append({
        "Symbol":     sym.replace("USDT",""),
        "AdjAlpha":   f"{av:.4f}" if av is not None else "—",
        "Dir":        "▲" if d>0 else "▼" if d<0 else "—",
        "Weight":     f"{w:+.3f}" if abs(w)>1e-6 else "—",
        "OBI":        f"{ob.get('obi',0.0):+.4f}",
        "MicroScore": f"{ms:+.4f}",
        "Confirmed":  "✅" if micro_confirms.get(sym) else "❌",
        "Spread(bps)":f"{ob.get('spread_bps',0.0):.3f}",
        "Price":      f"{current_prices.get(sym,0):.3f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Alpha history + Heatmap
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
ch, chm = st.columns(2)
with ch:
    st.markdown("**Alpha History**")
    hd = list(st.session_state.alpha_history)
    if len(hd) > 3:
        df_h = pd.DataFrame(hd).set_index("time")
        fh = go.Figure()
        clrs = px.colors.qualitative.Set2
        for i, sym in enumerate(TRADE_SYMBOLS):
            if sym in df_h.columns:
                fh.add_trace(go.Scatter(x=df_h.index, y=df_h[sym],
                    name=sym.replace("USDT",""),
                    line=dict(width=1.5, color=clrs[i%len(clrs)]), mode="lines"))
        fh.add_hline(y=0, line_dash="dash", line_color="#374151")
        fh.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10),
                         plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                         font=dict(color="#9ca3af", size=10, family="IBM Plex Mono"),
                         xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
                         legend=dict(orientation="h", y=-0.3, font=dict(size=9)))
        st.plotly_chart(fh, use_container_width=True)
    else:
        st.info("Collecting alpha history…")

with chm:
    st.markdown("**Residual Z Heatmap**")
    if resid_z:
        sh = list(resid_z.keys())
        fht = go.Figure(go.Heatmap(
            z=[[resid_z[s] for s in sh]], x=[s.replace("USDT","") for s in sh], y=["ResidZ"],
            colorscale=[[0,"#ef4444"],[0.5,"#111827"],[1,"#10b981"]],
            zmid=0, text=[[f"{resid_z[s]:.3f}" for s in sh]],
            texttemplate="%{text}", showscale=True))
        fht.update_layout(height=140, margin=dict(l=10,r=10,t=10,b=10),
                          plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                          font=dict(color="#9ca3af", size=10, family="IBM Plex Mono"))
        st.plotly_chart(fht, use_container_width=True)
        st.caption("Green = strong buy flow · Red = strong sell flow")

# ─────────────────────────────────────────────────────────────────────────────
# Risk
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
cr, cl = st.columns([1,2])
with cr:
    st.markdown("**Risk Manager**")
    rs = risk_mgr.status()
    sc = "#ef4444" if rs["trading_halted"] else "#10b981"
    st.markdown(f"""
| | |
|---|---|
| Status | <span style="color:{sc}">{'HALTED' if rs['trading_halted'] else 'ACTIVE'}</span> |
| Daily PnL | {rs['daily_pnl_pct']:+.3f}% |
| Daily Limit | -3.000% |
| CB Fired | {rs['circuit_breaker_count']} |
| Wallet | {wallet_balance:.2f} USDT |
""", unsafe_allow_html=True)
    if rs["halt_reason"]: st.error(f"Halt: {rs['halt_reason']}")
    if st.button("🔄 Reset Risk Manager"): risk_mgr.resume(); st.rerun()

with cl:
    st.markdown("**Circuit Breaker Log**")
    if risk_mgr.circuit_breaker_log:
        st.dataframe(pd.DataFrame(risk_mgr.circuit_breaker_log), use_container_width=True, hide_index=True)
    else:
        st.success("No circuit breakers triggered.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer + auto-rerun
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
f1,f2,f3 = st.columns(3)
with f1: st.caption(f"Last fetch: {fetch_time.strftime('%H:%M:%S')} UTC")
with f2: st.caption(f"API: {'Testnet' if testnet else 'Live'} · {'✅' if connected else '❌'} · {'🤖 AUTO' if (auto_execute and not paper_mode) else '📄 PAPER' if paper_mode else 'MANUAL'}")
with f3: st.caption(f"Refresh: {refresh_sec}s · Capital: {capital} USDT · Trades: {st.session_state.trade_count}")

# Fallback auto-rerun only when streamlit-autorefresh isn't installed (blocking).
if not _HAS_AUTOREFRESH:
    time.sleep(refresh_sec)
    st.rerun()
