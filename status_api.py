"""
status_api.py —— FastAPI status service

运行方式：
    START_API=1 MARKET_FEED=binance EXECUTION=paper python3.12 main.py

或在其它进程中导入 create_app(store)。
"""

from __future__ import annotations

import socket

from status import StatusStore


def create_app(store: StatusStore):
    try:
        from fastapi import FastAPI
    except ImportError as e:
        raise RuntimeError(
            "FastAPI status API requires fastapi and uvicorn. "
            "Install with: python3.12 -m pip install fastapi uvicorn"
        ) from e

    app = FastAPI(title="Algo Trading Status API")

    @app.get("/")
    def dashboard():
        from fastapi.responses import HTMLResponse

        return HTMLResponse(
            """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trading System Status</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #141b2d;
      --line: #263247;
      --text: #edf2f7;
      --muted: #95a3b8;
      --cyan: #38bdf8;
      --gold: #fbbf24;
      --red: #fb7185;
      --green: #34d399;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      font-size: 24px;
      margin: 0;
      font-weight: 650;
    }
    .muted { color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .value {
      font-size: 24px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      white-space: nowrap;
    }
    th { color: var(--muted); font-weight: 600; }
    .section {
      margin-top: 12px;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    .chart-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 280px;
    }
    .chart-title {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      margin-bottom: 12px;
    }
    canvas {
      width: 100%;
      height: 220px;
      display: block;
      border-radius: 6px;
      background:
        linear-gradient(180deg, rgba(56,189,248,0.04), rgba(56,189,248,0.01)),
        linear-gradient(90deg, rgba(149,163,184,0.06) 1px, transparent 1px),
        linear-gradient(0deg, rgba(149,163,184,0.06) 1px, transparent 1px);
      background-size: auto, 56px 56px, 56px 56px;
      background-position: 0 0, 0 0, 0 0;
    }
    .ok { color: var(--green); }
    .warn { color: var(--gold); }
    .bad { color: var(--red); }
    @media (max-width: 900px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chart-grid { grid-template-columns: 1fr; }
      header { display: block; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Trading System Status</h1>
      <div class="muted" id="updated">connecting...</div>
    </header>
    <section class="grid">
      <div class="panel"><div class="label">Risk State</div><div class="value" id="risk">--</div></div>
      <div class="panel"><div class="label">Daily PnL</div><div class="value" id="pnl">--</div></div>
      <div class="panel"><div class="label">Signals</div><div class="value" id="signals">--</div></div>
      <div class="panel"><div class="label">Orders / Fills</div><div class="value" id="orders">--</div></div>
    </section>
    <section class="chart-grid">
      <div class="chart-panel">
        <div class="chart-title">PnL Curve</div>
        <canvas id="pnlChart" width="560" height="220"></canvas>
      </div>
      <div class="chart-panel">
        <div class="chart-title">Mid Price Curves</div>
        <canvas id="midChart" width="420" height="220"></canvas>
      </div>
    </section>
    <section class="panel section">
      <div class="label">Books</div>
      <div class="muted" style="margin-bottom:10px">This page is the live dashboard. `/status` returns raw JSON for debugging.</div>
      <table>
        <thead><tr><th>Symbol</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Spread</th><th>Imb3</th><th>Valid</th></tr></thead>
        <tbody id="books"></tbody>
      </table>
    </section>
    <section class="panel section">
      <div class="label">Latest Signals</div>
      <table>
        <thead><tr><th>Symbol</th><th>Action</th><th>Strength</th><th>Strategy</th></tr></thead>
        <tbody id="signalRows"></tbody>
      </table>
    </section>
  </main>
  <script>
    const fmt = (x, d = 4) => (x === null || x === undefined) ? "--" : Number(x).toFixed(d);
    const actionLabel = (x) => x === 1 ? "LONG" : (x === -1 ? "SHORT" : (x === 0 ? "FLAT" : String(x ?? "--")));
    function row(cells) { return "<tr>" + cells.map(c => `<td>${c}</td>`).join("") + "</tr>"; }
    function money(x) {
      if (x === null || x === undefined) return "--";
      const n = Number(x);
      return (n >= 0 ? "+" : "") + n.toFixed(4);
    }
    function drawLineChart(canvasId, series, colors, yLabelFormatter) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const margin = {top: 16, right: 12, bottom: 20, left: 46};
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;

      const allValues = series.flatMap(s => s.points.map(p => Number(p.value))).filter(v => Number.isFinite(v));
      if (!allValues.length) {
        ctx.fillStyle = "#95a3b8";
        ctx.font = "12px Segoe UI";
        ctx.fillText("Waiting for enough data...", margin.left, height / 2);
        return;
      }

      let min = Math.min(...allValues);
      let max = Math.max(...allValues);
      if (min === max) {
        const bump = Math.abs(min || 1) * 0.01;
        min -= bump;
        max += bump;
      }

      ctx.strokeStyle = "rgba(149,163,184,0.25)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = margin.top + (plotH * i / 4);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
        ctx.stroke();
      }

      ctx.fillStyle = "#95a3b8";
      ctx.font = "11px Segoe UI";
      for (let i = 0; i <= 4; i++) {
        const y = margin.top + (plotH * i / 4);
        const value = max - ((max - min) * i / 4);
        ctx.fillText(yLabelFormatter(value), 6, y + 4);
      }

      series.forEach((line, idx) => {
        const pts = line.points.filter(p => Number.isFinite(Number(p.value)));
        if (!pts.length) return;
        ctx.strokeStyle = colors[idx % colors.length];
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        pts.forEach((p, i) => {
          const x = margin.left + (pts.length === 1 ? plotW / 2 : (plotW * i / (pts.length - 1)));
          const y = margin.top + plotH - ((Number(p.value) - min) / (max - min)) * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        const last = pts[pts.length - 1];
        const lastX = margin.left + (pts.length === 1 ? plotW / 2 : plotW);
        const lastY = margin.top + plotH - ((Number(last.value) - min) / (max - min)) * plotH;
        ctx.fillStyle = colors[idx % colors.length];
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.font = "11px Segoe UI";
      series.forEach((line, idx) => {
        ctx.fillStyle = colors[idx % colors.length];
        ctx.fillRect(margin.left + idx * 110, height - 10, 10, 3);
        ctx.fillText(line.label, margin.left + 14 + idx * 110, height - 6);
      });
    }
    async function refresh() {
      const res = await fetch("/status", {cache: "no-store"});
      const s = await res.json();
      const stats = s.runner_stats || {};
      const history = s.history || {};
      document.getElementById("updated").textContent = new Date().toLocaleTimeString();
      document.getElementById("risk").textContent = s.risk_state || "--";
      document.getElementById("risk").className = "value " + (s.risk_state === "NORMAL" ? "ok" : "warn");
      document.getElementById("pnl").textContent = money(s.daily_pnl);
      document.getElementById("pnl").className = "value " + ((Number(s.daily_pnl) || 0) >= 0 ? "ok" : "bad");
      document.getElementById("signals").textContent = stats.signals ?? 0;
      document.getElementById("orders").textContent = `${stats.orders ?? 0} / ${stats.fills ?? 0}`;
      document.getElementById("books").innerHTML = Object.values(s.books || {}).map(b => row([
        b.symbol, fmt(b.best_bid), fmt(b.best_ask), fmt(b.mid), fmt(b.spread), fmt(b.imbalance_3, 3),
        b.is_valid ? "<span class='ok'>yes</span>" : "<span class='bad'>no</span>"
      ])).join("");
      document.getElementById("signalRows").innerHTML = Object.values(s.signals || {}).map(ev => {
        const p = ev.payload || {};
        return row([p.symbol, actionLabel(p.action), fmt(p.strength, 2), p.strategy_id]);
      }).join("");

      drawLineChart(
        "pnlChart",
        [{label: "Daily PnL", points: history.pnl || []}],
        ["#34d399"],
        (v) => Number(v).toFixed(4)
      );

      const midSeries = Object.entries(history.mid_by_symbol || {}).map(([sym, points]) => ({
        label: sym,
        points: points
      }));
      drawLineChart(
        "midChart",
        midSeries,
        ["#38bdf8", "#fbbf24", "#fb7185", "#34d399"],
        (v) => {
          const n = Number(v);
          if (Math.abs(n) >= 1000) return n.toFixed(1);
          if (Math.abs(n) >= 100) return n.toFixed(2);
          return n.toFixed(4);
        }
      );
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
            """
        )

    @app.get("/status")
    def get_status():
        return store.snapshot()

    @app.get("/books")
    def get_books():
        return {"books": store.books()}

    @app.get("/positions")
    def get_positions():
        return {"positions": store.positions()}

    @app.get("/history")
    def get_history():
        return store.history()

    @app.get("/events/recent")
    def get_recent_events():
        return {"events": store.recent_events()}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def start_api_thread(store: StatusStore, *, host: str = "127.0.0.1", port: int = 8000):
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "FastAPI status API requires uvicorn. "
            "Install with: python3.12 -m pip install fastapi uvicorn"
        ) from e

    import threading

    _ensure_port_available(host, port)
    app = create_app(store)
    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port, "log_level": "info"},
        name="status-api",
        daemon=True,
    )
    thread.start()
    return thread


def _ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as e:
            raise RuntimeError(
                f"Status API port {host}:{port} is already in use. "
                "Stop the previous main.py process with Ctrl+C, or set STATUS_API_PORT=8001."
            ) from e


__all__ = ["create_app", "start_api_thread"]
