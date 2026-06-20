"""
main.py —— 本地组装入口

示例：
    MODE=backtest BOOK_CSV=./data/book.csv python3.12 main.py
    MODE=paper python3.12 main.py
    MODE=paper MARKET_FEED=binance EXECUTION=paper START_API=1 python3.12 main.py

当前 LiveBroker 尚未实现，会 fail-fast。
"""

from __future__ import annotations

import logging
import os
import queue
import time

from book import Book
from broker import LiveBroker, PaperBroker, SimBroker
from config import settings
from event import BookTickEvent, MarketEvent, PriceLevel
from feed import BinancePublicFeed, CSVFeed, LiveFeed, MarketDataFeed, MockFeed, MockTick
from portfolio import Portfolio
from record import EventRecorder
from risk import RiskManager
from runner import Runner
from status import StatusStore
from status_api import start_api_thread
from crypto_pca import CryptoPcaStrategy
from microstructure_strategy import MicrostructureImbalanceStrategy


def _configure_logging() -> None:
    level_name = (settings.log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_feed(event_queue: "queue.Queue[object]") -> MarketDataFeed:
    market_feed = os.getenv("MARKET_FEED", "").lower().strip()
    if market_feed == "binance":
        return BinancePublicFeed(
            event_queue,  # type: ignore[arg-type]
            symbols=settings.symbols,
            depth_levels=int(os.getenv("BINANCE_DEPTH_LEVELS", "5")),
        )
    if market_feed == "mock":
        return MockFeed(event_queue, _default_mock_ticks())  # type: ignore[arg-type]
    if market_feed == "csv":
        return CSVFeed(
            event_queue,  # type: ignore[arg-type]
            book_csv=os.getenv("BOOK_CSV"),
            trade_csv=os.getenv("TRADE_CSV"),
            funding_csv=os.getenv("FUNDING_CSV"),
            background=settings.mode != "backtest",
        )

    if settings.mode == "backtest":
        return CSVFeed(
            event_queue,  # type: ignore[arg-type]
            book_csv=os.getenv("BOOK_CSV"),
            trade_csv=os.getenv("TRADE_CSV"),
            funding_csv=os.getenv("FUNDING_CSV"),
            background=False,
        )

    if settings.mode == "paper":
        book_csv = os.getenv("BOOK_CSV")
        trade_csv = os.getenv("TRADE_CSV")
        funding_csv = os.getenv("FUNDING_CSV")
        if book_csv or trade_csv or funding_csv:
            return CSVFeed(
                event_queue,  # type: ignore[arg-type]
                book_csv=book_csv,
                trade_csv=trade_csv,
                funding_csv=funding_csv,
                background=True,
            )
        return MockFeed(event_queue, _default_mock_ticks())  # type: ignore[arg-type]

    if settings.mode == "live":
        return LiveFeed(event_queue)  # type: ignore[arg-type]

    raise ValueError(f"unsupported mode: {settings.mode}")


def build_broker(event_queue: "queue.Queue[object] | None" = None):
    execution = os.getenv("EXECUTION", settings.mode).lower().strip()
    if execution == "sim":
        return SimBroker(fee_rate=float(os.getenv("SIM_FEE_RATE", "0.0")))
    if execution == "paper":
        return PaperBroker()
    if execution == "live":
        return _build_live_broker(event_queue)

    if settings.mode == "backtest":
        return SimBroker(fee_rate=float(os.getenv("SIM_FEE_RATE", "0.0")))
    if settings.mode == "paper":
        return PaperBroker()
    if settings.mode == "live":
        return _build_live_broker(event_queue)
    raise ValueError(f"unsupported mode: {settings.mode}")


def _build_live_broker(event_queue: "queue.Queue[object] | None"):
    if not settings.binance_api_key or not settings.binance_secret:
        raise RuntimeError(
            "EXECUTION=live requires BINANCE_API_KEY / BINANCE_SECRET. "
            "Use EXECUTION=sim or EXECUTION=paper to run without credentials."
        )
    from binance_rest import BinanceFuturesRest

    rest = BinanceFuturesRest(
        settings.binance_api_key,
        settings.binance_secret,
        testnet=settings.binance_testnet,
    )
    return LiveBroker(
        rest,
        event_queue,
        symbols=settings.symbols,
        post_only=os.getenv("LIVE_POST_ONLY", "0") == "1",
    )


def build_runner(*, status_store: StatusStore | None = None) -> Runner:
    event_queue: queue.Queue[object] = queue.Queue(maxsize=settings.event_queue_maxsize)
    book = Book(symbols=settings.symbols)
    portfolio = Portfolio()
    risk = RiskManager()
    strategy = build_strategy()
    recorder = EventRecorder()
    feed = build_feed(event_queue)
    broker = build_broker(event_queue)

    return Runner(
        event_queue=event_queue,
        book=book,
        strategy=strategy,
        risk=risk,
        broker=broker,
        portfolio=portfolio,
        recorder=recorder,
        feed=feed,
        status_store=status_store,
    )


def build_strategy():
    strategy_name = os.getenv("STRATEGY", "microstructure").lower().strip()
    if strategy_name in ("crypto_pca", "pca", "combined"):
        return CryptoPcaStrategy(
            pca_window=int(os.getenv("PCA_WINDOW", "120")),
            beta_window=int(os.getenv("PCA_BETA_WINDOW", "60")),
            direction_sign=int(os.getenv("PCA_DIRECTION_SIGN", "1")),
            entry_alpha_threshold=float(os.getenv("PCA_ENTRY_ALPHA_THRESHOLD", "0.0")),
            # 300s default = the strategy note's baseline 5-minute portfolio refresh
            cooldown_ns=int(os.getenv("PCA_COOLDOWN_NS", "300000000000")),
        )
    if strategy_name in ("microstructure", "micro"):
        return MicrostructureImbalanceStrategy(
            levels=int(os.getenv("MICRO_LEVELS", "3")),
            entry_threshold=float(os.getenv("MICRO_ENTRY_THRESHOLD", "0.35")),
            exit_threshold=float(os.getenv("MICRO_EXIT_THRESHOLD", "0.10")),
            max_spread_bps=float(os.getenv("MICRO_MAX_SPREAD_BPS", "8.0")),
            cooldown_ns=int(os.getenv("MICRO_COOLDOWN_NS", "1000000000")),
            strength=float(os.getenv("MICRO_STRENGTH", "1.0")),
        )
    raise ValueError(f"unsupported STRATEGY={strategy_name!r}")


def run() -> None:
    _configure_logging()
    status_store = StatusStore()
    api_enabled = os.getenv("START_API", "0") == "1"
    api_host = os.getenv("STATUS_API_HOST", "127.0.0.1")
    api_port = int(os.getenv("STATUS_API_PORT", "8000"))
    market_feed = os.getenv("MARKET_FEED", "").lower().strip()
    execution = os.getenv("EXECUTION", settings.mode).lower().strip()

    _print_startup_banner(
        market_feed=market_feed or "(default)",
        execution=execution,
        api_enabled=api_enabled,
        api_host=api_host,
        api_port=api_port,
    )
    if api_enabled:
        start_api_thread(
            status_store,
            host=api_host,
            port=api_port,
        )

    runner = build_runner(status_store=status_store)
    if market_feed == "binance":
        print("Binance public feed is running. Open the dashboard URL above and wait for market_events to increase.")
        try:
            runner.run_forever(timeout=1.0)
        except KeyboardInterrupt:
            print("\nBinance feed stopped.")
        except RuntimeError as exc:
            logging.error("Live execution could not start: %s", exc)
            print(f"\n[live] {exc}")
        return

    if settings.mode == "backtest":
        runner.start()
        try:
            runner.drain()
        finally:
            runner.stop()
        print(runner.stats)
        _keep_api_alive_if_requested(api_enabled, api_host=api_host, api_port=api_port)
        return

    if settings.mode == "paper":
        runner.start()
        try:
            runner.drain()
        finally:
            runner.stop()
        print(runner.stats)
        _keep_api_alive_if_requested(api_enabled, api_host=api_host, api_port=api_port)
        return

    try:
        runner.run_forever()
    except KeyboardInterrupt:
        print("\nTrading system stopped.")


def _default_mock_ticks() -> list[MockTick]:
    ticks: list[MockTick] = []
    ts = 1_700_000_000_000_000_000
    for i, sym in enumerate(settings.symbols):
        base = 100.0 + i * 10.0
        event = BookTickEvent.of(
            ts + i,
            sym,
            [
                PriceLevel(base, 10.0),
                PriceLevel(base - 0.01, 8.0),
                PriceLevel(base - 0.02, 6.0),
            ],
            [
                PriceLevel(base + 0.01, 1.0),
                PriceLevel(base + 0.02, 1.0),
                PriceLevel(base + 0.03, 1.0),
            ],
        )
        ticks.append(MockTick(ts_ns=event.ts_ns, event=event))
    return ticks


def _keep_api_alive_if_requested(api_enabled: bool, *, api_host: str, api_port: int) -> None:
    """
    Mock/CSV feeds are finite, so runner.drain() returns quickly.
    If the status API is enabled, keep the process alive so the browser can inspect
    the final StatusStore snapshot.
    """
    if not api_enabled:
        return
    print(
        f"Status dashboard is still running at http://{api_host}:{api_port}/ . "
        "Press Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStatus API stopped.")


def _print_startup_banner(
    *,
    market_feed: str,
    execution: str,
    api_enabled: bool,
    api_host: str,
    api_port: int,
) -> None:
    symbols = ",".join(settings.symbols)
    print(
        "Starting trading system: "
        f"mode={settings.mode} market_feed={market_feed} execution={execution} symbols={symbols}"
    )
    print(
        f"Strategy={os.getenv('STRATEGY', 'microstructure')} | Microstructure strategy: "
        f"levels={os.getenv('MICRO_LEVELS', '3')} "
        f"entry={os.getenv('MICRO_ENTRY_THRESHOLD', '0.35')} "
        f"exit={os.getenv('MICRO_EXIT_THRESHOLD', '0.10')} "
        f"max_spread_bps={os.getenv('MICRO_MAX_SPREAD_BPS', '8.0')} "
        f"cooldown_ns={os.getenv('MICRO_COOLDOWN_NS', '1000000000')}"
    )
    if api_enabled:
        print(f"Dashboard: http://{api_host}:{api_port}/")
        print(f"Status JSON: http://{api_host}:{api_port}/status")
    else:
        print("Status API disabled. Set START_API=1 to enable the dashboard.")


if __name__ == "__main__":
    run()
