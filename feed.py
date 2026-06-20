"""
feed.py —— 行情接入（Market Data Feed Handler）

职责：
- 把外部数据源（实盘 WebSocket / CSV / mock）转成内部的 MarketEvent（BookTickEvent / TradeEvent）
- 通过队列把事件送入 runner 主循环

本文件仅定义接口与基础实现；真正的 Binance 连接逻辑可后续细化。
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional

from config import settings
from event import BookTickEvent, FundingEvent, MarketEvent, PriceLevel, TradeEvent, now_ns

logger = logging.getLogger(__name__)


def _parse_bool(value: str) -> bool:
    """Parse a CSV boolean cell (e.g. is_buyer_maker): true/1/yes -> True."""
    return str(value).strip().lower() in ("1", "true", "t", "yes", "y")


# ============================================================
# 抽象基类
# ============================================================


class MarketDataFeed(ABC):
    """
    行情源抽象接口。

    - start() / stop() 控制生命周期
    - 通过外部提供的 Queue[MarketEvent] 输出事件
    """

    def __init__(self, out_queue: Queue[MarketEvent]) -> None:
        self._out_queue = out_queue

    @abstractmethod
    def start(self) -> None:  # pragma: no cover - 接口
        ...

    @abstractmethod
    def stop(self) -> None:  # pragma: no cover - 接口
        ...


# ============================================================
# MockFeed —— 单元测试 / 本地开发
# ============================================================


@dataclass(frozen=True, slots=True)
class _MockTick:
    ts_ns: int
    event: MarketEvent


MockTick = _MockTick


class MockFeed(MarketDataFeed):
    """
    从预先构造的事件序列中依次吐出 MarketEvent。

    用途：
    - 单元测试 runner / strategy / risk
    - 本地开发不连交易所时快速自测
    """

    def __init__(self, out_queue: Queue[MarketEvent], ticks: Iterable[_MockTick]) -> None:
        super().__init__(out_queue)
        self._ticks = list(ticks)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mock-feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        for tick in self._ticks:
            if self._stop.is_set():
                break
            try:
                self._out_queue.put_nowait(tick.event)
            except Exception:
                # 若队列满，直接丢弃（测试环境下不阻塞）
                continue
            # 简单按时间戳节流：若下一条在未来，则 sleep 一小会（可选）
        # 结束即自然退出


# ============================================================
# CSVFeed —— 回测 / 复盘
# ============================================================


class CSVFeed(MarketDataFeed):
    """
    从 CSV 文件读取历史行情，按时间顺序吐出 MarketEvent。

    这里做一个简单示例格式（你后续可以按自己的数据格式调整）：
    - 行情文件包含列：ts_ns,symbol,bid,ask
    - 逐笔文件包含列：ts_ns,symbol,price,qty,is_buyer_maker
    - 资金费率文件包含列：ts_ns,symbol,funding_rate
    book/trade/funding 事件按 ts_ns 归并排序后顺序吐出，使回测拿到真实逐笔
    成交（真实 TFI，而非用盘口不平衡合成的代理）与真实资金费率。

    注意：funding 行被视为结算事件 —— runner 在收到 FundingEvent 时会对当前
    持仓结算资金费（多头支付正费率）。因此 funding_csv 的行应对应真实的
    资金费结算/发布时间戳（Binance 为每 8 小时），而不是任意的费率采样。
    """

    def __init__(
        self,
        out_queue: Queue[MarketEvent],
        *,
        book_csv: Optional[str] = None,
        trade_csv: Optional[str] = None,
        funding_csv: Optional[str] = None,
        background: bool | None = None,
    ) -> None:
        super().__init__(out_queue)
        self._book_csv = Path(book_csv) if book_csv else None
        self._trade_csv = Path(trade_csv) if trade_csv else None
        self._funding_csv = Path(funding_csv) if funding_csv else None
        # backtest 更希望可控的同步行为；paper/live 更希望后台避免阻塞主循环
        if background is None:
            self._background = settings.mode != "backtest"
        else:
            self._background = bool(background)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        if self._background:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="csv-feed", daemon=True)
            self._thread.start()
        else:
            # backtest 同步模式：阻塞 put，保证历史行情不丢、回测可复现
            for ev in self._iter_events():
                if self._stop.is_set():
                    break
                self._out_queue.put(ev)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        for ev in self._iter_events():
            if self._stop.is_set():
                break
            try:
                self._out_queue.put_nowait(ev)
            except Exception:
                continue

    def _iter_events(self) -> Iterable[object]:
        # 归并 book/trade/funding 三类事件，按 ts_ns 升序回放。
        events: list[tuple[int, object]] = []

        if self._book_csv and self._book_csv.exists():
            with self._book_csv.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    ts_ns = int(row["ts_ns"])
                    events.append((
                        ts_ns,
                        BookTickEvent.of(
                            ts_ns,
                            row["symbol"].upper(),
                            [PriceLevel(price=float(row["bid"]), qty=1.0)],
                            [PriceLevel(price=float(row["ask"]), qty=1.0)],
                        ),
                    ))

        if self._trade_csv and self._trade_csv.exists():
            with self._trade_csv.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    ts_ns = int(row["ts_ns"])
                    events.append((
                        ts_ns,
                        TradeEvent(
                            ts_ns=ts_ns,
                            symbol=row["symbol"].upper(),
                            price=float(row["price"]),
                            qty=float(row["qty"]),
                            is_buyer_maker=_parse_bool(row["is_buyer_maker"]),
                        ),
                    ))

        if self._funding_csv and self._funding_csv.exists():
            with self._funding_csv.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    ts_ns = int(row["ts_ns"])
                    events.append((
                        ts_ns,
                        FundingEvent(
                            ts_ns=ts_ns,
                            account_id="default",
                            symbol=row["symbol"].upper(),
                            funding_rate=float(row["funding_rate"]),
                            funding_amount=0.0,
                        ),
                    ))

        # 稳定排序：同一 ts_ns 时保持 book -> trade -> funding 的插入顺序。
        events.sort(key=lambda item: item[0])
        for _ts, ev in events:
            yield ev


# ============================================================
# BinancePublicFeed —— Binance public market data only
# ============================================================


class BinancePublicFeed(MarketDataFeed):
    """
    Binance 公开行情 WebSocket。

    只订阅 public streams，不需要 API key，不具备真实下单能力。
    默认使用 combined stream:
    - <symbol>@depth<levels>@100ms
    - <symbol>@trade
    """

    def __init__(
        self,
        out_queue: Queue[MarketEvent],
        *,
        symbols: tuple[str, ...] | None = None,
        depth_levels: int = 5,
        reconnect_delay_sec: float = 3.0,
        base_url: str = "wss://stream.binance.com:9443/stream?streams=",
    ) -> None:
        super().__init__(out_queue)
        self._symbols = tuple(s.lower() for s in (symbols or settings.symbols))
        self._depth_levels = depth_levels
        self._reconnect_delay_sec = max(0.1, float(reconnect_delay_sec))
        self._base_url = base_url
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ws = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ensure_websocket_client()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="binance-public-feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._ws = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                logger.info("Connecting Binance public stream: %s", self._url())
                websocket = self._load_websocket_module()
                self._ws = websocket.WebSocketApp(
                    self._url(),
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                logger.exception("Binance public feed loop crashed; reconnecting soon")
                if self._stop.is_set():
                    break
            if not self._stop.is_set():
                logger.warning(
                    "Binance public stream disconnected; reconnecting in %.1fs",
                    self._reconnect_delay_sec,
                )
                time.sleep(self._reconnect_delay_sec)

    def _url(self) -> str:
        streams: list[str] = []
        for sym in self._symbols:
            streams.append(f"{sym}@depth{self._depth_levels}@100ms")
            streams.append(f"{sym}@trade")
        return self._base_url + "/".join(streams)

    def _on_message(self, _ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
            data = msg.get("data", msg)
            stream = msg.get("stream", "")
            event = self._parse_data(data, stream=stream)
        except Exception:
            return
        if event is None:
            return
        try:
            self._out_queue.put_nowait(event)
        except Exception:
            # Market data is lossy by design at this boundary; book sequence/validity handles safety.
            pass

    def _parse_data(self, data: dict, *, stream: str = "") -> MarketEvent | None:
        event_type = data.get("e")
        if event_type == "trade":
            return self._parse_trade(data)
        if "lastUpdateId" in data and "bids" in data and "asks" in data:
            return self._parse_partial_depth(data, stream=stream)
        return None

    @staticmethod
    def _parse_trade(data: dict) -> TradeEvent:
        ts_ms = int(data.get("T") or data.get("E") or int(time.time() * 1000))
        return TradeEvent(
            ts_ns=ts_ms * 1_000_000,
            symbol=str(data["s"]).upper(),
            price=float(data["p"]),
            qty=float(data["q"]),
            is_buyer_maker=bool(data["m"]),
        )

    @staticmethod
    def _parse_partial_depth(data: dict, *, stream: str = "") -> BookTickEvent:
        ts_ns = now_ns()
        bids = tuple(PriceLevel(price=float(px), qty=float(qty)) for px, qty in data["bids"])
        asks = tuple(PriceLevel(price=float(px), qty=float(qty)) for px, qty in data["asks"])
        symbol = str(data.get("s") or data.get("symbol") or "").upper()
        if not symbol and stream:
            symbol = stream.split("@", 1)[0].upper()
        if not symbol:
            raise ValueError("partial depth payload missing symbol")
        return BookTickEvent.of(
            ts_ns,
            symbol,
            bids,
            asks,
            last_update_id=int(data["lastUpdateId"]),
        )

    @staticmethod
    def _on_error(_ws, error) -> None:
        logger.warning("Binance public stream error: %s", error)

    @staticmethod
    def _on_close(_ws, status_code, msg) -> None:
        logger.info("Binance public stream closed: status=%s msg=%s", status_code, msg)

    @staticmethod
    def _load_websocket_module():
        try:
            import websocket  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "BinancePublicFeed requires websocket-client. "
                "Install with: python3.12 -m pip install websocket-client"
            ) from e
        return websocket

    def _ensure_websocket_client(self) -> None:
        self._load_websocket_module()


# ============================================================
# LiveFeed —— real trading feed placeholder
# ============================================================


class LiveFeed(BinancePublicFeed):
    """
    真实交易系统的行情源占位。

    目前仅继承 public market data。真实 live trading 仍必须另外实现 LiveBroker、
    账户流、订单回报和重连恢复。
    """

    pass


__all__ = ["MarketDataFeed", "MockFeed", "MockTick", "CSVFeed", "BinancePublicFeed", "LiveFeed"]
