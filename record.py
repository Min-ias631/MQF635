"""
record.py —— 异步事件落盘（旁路，不阻塞 runner）

职责：
- 接收任意 event 对象，序列化后写入 JSONL 文件
- 后台线程负责 IO；主线程 log() 只做 put_nowait

用法（runner 旁路调用）::
    recorder = EventRecorder()
    recorder.start()
    recorder.log(market_event)
    ...
    recorder.stop()
"""

from __future__ import annotations

import json
import queue
import threading
import time
from enum import Enum
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from config import settings
from event import (
    CancelOrderEvent,
    FillEvent,
    ModifyOrderEvent,
    OrderEvent,
    OrderRejectEvent,
    RiskEvent,
)


def _to_jsonable(obj: Any) -> Any:
    """把 event / enum / dataclass 递归转成可 JSON 序列化的结构。"""
    if is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    # Enum 直接转成 value（OrderStatus/Side 等）
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def serialize_event(event: Any) -> dict[str, Any]:
    return {
        "event_type": type(event).__name__,
        "recorded_at_ns": time.time_ns(),
        "payload": _to_jsonable(event),
    }


class EventRecorder:
    """
    异步 JSONL 记录器。

    - 队列满时丢弃并计数（避免阻塞交易主循环）
    - 默认输出到 settings.data_dir / logs / events.jsonl
    """

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        max_queue_size: int | None = None,
        filename: str = "events.jsonl",
    ) -> None:
        self._data_dir = Path(data_dir or settings.data_dir)
        self._log_dir = self._data_dir / "logs"
        self._log_path = self._log_dir / filename
        self._deadletter_path = self._log_dir / "events_deadletter.jsonl"
        self._max_queue_size = max_queue_size or settings.event_queue_maxsize

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=self._max_queue_size)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dropped = 0
        self._written = 0
        self._dl_lock = threading.Lock()

    @staticmethod
    def _is_critical_event(event: Any) -> bool:
        # 这些事件静默丢失会导致“状态不可追溯”，需要尽量落盘
        return isinstance(
            event,
            (
                OrderEvent,
                CancelOrderEvent,
                ModifyOrderEvent,
                FillEvent,
                OrderRejectEvent,
                RiskEvent,
            ),
        )

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def written_count(self) -> int:
        return self._written

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._worker = threading.Thread(target=self._run, name="event-recorder", daemon=True)
        self._worker.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """优雅停止：发送哨兵，等待队列写完。"""
        if not self._worker:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)
        self._worker = None

    def log(self, event: Any) -> bool:
        """
        非阻塞写入请求。
        返回 True 表示已入队；False 表示队列满被丢弃。
        """
        if self._stop_event.is_set():
            return False
        line = json.dumps(serialize_event(event), ensure_ascii=False, separators=(",", ":"))
        try:
            self._queue.put_nowait(line)
            return True
        except queue.Full:
            self._dropped += 1
            # 对关键事件做 best-effort dead-letter（避免“静默丢”）
            if self._is_critical_event(event):
                try:
                    with self._dl_lock:
                        with self._deadletter_path.open("a", encoding="utf-8") as f:
                            f.write(line)
                            f.write("\n")
                except Exception:
                    # 最后兜底：别让观察层影响主链路
                    pass
            return False

    def _run(self) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                f.write(item)
                f.write("\n")
                self._written += 1
                if self._stop_event.is_set() and self._queue.empty():
                    break


__all__ = ["EventRecorder", "serialize_event"]
