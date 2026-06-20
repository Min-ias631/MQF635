"""
config.py —— 全局配置中心（零业务依赖）

原则：
1) 所有可调参数集中在这里，避免硬编码散落到各模块
2) API Key/Secret 从环境变量或 secrets.toml 读取，不写死在代码里
   （环境变量优先；secrets.toml 作为兜底，方便本地持久化凭证）
3) MODE 支持：live | backtest | paper
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


def _load_secret_file() -> dict:
    """Load credentials from the single project secrets file (fallback for env vars).

    Canonical location is `.streamlit/secrets.toml` — the SAME file the Streamlit UI
    reads via st.secrets — so there is one secrets file for the whole project.
    Search order: $SECRETS_FILE, ./.streamlit/secrets.toml, ./secrets.toml (legacy).
    Keys: BINANCE_API_KEY, BINANCE_API_SECRET (or BINANCE_SECRET), BINANCE_TESTNET,
    MODE, ... Explicit environment variables always take precedence.
    """
    here = Path(__file__).resolve().parent
    candidates = []
    if os.getenv("SECRETS_FILE"):
        candidates.append(Path(os.getenv("SECRETS_FILE")))
    candidates += [
        here / ".streamlit" / "secrets.toml",   # shared with the Streamlit UI
        here / "secrets.toml",                   # legacy fallback
    ]
    for path in candidates:
        try:
            if path and path.is_file():
                with open(path, "rb") as f:
                    return tomllib.load(f)
        except Exception:
            continue
    return {}


_SECRETS = _load_secret_file()


def _raw(name: str):
    """Env var first, then secrets.toml; '' / None treated as unset."""
    v = os.getenv(name)
    if v is None or v == "":
        v = _SECRETS.get(name)
    if v is None or v == "":
        return None
    return v


def _get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = _raw(name)
    return str(v) if v is not None else default


def _get_env_float(name: str, default: float) -> float:
    v = _raw(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={v!r}: expected float") from e


def _get_env_int(name: str, default: int) -> int:
    v = _raw(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={v!r}: expected int") from e


def _get_env_bool(name: str, default: bool) -> bool:
    v = _raw(name)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _split_symbols(v: Optional[str]) -> tuple[str, ...]:
    """
    支持：
    - 不提供：使用默认
    - 提供：例如 "BTCUSDT,ETHUSDT"
    """
    if v is None or v.strip() == "":
        return ("BTCUSDT", "ETHUSDT")
    parts = [x.strip().upper() for x in v.split(",")]
    return tuple([p for p in parts if p])


MODE = Literal["live", "backtest", "paper"]


@dataclass(frozen=True, slots=True)
class Settings:
    # ----------------------------
    # Exchange config
    # ----------------------------
    mode: MODE
    binance_api_key: Optional[str]
    binance_secret: Optional[str]

    # ----------------------------
    # Instruments
    # ----------------------------
    symbols: tuple[str, ...]

    # ----------------------------
    # Strategy parameters (example knobs)
    # ----------------------------
    # 用于示例/第一阶段调参；后续你可以按策略需要细化
    window_length: int
    signal_threshold: float
    strategy_weight: float

    # ----------------------------
    # Risk parameters
    # ----------------------------
    max_single_order_qty: float
    daily_max_loss: float
    max_total_position_qty: float
    max_order_rate_per_symbol: int  # 简单频率限制：每 symbol 每秒多少（由 runner/rate limiter 实现）

    # ----------------------------
    # Reliability / circuit breaker
    # ----------------------------
    max_consecutive_order_failures: int

    # ----------------------------
    # Runtime / IO
    # ----------------------------
    event_queue_maxsize: int
    log_level: str
    data_dir: str

    # Defaulted (kept last so existing Settings(...) construction stays compatible)
    binance_testnet: bool = True


def load_settings() -> Settings:
    mode_raw = (_get_env_str("MODE", "paper") or "paper").lower().strip()
    if mode_raw not in ("live", "backtest", "paper"):
        raise ValueError(f"Invalid MODE={mode_raw!r}, expected: live | backtest | paper")

    api_key = _get_env_str("BINANCE_API_KEY", None)
    # Accept both the UI's key name (BINANCE_API_SECRET) and the legacy one
    # (BINANCE_SECRET) so a single .streamlit/secrets.toml works everywhere.
    api_secret = _get_env_str("BINANCE_API_SECRET", None) or _get_env_str("BINANCE_SECRET", None)

    # live 模式要求凭证；paper/backtest 可以允许为空（避免你本地没 key 也不能跑）
    if mode_raw == "live":
        if not api_key or not api_secret:
            raise RuntimeError("BINANCE_API_KEY / BINANCE_SECRET must be set in live mode")

    return Settings(
        mode=mode_raw,  # type: ignore[arg-type]
        binance_api_key=api_key,
        binance_secret=api_secret,
        binance_testnet=_get_env_bool("BINANCE_TESTNET", True),
        symbols=_split_symbols(_get_env_str("SYMBOLS", None)),
        window_length=_get_env_int("WINDOW_LENGTH", 60),
        signal_threshold=_get_env_float("SIGNAL_THRESHOLD", 2.0),
        strategy_weight=_get_env_float("STRATEGY_WEIGHT", 1.0),
        max_single_order_qty=_get_env_float("MAX_SINGLE_ORDER_QTY", 0.01),
        daily_max_loss=_get_env_float("DAILY_MAX_LOSS", 50.0),
        max_total_position_qty=_get_env_float("MAX_TOTAL_POSITION_QTY", 0.02),
        max_order_rate_per_symbol=_get_env_int("MAX_ORDER_RATE_PER_SYMBOL", 5),
        max_consecutive_order_failures=_get_env_int("MAX_CONSECUTIVE_ORDER_FAILURES", 3),
        event_queue_maxsize=_get_env_int("EVENT_QUEUE_MAXSIZE", 10000),
        log_level=_get_env_str("LOG_LEVEL", "INFO") or "INFO",
        data_dir=_get_env_str("DATA_DIR", "./data") or "./data",
    )


# 全局配置实例：供其它模块导入使用
settings = load_settings()


__all__ = ["Settings", "settings"]

