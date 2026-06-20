"""events.py —— 系统通用语言（protocol layer）

所有跨模块通信只能传递本文件定义的事件对象。
本文件零业务依赖，只 import 标准库。
"""


from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Iterable, Mapping, Optional


# ============================================================
# 1. 枚举（与币安 REST/WS 字段对齐，方便 gateway 零翻译）
# ============================================================

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"

class TimeInForce(str, Enum):
    GTC = "GTC"   # Good Till Cancel
    IOC = "IOC"   # Immediate Or Cancel
    FOK = "FOK"   # Fill Or Kill

class OrderStatus(str, Enum):
    PENDING_NEW = "PENDING_NEW"   # 本地已创建，未发出
    NEW = "NEW"                   # 交易所已接受
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class ExecutionType(str, Enum):
    """交易所回报里的 executionReport.x 字段"""
    NEW = "NEW"
    TRADE = "TRADE"          # 成交
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class SignalAction(IntEnum):
    """策略输出意图，不含具体下单参数"""
    FLAT = 0      # 平仓 / 不持仓
    LONG = 1
    SHORT = -1

def freeze_meta(meta: Optional[Mapping[str, Any]]) -> Optional[tuple[tuple[str, Any], ...]]:
    """
    防御性冻结：把可变 dict/meta 变成不可变结构，避免下游修改。

    说明：
    - dataclass frozen 只保证“字段引用不可变”，但内嵌 dict/list 仍可能被修改
    - 因此这里做“深冻结”的近似：把 dict/list/tuple 都转换成不可变 tuple
    """
    if meta is None:
        return None

    def _freeze_value(v: Any) -> Any:
        # 常见可 JSON 序列化/不可变类型直接返回
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        # 映射/序列做递归冻结
        if isinstance(v, Mapping):
            return tuple((str(k), _freeze_value(val)) for k, val in v.items())
        if isinstance(v, (list, tuple)):
            return tuple(_freeze_value(x) for x in v)
        # 兜底：对不可预期类型保留引用（meta 本身是调试用途；若要严格，可后续再扩展）
        return v

    return tuple((str(k), _freeze_value(val)) for k, val in meta.items())


# ============================================================
# 2. 工具
# ============================================================

def now_ns() -> int:
    """统一时钟入口；回测会被 SimClock monkey-patch。"""
    return time.time_ns()


# ============================================================
# 3. 行情类事件（feed → book / strategy）
# ============================================================

@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: float
    qty: float

@dataclass(frozen=True, slots=True)
class BookTickEvent:
    """盘口快照（top N levels）"""
    ts_ns: int
    symbol: str
    bids: tuple[PriceLevel, ...]   # 用 tuple 冻结，禁止 append
    asks: tuple[PriceLevel, ...]
    last_update_id: Optional[int] = None  # 来自交易所的快照序列号（用于增量簿一致性校验）

    @staticmethod
    def of(
        ts_ns: int,
        symbol: str,
        bids: Iterable[PriceLevel],
        asks: Iterable[PriceLevel],
        *,
        last_update_id: Optional[int] = None,
    ) -> "BookTickEvent":
        # 防御性拷贝：传入 list 也强制冻结
        return BookTickEvent(
            ts_ns,
            symbol,
            tuple(bids),
            tuple(asks),
            last_update_id=last_update_id,
        )


@dataclass(frozen=True, slots=True)
class BookDeltaEvent:
    """
    盘口增量更新（用于 Binance depthUpdate 风格）。

    语义：
    - first_update_id/final_update_id 给出该批更新覆盖的序列区间
    - bids/asks 中 qty==0 表示删除该档位
    """

    ts_ns: int
    symbol: str
    first_update_id: int
    final_update_id: int
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]

    @staticmethod
    def of(
        ts_ns: int,
        symbol: str,
        first_update_id: int,
        final_update_id: int,
        bids: Iterable[PriceLevel],
        asks: Iterable[PriceLevel],
    ) -> "BookDeltaEvent":
        return BookDeltaEvent(
            ts_ns=ts_ns,
            symbol=symbol,
            first_update_id=first_update_id,
            final_update_id=final_update_id,
            bids=tuple(bids),
            asks=tuple(asks),
        )

@dataclass(frozen=True, slots=True)
class TradeEvent:
    """逐笔成交（交易所推送的 public trade）"""
    ts_ns: int
    symbol: str
    price: float
    qty: float
    is_buyer_maker: bool   # True = 主动卖单吃买盘


# 统一的市场事件类型（feed -> book/strategy）
MarketEvent = BookTickEvent | BookDeltaEvent | TradeEvent


# ============================================================
# 4. 策略 → 风控
# ============================================================

@dataclass(frozen=True, slots=True)
class SignalEvent:
    """策略只表达"想做什么"，不含 size/price，由 risk+portfolio 决定"""
    ts_ns: int
    symbol: str
    strategy_id: str
    action: SignalAction
    strength: float = 1.0          # [0,1]，给组合权重用
    # 可选调试信息：要求生产方用 freeze_meta() 冻结成 tuple，避免 dict 在 frozen 下仍可变
    meta: Optional[tuple[tuple[str, Any], ...]] = None


# ============================================================
# 5. 风控 → broker
# ============================================================

@dataclass(frozen=True, slots=True)
class OrderEvent:
    """已通过风控、准备发往交易所的订单"""
    ts_ns: int
    client_order_id: str           # 本地唯一，幂等键
    symbol: str
    side: Side
    order_type: OrderType
    qty: float
    price: Optional[float] = None  # MARKET 时为 None
    stop_price: Optional[float] = None  # STOP_* 时为触发价；STOP_LIMIT 同时需要 price=限价
    tif: TimeInForce = TimeInForce.GTC
    strategy_id: str = ""          # 溯源用
    account_id: str = "default"    # 预留多账户扩展


@dataclass(frozen=True, slots=True)
class CancelOrderEvent:
    """
    撤单意图（risk/runner -> broker）
    由 broker 根据 orig_client_order_id 映射交易所订单并发起 cancel。
    """
    ts_ns: int
    account_id: str = "default"
    cancel_client_order_id: str = ""   # 用于幂等（不同实现也可以直接复用 orig id）
    orig_client_order_id: str = ""     # 要撤的订单（来自 OrderEvent.client_order_id）
    symbol: str = ""
    strategy_id: str = ""


@dataclass(frozen=True, slots=True)
class ModifyOrderEvent:
    """
    改单/替换订单意图（risk/runner -> broker）
    有些交易所不支持“原地修改”，broker 可以实现为：cancel + new（但事件层仍表达“替换”语义）。
    """
    ts_ns: int
    account_id: str = "default"
    modify_client_order_id: str = ""   # 用于幂等
    orig_client_order_id: str = ""     # 要替换的订单
    symbol: str = ""
    side: Optional[Side] = None         # 如果 None，则表示不变（由 broker 读取原单决定）
    qty: Optional[float] = None        # 如果 None，则表示不变
    price: Optional[float] = None      # limit price（若用于 LIMIT/STOP_LIMIT）
    stop_price: Optional[float] = None # trigger price（若用于 STOP_*）
    tif: Optional[TimeInForce] = None
    strategy_id: str = ""


# ============================================================
# 6. broker → portfolio / recorder
# ============================================================

@dataclass(frozen=True, slots=True)
class OrderAckEvent:
    """交易所接受订单的确认"""
    ts_ns: int
    client_order_id: str
    exchange_order_id: str
    status: OrderStatus
    account_id: str = "default"

@dataclass(frozen=True, slots=True)
class FillEvent:
    """单次成交回报（可能多次 fill 拼成一个 order）"""
    ts_ns: int
    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: Side
    fill_price: float
    fill_qty: float
    cum_qty: float                 # 累计已成交
    leaves_qty: float              # 剩余
    fee: float
    fee_asset: str
    status: OrderStatus            # PARTIALLY_FILLED / FILLED
    exec_type: ExecutionType = ExecutionType.TRADE
    # 每一次成交的唯一标识（用于 portfolio/record 去重）。
    # 若交易所返回字段不完整，broker 可在本地生成稳定 hash，但必须保持跨重连一致。
    exchange_trade_id: Optional[str] = None
    account_id: str = "default"

@dataclass(frozen=True, slots=True)
class OrderRejectEvent:
    ts_ns: int
    client_order_id: str
    reason: str
    status: OrderStatus = OrderStatus.REJECTED
    account_id: str = "default"


# ============================================================
# 7. 系统类事件（risk / runner / recorder）
# ============================================================

class RiskAction(str, Enum):
    HALT = "HALT"                  # 熔断：拒绝所有新单
    RESUME = "RESUME"
    KILL = "KILL"                  # kill switch：撤所有单 + 平仓
    WARN = "WARN"

@dataclass(frozen=True, slots=True)
class RiskEvent:
    ts_ns: int
    action: RiskAction
    reason: str
    source: str                    # 哪个检查项触发的


# ============================================================
# 8. 账户类事件（旁路/可选：为资金/持仓/合约元数据预留）
# ============================================================

@dataclass(frozen=True, slots=True)
class BalanceUpdateEvent:
    ts_ns: int
    account_id: str
    asset: str
    balance: float
    available: float
    locked: float = 0.0


@dataclass(frozen=True, slots=True)
class AccountPositionEvent:
    """
    持仓更新（spot 可以表达 qty/均价；合约可以表达 合约持仓与方向）
    """
    ts_ns: int
    account_id: str
    symbol: str
    position_qty: float           # spot/合约统一用数量表达（合约正负表示方向也可）
    avg_price: float
    mark_price: Optional[float] = None


@dataclass(frozen=True, slots=True)
class FundingEvent:
    """
    合约资金费率更新（现货可不使用）
    """
    ts_ns: int
    account_id: str
    symbol: str
    funding_rate: float
    funding_amount: float
    next_funding_time_ns: Optional[int] = None


@dataclass(frozen=True, slots=True)
class LiquidationEvent:
    """
    强平/清算事件（合约可用；现货不使用）
    """
    ts_ns: int
    account_id: str
    symbol: str
    liquidation_price: Optional[float] = None
    pnl_realized: Optional[float] = None
    liquidation_id: Optional[str] = None
