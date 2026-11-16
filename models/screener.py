from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RuleModel(BaseModel):
    type: str
    on: bool = True
    _id: Optional[str] = None
    # generic parameters
    nmin: Optional[int] = None
    nmax: Optional[int] = None
    vol: Optional[float] = None
    max_up: Optional[int] = None
    each: Optional[bool] = None
    dmin: Optional[float] = None
    dmax: Optional[float] = None
    x: Optional[float] = None
    days: Optional[int] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    n: Optional[int] = None
    dir: Optional[str] = None
    kind: Optional[str] = None
    len: Optional[int] = None
    k1: Optional[str] = None
    n1: Optional[int] = None
    k2: Optional[str] = None
    n2: Optional[int] = None
    within: Optional[int] = None
    tol: Optional[float] = None
    legs: Optional[int] = None
    mode: Optional[str] = None
    pat: Optional[str] = None
    skip: Optional[int] = None
    amin: Optional[int] = None
    amax: Optional[int] = None
    max: Optional[float] = None
    # common options
    ago: Optional[int] = 0
    tf: Optional[str] = "daily"
    grp: Optional[str] = "—"
    must: Optional[bool] = False


class SetupCreateRequest(BaseModel):
    id: Optional[int] = None
    name: str
    price_min: float = 0.0
    price_max: float = 99999.0
    visual_rules: Optional[str] = ""
    min_pass: Optional[int] = 0
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    owner: Optional[str] = "admin"


class SetupResponse(BaseModel):
    id: int
    name: str
    price_min: float
    price_max: float
    visual_rules: Optional[str] = ""
    rule_config: Optional[Dict[str, Any]] = None
    owner: Optional[str] = "admin"
    created_at: Optional[str] = None


class ScanRequest(BaseModel):
    universe: str = "liquid"
    custom_symbols: Optional[List[str]] = None
    rules: List[Dict[str, Any]]
    price_min: float = 20.0
    price_max: float = 50000.0
    min_turnover: float = 0.0
    want_near: bool = True
    min_pass: int = 0


class RuleCheck(BaseModel):
    ok: bool
    chip: str
    detail: str
    label: str
    desc: str
    grp: str
    must: bool
    type: str


class StockResult(BaseModel):
    symbol: str
    close: float
    chg: float
    turnover: float
    vol_x: Optional[float] = None
    rsi: Optional[float] = None
    checks: List[RuleCheck]
    fails: List[Any]
    status: str
    passed: int
    total: int
    miss: str
    metrics: Dict[str, Any]
    date: str
    projected: bool


class FunnelItem(BaseModel):
    desc: str
    pass_: int = Field(alias="pass")
    only: int
    model_config = {"populate_by_name": True}


class ScanStats(BaseModel):
    universe: int
    priced: int
    in_band: int
    liquid: int
    match: int
    near: int
    secs: float


class ScanResponse(BaseModel):
    match: List[StockResult]
    near: List[StockResult]
    stats: ScanStats
    funnel: List[FunnelItem]
    scanned_at: str
