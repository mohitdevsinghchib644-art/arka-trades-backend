import math
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .indicators import (
    _rma_series,
    rsi_wilder,
    moving_average,
    avg_volume,
    atr,
    macd,
    adx,
    detect_candle,
)
from .market_data import (
    get_snapshots,
    get_histories,
    get_nifty_close,
    live_volume_fraction,
)

IST = timezone(timedelta(hours=5, minutes=30))

MA_KINDS = ["EMA", "SMA"]
CANDLES = [
    "hammer", "bullish engulfing", "bearish engulfing", "doji",
    "strong close", "weak close", "green candle", "red candle"
]

RULES = {
    "pullback": {
        "label": "Pullback on low volume",
        "fields": [
            ("nmin", "Min candles", "i", 2, 1, 10),
            ("nmax", "Max candles", "i", 4, 1, 10),
            ("max_up", "Up candles allowed", "i", 0, 0, 5),
            ("vol", "Avg volume ≤ × 20-day avg", "f", 0.8, 0.1, 2.0),
            ("each", "Every candle below avg volume", "b", False, None, None),
            ("dmin", "Depth min %", "f", 0.0, 0.0, 50.0),
            ("dmax", "Depth max % (0 = any)", "f", 0.0, 0.0, 60.0),
        ],
    },
    "lowvol": {
        "label": "Consecutive low-volume candles",
        "fields": [
            ("nmin", "Min candles", "i", 3, 1, 15),
            ("nmax", "Max candles (0 = any)", "i", 0, 0, 30),
            ("x", "Volume below × 20-day avg", "f", 1.0, 0.1, 2.0),
        ],
    },
    "runup": {
        "label": "Prior run-up (momentum)",
        "fields": [
            ("days", "Look-back candles", "i", 30, 5, 250),
            ("lo", "Rise at least %", "f", 20.0, 0.0, 1000.0),
            ("hi", "Rise at most % (0 = any)", "f", 0.0, 0.0, 5000.0),
        ],
    },
    "rsi": {
        "label": "RSI in range",
        "fields": [
            ("lo", "Min", "f", 50.0, 0.0, 100.0),
            ("hi", "Max", "f", 100.0, 0.0, 100.0),
            ("n", "Period", "i", 14, 2, 50),
        ],
    },
    "rsi_avg": {
        "label": "RSI vs its average",
        "fields": [
            ("dir", "RSI is", "s", "above", ["above", "below"], None),
            ("n", "Average length", "i", 14, 2, 50),
        ],
    },
    "ma_pos": {
        "label": "Price vs moving average",
        "fields": [
            ("kind", "Type", "s", "EMA", MA_KINDS, None),
            ("len", "Length", "i", 20, 2, 300),
            ("dir", "Price is", "s", "above", ["above", "below", "near"], None),
            ("dmin", "Distance min %", "f", 0.0, 0.0, 100.0),
            ("dmax", "Distance max % (0 = any)", "f", 0.0, 0.0, 200.0),
        ],
    },
    "ma_stack": {
        "label": "Moving-average alignment",
        "fields": [
            ("k1", "Fast type", "s", "EMA", MA_KINDS, None),
            ("n1", "Fast length", "i", 20, 2, 300),
            ("k2", "Slow type", "s", "EMA", MA_KINDS, None),
            ("n2", "Slow length", "i", 50, 2, 300),
        ],
    },
    "ma_cross": {
        "label": "Fresh MA crossover",
        "fields": [
            ("k1", "Fast type", "s", "EMA", MA_KINDS, None),
            ("n1", "Fast length", "i", 20, 2, 300),
            ("k2", "Slow type", "s", "EMA", MA_KINDS, None),
            ("n2", "Slow length", "i", 50, 2, 300),
            ("within", "Crossed within candles", "i", 5, 1, 60),
        ],
    },
    "vol_spike": {
        "label": "Volume vs 20-day average",
        "fields": [
            ("lo", "At least ×", "f", 1.5, 0.0, 50.0),
            ("hi", "At most × (0 = any)", "f", 0.0, 0.0, 100.0),
        ],
    },
    "atr": {
        "label": "ATR % (typical daily range)",
        "fields": [
            ("lo", "Min %", "f", 1.0, 0.0, 50.0),
            ("hi", "Max % (0 = any)", "f", 0.0, 0.0, 50.0),
            ("n", "Period", "i", 14, 2, 50),
        ],
    },
    "day_move": {
        "label": "Today's move %",
        "fields": [
            ("mode", "Measure", "s", "size", ["size", "up", "down"], None),
            ("lo", "Min %", "f", 2.0, 0.0, 50.0),
            ("hi", "Max % (0 = any)", "f", 0.0, 0.0, 50.0),
        ],
    },
    "change": {
        "label": "Price change over N candles",
        "fields": [
            ("days", "Candles", "i", 5, 1, 250),
            ("lo", "Min %", "f", 0.0, -100.0, 1000.0),
            ("hi", "Max %", "f", 100.0, -100.0, 5000.0),
        ],
    },
    "near_high": {
        "label": "Distance below N-day high",
        "fields": [
            ("days", "High of last candles", "i", 252, 5, 260),
            ("lo", "Min % below", "f", 0.0, 0.0, 100.0),
            ("hi", "Max % below", "f", 15.0, 0.0, 100.0),
        ],
    },
    "breakout": {
        "label": "Breakout above N-day high",
        "fields": [
            ("days", "Previous candles", "i", 20, 3, 260),
            ("amin", "Happened at least N candles ago", "i", 0, 0, 60),
            ("amax", "…and at most N candles ago", "i", 0, 0, 60),
        ],
    },
    "tight": {
        "label": "Tight range (consolidation)",
        "fields": [
            ("n", "Last candles", "i", 5, 2, 60),
            ("max", "High–low range at most %", "f", 6.0, 0.5, 50.0),
        ],
    },
    "nr": {
        "label": "Narrow-range candle (NR-N)",
        "fields": [("n", "Narrowest of last", "i", 7, 3, 30)],
    },
    "inside": {"label": "Inside bar (today inside yesterday)", "fields": []},
    "gap": {
        "label": "Gap at today's open %",
        "fields": [("lo", "Min %", "f", 1.0, -50.0, 50.0), ("hi", "Max %", "f", 20.0, -50.0, 50.0)],
    },
    "rs": {
        "label": "Relative strength vs NIFTY 50",
        "fields": [
            ("days", "Over candles", "i", 63, 5, 250),
            ("lo", "Beats NIFTY by at least % pts", "f", 0.0, -100.0, 500.0),
        ],
    },
    "macd": {
        "label": "MACD (12, 26, 9)",
        "fields": [("mode", "Condition", "s", "above signal", ["above signal", "above zero", "histogram rising"], None)],
    },
    "adx": {
        "label": "ADX trend strength",
        "fields": [
            ("lo", "Min", "f", 20.0, 0.0, 100.0),
            ("hi", "Max (0 = any)", "f", 0.0, 0.0, 100.0),
            ("n", "Period", "i", 14, 2, 50),
        ],
    },
    "candle": {
        "label": "Candle shape",
        "fields": [("pat", "Shape", "s", "hammer", CANDLES, None)],
    },
    "ma_touch": {
        "label": "Touched a moving average and held",
        "fields": [
            ("kind", "Type", "s", "EMA", MA_KINDS, None),
            ("len", "Length", "i", 20, 2, 300),
            ("within", "Touched within last candles", "i", 1, 1, 10),
            ("tol", "Touch tolerance %", "f", 0.5, 0.0, 5.0),
        ],
    },
    "contract": {
        "label": "Shrinking ranges (VCP style)",
        "fields": [
            ("legs", "Number of legs", "i", 3, 2, 4),
            ("n", "Candles per leg", "i", 5, 3, 20),
        ],
    },
    "vol_fall": {
        "label": "Volume falling candle after candle",
        "fields": [
            ("n", "Last candles", "i", 3, 2, 10),
            ("skip", "Exceptions allowed", "i", 0, 0, 3),
        ],
    },
    "delivery": {
        "label": "Delivery % (NSE)",
        "fields": [
            ("n", "Average of last sessions", "i", 5, 1, 20),
            ("lo", "At least %", "f", 40.0, 0.0, 100.0),
            ("x", "At least × its earlier average (0 = skip)", "f", 0.0, 0.0, 5.0),
        ],
    },
}

COMMON_FIELDS = [
    ("ago", "Measured N candles ago (0 = latest)", "i", 0, 0, 60),
    ("tf", "Candles", "s", "daily", ["daily", "weekly"], None),
    ("grp", "Either-or group", "s", "—", ["—", "A", "B", "C"], None),
    ("must", "Always required", "b", False, None, None),
]


def all_fields(rtype: str):
    return RULES.get(rtype, {}).get("fields", []) + COMMON_FIELDS


def new_rule(rtype: str, **vals) -> Dict[str, Any]:
    r = {"type": rtype, "on": True, "_id": uuid.uuid4().hex[:8]}
    for name, _, _, default, _, _ in all_fields(rtype):
        r[name] = default
    for k, v in vals.items():
        r[k] = v
    return clean_rule(r)


def clean_rule(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = r.get("type")
    if t not in RULES:
        return None
    out = {"type": t, "on": bool(r.get("on", True)), "_id": r.get("_id") or uuid.uuid4().hex[:8]}
    for name, _, kind, default, a, b in all_fields(t):
        v = r.get(name, default)
        try:
            if kind == "i":
                v = int(round(float(v)))
                v = max(a, min(b, v)) if (a is not None and b is not None) else v
            elif kind == "f":
                v = float(v)
                v = max(a, min(b, v)) if (math.isfinite(v) and a is not None and b is not None) else default
            elif kind == "s":
                v = str(v)
                v = v if v in a else default
            elif kind == "b":
                v = bool(v)
        except (TypeError, ValueError):
            v = default
        out[name] = v
    return out


def describe(r: Dict[str, Any]) -> str:
    t = r["type"]
    s = t
    if t == "pullback":
        s = f"Pullback {r.get('nmin', 2)}–{r.get('nmax', 4)} candles, vol ≤ {r.get('vol', 0.8):g}× 20D avg"
    elif t == "lowvol":
        s = f"Low volume {r.get('nmin', 3)}+ candles (< {r.get('x', 1.0):g}× avg)"
    elif t == "runup":
        s = f"Rose ≥ {r.get('lo', 20):g}% in last {r.get('days', 30)} candles"
    elif t == "rsi":
        s = f"RSI({r.get('n', 14)}) between {r.get('lo', 50):g} and {r.get('hi', 100):g}"
    elif t == "rsi_avg":
        s = f"RSI {r.get('dir', 'above')} its {r.get('n', 14)}-avg"
    elif t == "ma_pos":
        s = f"Price {r.get('dir', 'above')} {r.get('kind', 'EMA')}{r.get('len', 20)}"
    elif t == "ma_stack":
        s = f"{r.get('k1', 'EMA')}{r.get('n1', 20)} above {r.get('k2', 'SMA')}{r.get('n2', 50)}"
    elif t == "ma_cross":
        s = f"{r.get('k1', 'EMA')}{r.get('n1', 20)} crossed {r.get('k2', 'SMA')}{r.get('n2', 50)} in {r.get('within', 5)}c"
    elif t == "vol_spike":
        s = f"Today's vol ≥ {r.get('lo', 1.5):g}× 20D avg"
    elif t == "atr":
        s = f"ATR({r.get('n', 14)}) ≥ {r.get('lo', 1.0):g}% of price"
    elif t == "day_move":
        s = f"Today's move ≥ {r.get('lo', 2.0):g}%"
    elif t == "change":
        s = f"Change over {r.get('days', 5)}c between {r.get('lo', 0):g}% and {r.get('hi', 100):g}%"
    elif t == "near_high":
        s = f"≤ {r.get('hi', 15):g}% below {r.get('days', 252)}D high"
    elif t == "breakout":
        s = f"Breakout above {r.get('days', 20)}D high"
    elif t == "tight":
        s = f"Last {r.get('n', 5)}c inside {r.get('max', 6):g}% range"
    elif t == "nr":
        s = f"Narrowest range of last {r.get('n', 7)} candles (NR{r.get('n', 7)})"
    elif t == "inside":
        s = "Inside bar (High/Low inside prior bar)"
    elif t == "gap":
        s = f"Opened gap {r.get('lo', 1.0):g}%–{r.get('hi', 20.0):g}%"
    elif t == "rs":
        s = f"Outperforms NIFTY 50 by ≥ {r.get('lo', 0):g}% over {r.get('days', 63)}c"
    elif t == "macd":
        s = f"MACD {r.get('mode', 'above signal')}"
    elif t == "adx":
        s = f"ADX({r.get('n', 14)}) ≥ {r.get('lo', 20):g}"
    elif t == "candle":
        s = f"Candle shape is {r.get('pat', 'hammer')}"
    elif t == "ma_touch":
        s = f"Low tested {r.get('kind', 'EMA')}{r.get('len', 20)} (within {r.get('tol', 0.5):g}%) and held"
    elif t == "contract":
        s = f"Volatility contraction (VCP) {r.get('legs', 3)} legs"
    elif t == "vol_fall":
        s = f"Volume falling candle-over-candle ({r.get('n', 3)} candles)"

    if r.get("tf") == "weekly":
        s += " (weekly)"
    if r.get("ago"):
        s += f" ({r['ago']}c ago)"
    if r.get("must"):
        s += " [Required]"
    return s


def _t(*rules, **kw):
    return {"rules": list(rules), **kw}


TEMPLATES = {
    "Momentum pullback on dry volume": _t(
        ("runup", dict(days=30, lo=20)),
        ("rsi", dict(lo=50, hi=100)),
        ("ma_pos", dict(kind="EMA", len=20, dir="above")),
        ("ma_stack", dict(k1="EMA", n1=20, k2="SMA", n2=50)),
        ("pullback", dict(nmin=2, nmax=4, vol=0.8, max_up=1)),
        desc="Stock gained 20%+ in 30 days and holds above EMA 20, then pulls back 2–4 candles on low volume.",
    ),
    "Pullback to EMA 20 that held": _t(
        ("ma_stack", dict(k1="EMA", n1=20, k2="SMA", n2=50)),
        ("ma_pos", dict(kind="SMA", len=200, dir="above")),
        ("ma_touch", dict(kind="EMA", len=20, within=2, tol=0.5)),
        ("candle", dict(pat="strong close")),
        ("rsi", dict(lo=45, hi=70)),
        desc="Strong uptrend; a recent low touched EMA 20 within 2 candles and closed strong.",
    ),
    "Breakout near 52-week high on volume": _t(
        ("breakout", dict(days=20)),
        ("near_high", dict(days=252, lo=0, hi=5)),
        ("vol_spike", dict(lo=1.5)),
        ("candle", dict(pat="strong close")),
        ("rs", dict(days=63, lo=0)),
        desc="Breakout above 20-candle high within 5% of 52-week high, on 1.5× volume, outperforming NIFTY.",
    ),
    "Tight base after a run (flag)": _t(
        ("runup", dict(days=40, lo=25)),
        ("tight", dict(n=7, max=8)),
        ("lowvol", dict(nmin=3, nmax=0, x=1.0)),
        ("ma_pos", dict(kind="EMA", len=20, dir="above")),
        ("near_high", dict(days=60, lo=0, hi=12)),
        desc="A 25%+ run in 40 candles, then 7 candles inside an 8% range on light volume near highs.",
    ),
    "Volatility contraction (VCP)": _t(
        ("contract", dict(legs=3, n=7)),
        ("vol_fall", dict(n=3, skip=1)),
        ("near_high", dict(days=252, lo=0, hi=20)),
        ("ma_stack", dict(k1="SMA", n1=50, k2="SMA", n2=200)),
        ("ma_pos", dict(kind="SMA", len=50, dir="above")),
        desc="Three consecutive contracting consolidation legs on decreasing volume in an established trend.",
    ),
    "NR7 + inside bar (coiled)": _t(
        ("nr", dict(n=7)),
        ("inside", {}),
        ("ma_pos", dict(kind="EMA", len=20, dir="above")),
        ("adx", dict(lo=20)),
        desc="Narrowest candle of 7 that is also an inside bar, above EMA 20 with ADX trend confirmation.",
    ),
    "Breakout 3–8 candles ago, now retesting": _t(
        ("breakout", dict(days=30, amin=3, amax=8)),
        ("pullback", dict(nmin=2, nmax=5, vol=0.9, max_up=1)),
        ("ma_pos", dict(kind="EMA", len=20, dir="above")),
        ("rsi", dict(lo=50, hi=75)),
        desc="Broke 30-day high 3–8 candles ago and is currently pulling back lightly towards EMA 20.",
    ),
    "Weekly uptrend, daily oversold bounce": _t(
        ("ma_pos", dict(kind="EMA", len=20, dir="above", tf="weekly")),
        ("rsi", dict(lo=50, hi=100, tf="weekly")),
        ("rsi", dict(lo=30, hi=45, ago=1)),
        ("candle", dict(pat="green candle")),
        ("candle", dict(pat="strong close")),
        desc="Above 20-week EMA with weekly RSI > 50; daily RSI was oversold yesterday and bounced with strong close.",
    ),
}


class Ctx:
    """Stock candle container with lazy memoized indicators and weekly/sub-slice views."""

    def __init__(self, sym: str, df: pd.DataFrame, live_frac: Optional[float] = None, nifty: Optional[pd.Series] = None):
        df = df[pd.to_numeric(df["Close"], errors="coerce") > 0].copy()
        for c in ("Open", "High", "Low", "Close", "Volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Volume"] = df["Volume"].fillna(0.0)
        df[["Open", "High", "Low"]] = df[["Open", "High", "Low"]].apply(lambda s: s.fillna(df["Close"]))
        self.sym, self.df, self.nifty = sym, df, nifty
        self.c, self.o, self.h, self.l = df["Close"], df["Open"], df["High"], df["Low"]
        v = df["Volume"].astype(float).copy()
        self.projected = False
        if live_frac is not None and len(v):
            v.iloc[-1] = v.iloc[-1] / max(0.15, min(1.0, live_frac))
            self.projected = True
        self.v = v
        self.memo, self.root, self.end = {}, self, len(df)

    def at(self, end: int):
        root = self.root
        end = max(0, min(int(end), root.end))
        if end == root.end:
            return root
        k = ("view", end)
        if k not in root.memo:
            x = Ctx.__new__(Ctx)
            x.sym, x.nifty, x.root, x.end, x.memo, x.projected = root.sym, root.nifty, root, end, {}, False
            x.df = root.df.iloc[:end]
            x.c, x.o, x.h, x.l, x.v = root.c.iloc[:end], root.o.iloc[:end], root.h.iloc[:end], root.l.iloc[:end], root.v.iloc[:end]
            root.memo[k] = x
        return root.memo[k]

    def _ind(self, key, fn):
        root = self.root
        if key not in root.memo:
            root.memo[key] = fn(root)
        full = root.memo[key]
        return full if self is root else full.iloc[: self.end]

    def ma(self, kind: str, n: int) -> pd.Series:
        return self._ind(("ma", kind, n), lambda r: moving_average(r.c, kind, n))

    def rsi(self, n: int = 14) -> pd.Series:
        return self._ind(("rsi", n), lambda r: rsi_wilder(r.c, n))

    def avg_vol(self) -> pd.Series:
        return self._ind(("avgv",), lambda r: avg_volume(r.v, 20))

    def atr(self, n: int = 14) -> pd.Series:
        return self._ind(("atr", n), lambda r: atr(r.h, r.l, r.c, n))

    def macd(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return self._ind(("macd",), lambda r: macd(r.c))

    def adx(self, n: int = 14) -> pd.Series:
        return self._ind(("adx", n), lambda r: adx(r.h, r.l, r.c, n))

    def weekly(self):
        if "wk" in self.memo:
            return self.memo["wk"]
        root = self.root
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        if "wkfull" not in root.memo:
            full = root.df.resample("W-FRI").agg(agg).dropna(subset=["Close"])
            root.memo["wkfull"] = full
        full = root.memo["wkfull"]
        x = Ctx.__new__(Ctx)
        x.sym, x.df, x.nifty, x.projected, x.memo, x.end = self.sym, full, self.nifty, False, {}, len(full)
        x.c, x.o, x.h, x.l, x.v, x.root = full["Close"], full["Open"], full["High"], full["Low"], full["Volume"].astype(float), x
        self.memo["wk"] = x
        return x

    def turnover_cr(self) -> float:
        """Average daily traded value over last 20 candles in ₹ crore."""
        tv = (self.c * self.v).iloc[-21:-1] if len(self.c) > 21 else (self.c * self.v)
        return float(tv.mean() / 1e7) if len(tv) else 0.0


def _last(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s) == 0:
        return None
    try:
        v = float(s.iloc[-1])
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _find_pullback(ctx: Ctx, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c, h, l, v = (x.to_numpy(dtype=float) for x in (ctx.c, ctx.h, ctx.l, ctx.v))
    N = len(c)
    nmin = int(r.get("nmin", 2))
    nmax = int(r.get("nmax", 4))
    max_up = int(r.get("max_up", 0))
    vol_max = float(r.get("vol", 0.8))
    each_under = bool(r.get("each", False))
    dmin = float(r.get("dmin", 0.0))
    dmax = float(r.get("dmax", 0.0))

    for n in range(nmax, nmin - 1, -1):
        piv = N - n - 1
        if piv < 21:
            continue
        seg = slice(N - n, N)
        if h[piv] < h[seg].max():
            continue
        if h[piv] < h[max(0, piv - max(5, nmax)) : piv].max():
            continue
        ups = int((np.diff(c[piv:N]) > 0).sum())
        if ups > max_up or c[-1] >= c[piv]:
            continue
        base = v[piv - 19 : piv + 1].mean()
        if base <= 0:
            continue
        sv = v[seg]
        ratio = float(sv.mean() / base)
        if ratio > vol_max or (each_under and (sv >= base).any()):
            continue
        depth = float((h[piv] - l[seg].min()) / h[piv] * 100)
        if depth < dmin or (dmax > 0 and depth > dmax):
            continue
        return {"n": n, "ratio": ratio, "depth": depth, "ups": ups}
    return None


def _eval_rule(ctx: Ctx, r: Dict[str, Any]) -> Tuple[bool, str, str, Dict[str, Any]]:
    t = r["type"]
    c = ctx.c
    cur = _last(c)
    if cur is None:
        return False, f"{t.upper()} ?", "No price data", {}

    try:
        if t == "pullback":
            p = _find_pullback(ctx, r)
            if p:
                return (
                    True,
                    f"PB {p['n']}c · {p['ratio']:.2f}×",
                    f"{p['n']}-candle pullback, {p['depth']:.1f}% depth, vol {p['ratio']:.2f}× avg",
                    p,
                )
            return False, "PB ✗", "No valid pullback on dry volume found", {}

        if t == "lowvol":
            av = ctx.avg_vol()
            x = float(r.get("x", 1.0))
            nmin = int(r.get("nmin", 3))
            nmax = int(r.get("nmax", 0))
            below = (ctx.v < x * av).to_numpy()[::-1]
            k = 0
            for b in below:
                if not b:
                    break
                k += 1
            ok = k >= nmin and (not nmax or k <= nmax)
            return ok, f"LOWVOL {k}c", f"{k} consecutive low-volume candles (needs {nmin}+)", {"lowvol_count": k}

        if t == "runup":
            days = int(r.get("days", 30))
            lo = float(r.get("lo", 20.0))
            hi = float(r.get("hi", 0.0))
            w = c.iloc[-days - 1 : -1]
            if len(w) < 5:
                return False, "RUN ?", "not enough candles", {}
            arr = w.to_numpy()
            i = int(np.argmin(arr))
            rise = float((arr[i:].max() / arr[i] - 1) * 100) if arr[i] > 0 else 0.0
            ok = rise >= lo and (not hi or rise <= hi)
            return ok, f"RUN {rise:+.0f}%", f"Gained {rise:.1f}% in {days} candles", {"runup_pct": rise}

        if t == "rsi":
            period = int(r.get("n", 14))
            lo = float(r.get("lo", 50.0))
            hi = float(r.get("hi", 100.0))
            val = _last(ctx.rsi(period))
            if val is None:
                return False, "RSI ?", "not enough history", {}
            ok = lo <= val <= hi
            return ok, f"RSI {val:.0f}", f"RSI({period}) is {val:.1f} (target {lo}–{hi})", {"rsi": val}

        if t == "rsi_avg":
            n = int(r.get("n", 14))
            rs = ctx.rsi(14)
            a = _last(rs.rolling(n).mean())
            val = _last(rs)
            if val is None or a is None:
                return False, "RSI/AVG ?", "not enough history", {}
            ok = (val > a) if r.get("dir", "above") == "above" else (val < a)
            symbol_dir = ">" if val > a else "<"
            return ok, f"RSI {symbol_dir} AVG", f"RSI {val:.1f} vs {n}-avg {a:.1f}", {}

        if t == "ma_pos":
            kind = r.get("kind", "EMA")
            n = int(r.get("len", 20))
            m = _last(ctx.ma(kind, n))
            if m is None or cur is None:
                return False, f"{kind}{n} ?", "not enough history", {}
            dist = (cur / m - 1) * 100
            dir_ = r.get("dir", "above")
            dmin = float(r.get("dmin", 0.0))
            dmax = float(r.get("dmax", 0.0))
            if dir_ == "near":
                lim = dmax or 2.0
                ok = dmin <= abs(dist) <= lim
            else:
                ok = (dist > 0) if dir_ == "above" else (dist < 0)
                ok = ok and abs(dist) >= dmin and (not dmax or abs(dist) <= dmax)
            return ok, f"{kind}{n} {dist:+.1f}%", f"Price {dist:+.1f}% vs {kind}{n}", {"ma_dist": dist}

        if t == "ma_stack":
            k1, n1 = r.get("k1", "EMA"), int(r.get("n1", 20))
            k2, n2 = r.get("k2", "SMA"), int(r.get("n2", 50))
            f = _last(ctx.ma(k1, n1))
            s = _last(ctx.ma(k2, n2))
            if f is None or s is None:
                return False, "STACK ?", "not enough history", {}
            ok = f > s
            return ok, f"{k1}{n1}>{k2}{n2}" if ok else f"{k1}{n1}<{k2}{n2}", f"{k1}{n1} ({f:.1f}) vs {k2}{n2} ({s:.1f})", {}

        if t == "ma_cross":
            k1, n1 = r.get("k1", "EMA"), int(r.get("n1", 20))
            k2, n2 = r.get("k2", "SMA"), int(r.get("n2", 50))
            within = int(r.get("within", 5))
            f, s = ctx.ma(k1, n1), ctx.ma(k2, n2)
            d = (f - s).dropna()
            if len(d) < within + 1:
                return False, "CROSS ?", "not enough history", {}
            w = d.iloc[-within - 1 :]
            crossed = bool(((w.shift(1) <= 0) & (w > 0)).any()) and d.iloc[-1] > 0
            return crossed, "X-OVER ✓" if crossed else "X-OVER ✗", f"Crossed within {within} candles", {}

        if t == "vol_spike":
            av = _last(ctx.avg_vol())
            if not av:
                return False, "VOL ?", "no volume history", {}
            lo = float(r.get("lo", 1.5))
            hi = float(r.get("hi", 0.0))
            x = float(ctx.v.iloc[-1]) / av
            ok = x >= lo and (not hi or x <= hi)
            return ok, f"VOL {x:.1f}×", f"Volume {x:.2f}× 20D average", {"vol_x": x}

        if t == "atr":
            period = int(r.get("n", 14))
            a = _last(ctx.atr(period))
            if a is None or not cur:
                return False, "ATR ?", "not enough history", {}
            pct = a / cur * 100
            lo = float(r.get("lo", 1.0))
            hi = float(r.get("hi", 0.0))
            ok = pct >= lo and (not hi or pct <= hi)
            return ok, f"ATR {pct:.1f}%", f"ATR is {pct:.2f}% of price", {"atr_pct": pct}

        if t == "day_move":
            prev = float(c.iloc[-2]) if len(c) > 1 else cur
            ch = (cur / prev - 1) * 100
            mode = r.get("mode", "size")
            lo = float(r.get("lo", 2.0))
            hi = float(r.get("hi", 0.0))
            val = abs(ch) if mode == "size" else (ch if mode == "up" else -ch)
            ok = val >= lo and (not hi or val <= hi)
            return ok, f"DAY {ch:+.1f}%", f"Today's move is {ch:+.2f}%", {"day_chg": ch}

        if t == "change":
            days = int(r.get("days", 5))
            if len(c) <= days:
                return False, "CHG ?", "not enough history", {}
            base = float(c.iloc[-1 - days])
            ch = (cur / base - 1) * 100
            lo, hi = float(r.get("lo", 0.0)), float(r.get("hi", 100.0))
            ok = lo <= ch <= hi
            return ok, f"{days}D {ch:+.0f}%", f"{days}-day change is {ch:+.1f}%", {"change_pct": ch}

        if t == "near_high":
            days = int(r.get("days", 252))
            hh = float(ctx.h.iloc[-days:].max())
            d = (1 - cur / hh) * 100 if hh else 0.0
            lo, hi = float(r.get("lo", 0.0)), float(r.get("hi", 15.0))
            ok = lo <= d <= hi
            return ok, f"{d:.1f}% < HIGH", f"{d:.1f}% below {days}-candle high ({hh:,.2f})", {"off_high": d}

        if t == "breakout":
            days = int(r.get("days", 20))
            amin, amax = int(r.get("amin", 0)), int(r.get("amax", 0))
            hv, cv = ctx.h.to_numpy(dtype=float), c.to_numpy(dtype=float)
            if len(hv) < days + amax + 1:
                return False, "BO ?", "not enough history", {}
            for k in range(amin, amax + 1):
                e = len(cv) - k
                lvl = float(hv[e - 1 - days : e - 1].max())
                if cv[e - 1] > lvl:
                    return True, f"BO ✓ {lvl:,.0f}" + ("" if k == 0 else f" −{k}c"), f"Closed above {days}c high", {"bo_ago": k}
            lvl = float(hv[-1 - days : -1].max())
            return False, f"BO ✗ {lvl:,.0f}", f"Below {days}c high ({lvl:,.2f})", {}

        if t == "tight":
            w = int(r.get("n", 5))
            rng = (float(ctx.h.iloc[-w:].max()) - float(ctx.l.iloc[-w:].min())) / cur * 100
            max_rng = float(r.get("max", 6.0))
            ok = rng <= max_rng
            return ok, f"RANGE {rng:.1f}%", f"Last {w}c span {rng:.1f}% (max {max_rng}%)", {"tight_rng": rng}

        if t == "nr":
            n = int(r.get("n", 7))
            rg = (ctx.h - ctx.l).iloc[-n:]
            ok = float(rg.iloc[-1]) <= float(rg.min()) + 1e-9
            return ok, f"NR{n} {'✓' if ok else '✗'}", f"NR{n} pattern {'met' if ok else 'not met'}", {}

        if t == "inside":
            ok = float(ctx.h.iloc[-1]) <= float(ctx.h.iloc[-2]) and float(ctx.l.iloc[-1]) >= float(ctx.l.iloc[-2])
            return ok, f"INSIDE {'✓' if ok else '✗'}", "Inside candle" if ok else "Not an inside bar", {}

        if t == "gap":
            prev = float(c.iloc[-2]) if len(c) > 1 else cur
            g = (float(ctx.o.iloc[-1]) / prev - 1) * 100
            lo, hi = float(r.get("lo", 1.0)), float(r.get("hi", 20.0))
            ok = lo <= g <= hi
            return ok, f"GAP {g:+.1f}%", f"Opened gap {g:+.1f}%", {"gap_pct": g}

        if t == "rs":
            nifty = ctx.nifty
            days = int(r.get("days", 63))
            if nifty is None or len(c) <= days:
                return False, "RS ?", "NIFTY data unavailable", {}
            d0, d1 = c.index[-1 - days], c.index[-1]
            n0, n1 = nifty[nifty.index <= d0], nifty[nifty.index <= d1]
            if not len(n0) or not len(n1):
                return False, "RS ?", "NIFTY data unavailable", {}
            sr = (cur / float(c.loc[d0]) - 1) * 100
            nr = (float(n1.iloc[-1]) / float(n0.iloc[-1]) - 1) * 100
            ex = sr - nr
            lo = float(r.get("lo", 0.0))
            ok = ex >= lo
            return ok, f"RS {ex:+.0f}", f"Stock {sr:+.1f}% vs NIFTY {nr:+.1f}% ({ex:+.1f}% excess)", {"rs_excess": ex}

        if t == "macd":
            m, sg, hist = ctx.macd()
            mode = r.get("mode", "above signal")
            if mode == "above signal":
                ok = float(m.iloc[-1]) > float(sg.iloc[-1])
            elif mode == "above zero":
                ok = float(m.iloc[-1]) > 0
            else:
                ok = float(hist.iloc[-1]) > float(hist.iloc[-2])
            return ok, f"MACD {'✓' if ok else '✗'}", f"MACD {mode}", {}

        if t == "adx":
            period = int(r.get("n", 14))
            a = _last(ctx.adx(period))
            if a is None:
                return False, "ADX ?", "not enough history", {}
            lo = float(r.get("lo", 20.0))
            hi = float(r.get("hi", 0.0))
            ok = a >= lo and (not hi or a <= hi)
            return ok, f"ADX {a:.0f}", f"ADX({period}) is {a:.1f}", {"adx": a}

        if t == "candle":
            o, h, l = float(ctx.o.iloc[-1]), float(ctx.h.iloc[-1]), float(ctx.l.iloc[-1])
            po = float(ctx.o.iloc[-2]) if len(c) > 1 else None
            pc = float(ctx.c.iloc[-2]) if len(c) > 1 else None
            pat = r.get("pat", "hammer")
            ok = detect_candle(o, h, l, cur, po, pc, pat)
            return ok, f"{pat.upper()} {'✓' if ok else '✗'}", f"Candle shape: {pat}", {}

        if t == "ma_touch":
            kind = r.get("kind", "EMA")
            n = int(r.get("len", 20))
            within = int(r.get("within", 1))
            tol = float(r.get("tol", 0.5))
            ma_s = ctx.ma(kind, n)
            m = _last(ma_s)
            if m is None or len(ma_s.dropna()) < within:
                return False, "TOUCH ?", "not enough history", {}
            gap = (ctx.l.iloc[-within:] / ma_s.iloc[-within:] - 1) * 100
            best = float(gap.min())
            ok = best <= tol and cur > m
            return ok, f"TOUCH {kind}{n} {'✓' if ok else '✗'}", f"Tested {kind}{n} within {best:+.2f}%", {"touch_dist": best}

        if t == "contract":
            legs, n = int(r.get("legs", 3)), int(r.get("n", 5))
            if len(c) < legs * n + 1:
                return False, "VCP ?", "not enough history", {}
            hv = ctx.h.to_numpy(dtype=float)[-legs * n :]
            lv = ctx.l.to_numpy(dtype=float)[-legs * n :]
            rg = [(hv[i * n : (i + 1) * n].max() - lv[i * n : (i + 1) * n].min()) / cur * 100 for i in range(legs)]
            ok = all(rg[i] < rg[i - 1] for i in range(1, legs))
            return ok, f"VCP {'✓' if ok else '✗'}", "Shrinking ranges: " + " → ".join(f"{x:.1f}%" for x in rg), {}

        if t == "vol_fall":
            n = int(r.get("n", 3))
            skip = int(r.get("skip", 0))
            vv = ctx.v.to_numpy(dtype=float)[-n - 1 :]
            if len(vv) < n + 1:
                return False, "VOL↓ ?", "not enough history", {}
            rises = int((np.diff(vv) >= 0).sum())
            ok = rises <= skip
            return ok, f"VOL↓ {n - rises}/{n}", f"Volume fell on {n - rises} of {n} candles", {}

    except Exception:
        return False, f"{t.upper()} ?", "calculation error", {}

    return False, t, "unknown rule", {}


def evaluate_rule(ctx: Ctx, r: Dict[str, Any]) -> Tuple[bool, str, str, Dict[str, Any]]:
    target_ctx = ctx
    tag = ""
    if r.get("tf") == "weekly":
        target_ctx = target_ctx.weekly()
        tag += " W"
        if len(target_ctx.c) < 12:
            return False, f"{r['type'].upper()} W ?", "not enough weekly candles", {}

    ago = int(r.get("ago") or 0)
    if ago > 0:
        if target_ctx.end - ago < 12:
            return False, f"{r['type'].upper()} ?", f"not enough candles {ago} periods ago", {}
        target_ctx = target_ctx.at(target_ctx.end - ago)
        tag += f" −{ago}c"

    ok, chip, detail, m = _eval_rule(target_ctx, r)
    if tag:
        chip += tag
        detail += f" ({tag.strip()})"
    return ok, chip, detail, m


def verdict(checks: List[Dict[str, Any]], min_pass: int = 0) -> Dict[str, Any]:
    units, groups = [], {}
    for ck in checks:
        g = ck.get("grp", "—")
        if g != "—":
            if g not in groups:
                groups[g] = {"ok": False, "must": False, "checks": []}
                units.append(groups[g])
            u = groups[g]
        else:
            u = {"ok": False, "must": False, "checks": []}
            units.append(u)
        u["checks"].append(ck)
        u["ok"] = u["ok"] or ck["ok"]
        u["must"] = u["must"] or ck.get("must", False)

    total = len(units)
    need = total if not min_pass else max(1, min(int(min_pass), total))
    passed = sum(1 for u in units if u["ok"])
    failed = [u for u in units if not u["ok"]]
    req_fail = [u for u in failed if u["must"]]
    short = max(0, need - passed)

    status = "match" if not short and not req_fail else "near" if max(short, len(req_fail)) == 1 and total >= 2 else "no"
    order = req_fail + [u for u in failed if not u["must"]]
    fails = [ck for u in order for ck in u["checks"]]
    miss = ""
    if status != "match" and need < total:
        miss = f"passes {passed} of {total}, needs {need}" + (" · required rule failed" if req_fail else "")
    return {"passed": passed, "total": total, "need": need, "status": status, "fails": fails, "miss": miss}


def evaluate_stock(
    sym: str,
    df: pd.DataFrame,
    rules: List[Dict[str, Any]],
    live_frac: Optional[float] = None,
    nifty: Optional[pd.Series] = None,
    min_pass: int = 0,
) -> Optional[Dict[str, Any]]:
    if df is None or len(df) < 30:
        return None
    ctx = Ctx(sym, df, live_frac, nifty)
    if len(ctx.c) < 30:
        return None

    cur = float(ctx.c.iloc[-1])
    prev = float(ctx.c.iloc[-2]) if len(ctx.c) > 1 else cur
    checks, metrics = [], {}
    for r in rules:
        if not r.get("on", True):
            continue
        ok, chip, detail, m = evaluate_rule(ctx, r)
        checks.append({
            "ok": bool(ok),
            "chip": chip,
            "detail": detail,
            "label": RULES.get(r["type"], {}).get("label", r["type"]),
            "desc": describe(r),
            "grp": r.get("grp", "—"),
            "must": bool(r.get("must", False)),
            "type": r["type"],
        })
        metrics.update(m)

    av = _last(ctx.avg_vol())
    v = verdict(checks, min_pass)
    vol_x = float(ctx.v.iloc[-1]) / av if av and av > 0 else None

    return {
        "symbol": sym,
        "close": round(cur, 2),
        "chg": round((cur / prev - 1) * 100, 2) if prev else 0.0,
        "turnover": round(ctx.turnover_cr(), 2),
        "vol_x": round(vol_x, 2) if vol_x is not None else None,
        "rsi": round(float(_last(ctx.rsi(14))), 1) if _last(ctx.rsi(14)) is not None else None,
        "checks": checks,
        "fails": v["fails"],
        "status": v["status"],
        "passed": v["passed"],
        "total": v["total"],
        "miss": v["miss"],
        "metrics": metrics,
        "date": str(ctx.c.index[-1].date()),
        "projected": ctx.projected,
    }


def run_scan(
    universe: List[str],
    rules: List[Dict[str, Any]],
    pmin: float = 20.0,
    pmax: float = 50000.0,
    min_turnover: float = 0.0,
    want_near: bool = True,
    min_pass: int = 0,
) -> Dict[str, Any]:
    t0 = time.time()
    on_rules = [r for r in rules if r.get("on", True)]
    need_nifty = any(r["type"] == "rs" for r in on_rules)
    need_long = any(
        r.get("tf") == "weekly"
        or (r["type"] in ("ma_pos", "ma_stack", "ma_cross", "ma_touch") and max(r.get("len", 0), r.get("n1", 0), r.get("n2", 0)) > 150)
        for r in on_rules
    )

    # Stage 1: Fast 5-day snapshots for pre-filtering
    snaps = get_snapshots(universe)
    band: List[str] = []
    for sym, sn in snaps.items():
        if sn.empty or "Close" not in sn.columns:
            continue
        c = pd.to_numeric(sn["Close"], errors="coerce").dropna()
        if not len(c):
            continue
        last_px = float(c.iloc[-1])
        if not (pmin <= last_px <= pmax):
            continue
        tv = (sn["Close"] * sn["Volume"]).iloc[:-1] if len(sn) > 1 else (sn["Close"] * sn["Volume"])
        if min_turnover > 0 and (float(tv.mean()) / 1e7) < (min_turnover * 0.4):
            continue
        band.append(sym)

    # Stage 2: Full historical candles for survivors
    hist = get_histories(band, snaps, long=need_long)
    nifty = get_nifty_close() if need_nifty else None

    matches, near_matches, liquid_count = [], [], 0
    funnel = [{"desc": describe(r), "pass": 0, "only": 0} for r in on_rules]

    for sym in band:
        df = hist.get(sym)
        if df is None or len(df) < 30:
            continue
        live_frac = live_volume_fraction(df.index[-1])
        res = evaluate_stock(sym, df, on_rules, live_frac=live_frac, nifty=nifty, min_pass=min_pass)
        if res is None or res["turnover"] < min_turnover:
            continue

        liquid_count += 1
        bad_indices = [k for k, ck in enumerate(res["checks"]) if not ck["ok"]]
        for k, ck in enumerate(res["checks"]):
            funnel[k]["pass"] += 1 if ck["ok"] else 0
        if len(bad_indices) == 1:
            funnel[bad_indices[0]]["only"] += 1

        if res["status"] == "match":
            matches.append(res)
        elif res["status"] == "near" and want_near:
            near_matches.append(res)

    elapsed = round(time.time() - t0, 2)
    stats = {
        "universe": len(universe),
        "priced": len(snaps),
        "in_band": len(band),
        "liquid": liquid_count,
        "match": len(matches),
        "near": len(near_matches),
        "secs": elapsed,
    }

    return {
        "match": matches,
        "near": near_matches,
        "stats": stats,
        "funnel": funnel,
        "scanned_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }
