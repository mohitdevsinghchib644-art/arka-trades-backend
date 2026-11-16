import os
import time
import pickle
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_DIR = Path(".cache")
STORE_FILE = CACHE_DIR / "scan_store.pkl"
BATCH_SIZE = 120

LIQUID = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
    "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ADANIENT", "ADANIPORTS", "POWERGRID", "NTPC", "M&M", "TMPV", "TATASTEEL", "JSWSTEEL", "BAJAJFINSV",
    "ONGC", "COALINDIA", "INDUSINDBK", "GRASIM", "HDFCLIFE", "SBILIFE", "DRREDDY", "CIPLA", "EICHERMOT", "BRITANNIA",
    "DIVISLAB", "APOLLOHOSP", "HEROMOTOCO", "BPCL", "TECHM", "UPL", "HINDALCO", "TATACONSUM", "BAJAJ-AUTO", "SHREECEM",
    "VEDL", "GODREJCP", "DABUR", "PIDILITIND", "SIEMENS", "AMBUJACEM", "BANKBARODA", "CANBK", "PNB", "IDFCFIRSTB",
    "FEDERALBNK", "AUBANK", "BANDHANBNK", "CHOLAFIN", "LICHSGFIN", "MUTHOOTFIN", "PFC", "RECLTD", "IRFC", "IEX",
    "ETERNAL", "NYKAA", "PAYTM", "POLICYBZR", "DMART", "TRENT", "JUBLFOOD", "PERSISTENT", "COFORGE", "LTIM", "MPHASIS",
    "OFSS", "LTTS", "DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "PGHH", "COLPAL", "MARICO", "EMAMILTD", "VBL",
    "TVSMOTOR", "ASHOKLEY", "BALKRISIND", "MOTHERSON", "BOSCHLTD", "BSE", "INDIGO", "MAXHEALTH", "JIOFIN", "BEL", "HAL",
    "SHRIRAMFIN", "DIXON", "POLYCAB", "CDSL", "ANGELONE", "MCX", "KAYNES", "SUZLON", "IREDA", "RVNL", "TATAPOWER"
]

_STORE_INSTANCE: Optional[Dict] = None
_STORE_LOCK = threading.Lock()


def _get_store() -> Dict:
    global _STORE_INSTANCE
    with _STORE_LOCK:
        if _STORE_INSTANCE is not None:
            return _STORE_INSTANCE
        s = {
            "hist": {},
            "snap": {},
            "snap_key": {},
            "lock": threading.Lock(),
            "nifty": None,
            "nifty_at": 0.0,
            "long_ok": set(),
        }
        try:
            if STORE_FILE.exists():
                with open(STORE_FILE, "rb") as fh:
                    d = pickle.load(fh)
                if isinstance(d, dict) and isinstance(d.get("hist"), dict):
                    s["hist"] = d["hist"]
                    s["long_ok"] = set(d.get("long_ok") or [])
        except Exception:
            pass
        _STORE_INSTANCE = s
        return _STORE_INSTANCE


def _save_store(s: Dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STORE_FILE, "wb") as fh:
            pickle.dump(
                {"hist": s["hist"], "long_ok": sorted(s.get("long_ok") or [])},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass


def is_market_open(now: Optional[datetime] = None) -> bool:
    """True if Indian markets are currently within trading hours (Mon-Fri 09:15 - 15:30 IST)."""
    now = now or datetime.now(IST)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 30


def live_volume_fraction(last_date) -> Optional[float]:
    """Fraction of trading day elapsed for projecting volume during live trading."""
    now = datetime.now(IST)
    if not is_market_open(now):
        return None
    try:
        if pd.Timestamp(last_date).date() != now.date():
            return None
    except Exception:
        return None
    elapsed_minutes = (now.hour * 60 + now.minute) - (9 * 60 + 15)
    return max(0.15, min(1.0, elapsed_minutes / 375.0))


def data_slot_key() -> str:
    """10-minute snapshot slots during live trading hours, once per session afterwards."""
    now = datetime.now(IST)
    if is_market_open(now):
        return f"L{now:%Y%m%d%H}{now.minute // 10}"
    t = now.hour * 60 + now.minute
    return f"C{now:%Y%m%d}{'post' if t > 15 * 60 + 30 else 'pre'}"


def _norm_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Close"] > 0]
    return df


def _parse_download(df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = list(df.columns.get_level_values(0).unique())
        lv1 = list(df.columns.get_level_values(1).unique())
        for t in tickers:
            try:
                if t in lv0:
                    sub = df[t]
                elif t in lv1:
                    sub = df.xs(t, axis=1, level=1)
                else:
                    continue
                sub = sub.dropna(how="all")
                if not sub.empty and "Close" in sub.columns:
                    out[t] = sub
            except Exception:
                continue
    elif len(tickers) == 1 and "Close" in df.columns:
        out[tickers[0]] = df.dropna(how="all")
    return out


def _download_symbols(symbols: List[str], period: str = "5d") -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return out
    batches = [symbols[i : i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

    def get_batch(b: List[str]) -> Dict[str, pd.DataFrame]:
        tickers = [f"{s}.NS" for s in b]
        raw = yf.download(
            tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
        parsed = _parse_download(raw, tickers)
        return {s: _norm_frame(parsed[f"{s}.NS"]) for s in b if f"{s}.NS" in parsed}

    for b in batches:
        try:
            got = get_batch(b)
        except Exception:
            got = {}
        if len(got) < len(b) * 0.5 and len(b) > 10:
            miss = [s for s in b if s not in got]
            half_idx = len(miss) // 2
            for half in (miss[:half_idx], miss[half_idx:]):
                time.sleep(0.5)
                try:
                    got.update(get_batch(half))
                except Exception:
                    pass
        out.update({k: v for k, v in got.items() if not v.empty})
    return out


def get_snapshots(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetches latest 5-day snapshots per symbol, cached by data_slot_key."""
    s = _get_store()
    key = data_slot_key()
    need = [x for x in symbols if s["snap_key"].get(x) != key]
    if need:
        got = _download_symbols(need, period="5d")
        with s["lock"]:
            for sym in need:
                if sym in got:
                    s["snap"][sym] = got[sym]
                    s["snap_key"][sym] = key
    return {x: s["snap"].get(x) for x in symbols if s["snap"].get(x) is not None}


def get_histories(
    symbols: List[str], snaps: Dict[str, pd.DataFrame], long: bool = False
) -> Dict[str, pd.DataFrame]:
    """Fetches up to 1-2 years of historical candles, topped up with the latest snapshot."""
    s = _get_store()
    need = []
    for x in symbols:
        h = s["hist"].get(x)
        sn = snaps.get(x)
        if h is None or len(h) < 60 or (long and x not in s["long_ok"]):
            need.append(x)
            continue
        if sn is None or sn.empty:
            continue
        if h.index[-1] < sn.index[0]:
            need.append(x)
            continue
        common = h.index.intersection(sn.index[:-1])
        if len(common):
            ratio = (h.loc[common, "Close"] / sn.loc[common, "Close"]).dropna()
            if len(ratio) and (ratio - 1).abs().max() > 0.03:
                need.append(x)

    if need:
        got = _download_symbols(need, period="2y" if long else "1y")
        with s["lock"]:
            s["hist"].update(got)
            if long:
                s["long_ok"].update(need)

    out: Dict[str, pd.DataFrame] = {}
    with s["lock"]:
        for x in symbols:
            h = s["hist"].get(x)
            sn = snaps.get(x)
            if h is None or h.empty:
                continue
            if sn is not None and not sn.empty and h.index[-1] >= sn.index[0] - pd.Timedelta(days=10):
                h = pd.concat([h[h.index < sn.index[0]], sn])
                h = h[~h.index.duplicated(keep="last")].tail(520)
                s["hist"][x] = h
            out[x] = h

    if need:
        _save_store(s)
    return out


def get_nifty_close() -> Optional[pd.Series]:
    """Cached daily closing prices of NIFTY 50 (^NSEI) for relative strength."""
    s = _get_store()
    if s["nifty"] is not None and time.time() - s["nifty_at"] < 3600:
        return s["nifty"]
    try:
        df = yf.download(
            "^NSEI", period="2y", interval="1d", progress=False, auto_adjust=False, threads=False
        )
        c = df["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = pd.to_numeric(c, errors="coerce").dropna()
        idx = pd.to_datetime(c.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        c.index = idx.normalize()
        s["nifty"] = c
        s["nifty_at"] = time.time()
    except Exception:
        pass
    return s["nifty"]


def get_universe(
    kind: str = "liquid", custom_symbols: Optional[List[str]] = None
) -> Tuple[List[str], str]:
    """Resolves symbol universe based on requested mode."""
    if custom_symbols and len(custom_symbols) > 0:
        clean = [s.strip().upper() for s in custom_symbols if s.strip()]
        return clean, f"{len(clean)} custom symbols"

    k = (kind or "").lower()
    if "nifty 500" in k or "n500" in k:
        # Check if local CSV exists, else fallback to LIQUID
        n500_path = Path("data/ind_nifty500list.csv")
        if n500_path.exists():
            try:
                df = pd.read_csv(n500_path)
                col = next((c for c in ["Symbol", "symbol"] if c in df.columns), df.columns[0])
                syms = [s.strip().upper() for s in df[col].dropna().tolist() if s.strip()]
                return syms, f"{len(syms)} NIFTY 500 stocks"
            except Exception:
                pass
        return LIQUID, f"{len(LIQUID)} liquid stocks (fallback for NIFTY 500)"

    return LIQUID, f"{len(LIQUID)} highly liquid NSE stocks"
