import numpy as np
import pandas as pd
from typing import Tuple, Optional


def _rma(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing (seeded with SMA), identical to TradingView's ta.rma."""
    out = np.full(len(values), np.nan)
    if len(values) <= n:
        return out
    out[n] = np.nanmean(values[1 : n + 1])
    for i in range(n + 1, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def _rma_series(x: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing for pandas Series via exponential weighted mean."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) using Wilder's smoothing."""
    c = close.astype(float).to_numpy()
    if len(c) <= n:
        return pd.Series(np.nan, index=close.index)
    d = np.diff(c, prepend=np.nan)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    gain[0] = np.nan
    loss[0] = np.nan
    ag, al = _rma(gain, n), _rma(loss, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + ag / al)
    rsi = np.where((al == 0) & ~np.isnan(ag), 100.0, rsi)
    return pd.Series(rsi, index=close.index)


def moving_average(close: pd.Series, kind: str = "EMA", n: int = 20) -> pd.Series:
    """Calculates Exponential (EMA) or Simple (SMA) moving average."""
    if kind.upper() == "EMA":
        return close.ewm(span=n, adjust=False, min_periods=n).mean()
    return close.rolling(n, min_periods=n).mean()


def avg_volume(volume: pd.Series, n: int = 20) -> pd.Series:
    """20-day average volume ending YESTERDAY (so today's volume is evaluated against it)."""
    return volume.rolling(n).mean().shift(1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range (ATR) with Wilder's RMA."""
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return _rma_series(tr, n)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence -> (macd, signal, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average Directional Index (ADX) measuring trend strength (0-100)."""
    up = high.diff()
    dn = -low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = atr(high, low, close, n)
    pdi = 100.0 * _rma_series(pdm, n) / tr.replace(0, np.nan)
    mdi = 100.0 * _rma_series(mdm, n) / tr.replace(0, np.nan)
    denom = (pdi + mdi).replace(0, np.nan)
    dx = (100.0 * (pdi - mdi).abs() / denom).replace([np.inf, -np.inf], np.nan)
    return _rma_series(dx.fillna(0), n)


def detect_candle(
    cur_open: float,
    cur_high: float,
    cur_low: float,
    cur_close: float,
    prev_open: Optional[float] = None,
    prev_close: Optional[float] = None,
    pattern: str = "hammer",
) -> bool:
    """Classifies the latest candlestick shape according to technical patterns."""
    rng = cur_high - cur_low
    if rng <= 0:
        return False
    body = abs(cur_close - cur_open)
    top_wick = cur_high - max(cur_close, cur_open)
    bot_wick = min(cur_close, cur_open) - cur_low
    pos = (cur_close - cur_low) / rng

    pat = pattern.lower().strip()
    if pat == "hammer":
        return bot_wick >= 2 * body and top_wick <= 0.25 * rng and body <= 0.4 * rng
    if pat == "doji":
        return body <= 0.1 * rng
    if pat == "strong close":
        return pos >= 0.75
    if pat == "weak close":
        return pos <= 0.25
    if pat == "green candle":
        return cur_close > cur_open
    if pat == "red candle":
        return cur_close < cur_open
    if pat == "bullish engulfing":
        if prev_open is None or prev_close is None:
            return False
        return (
            prev_close < prev_open
            and cur_close > cur_open
            and cur_close >= prev_open
            and cur_open <= prev_close
        )
    if pat == "bearish engulfing":
        if prev_open is None or prev_close is None:
            return False
        return (
            prev_close > prev_open
            and cur_close < cur_open
            and cur_close <= prev_open
            and cur_open >= prev_close
        )
    return False
