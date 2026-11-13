# -*- coding: utf-8 -*-
"""
Quant engine for the US watchlist dashboard.

Reuses the repo's own data layer (DataFetcherManager) and technical scorer
(StockTrendAnalyzer), then adds:
  * a rule-based "playbook" that turns indicators into a concrete plan
    (setup, entry zone / trigger, stop, target, reward:risk)
  * a day-by-day replay of the playbook used to back-fill the History tab.

Everything here is rule-based technical analysis on daily bars. It is decision
support, not advice. No look-ahead: the plan for day i only sees bars[:i+1].
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.stock_analyzer import (
    MACDStatus,
    RSIStatus,
    StockTrendAnalyzer,
    TrendAnalysisResult,
    TrendStatus,
    VolumeStatus,
)

logger = logging.getLogger(__name__)

MIN_BARS = 100          # need MA60 + warm-up before the analyzer is meaningful
MIN_RR = 1.5            # below this a "buy" is downgraded to "wait"

# ---------------------------------------------------------------- label maps
TREND_LABEL = {
    TrendStatus.STRONG_BULL: "Strong uptrend",
    TrendStatus.BULL: "Uptrend",
    TrendStatus.WEAK_BULL: "Weak uptrend",
    TrendStatus.CONSOLIDATION: "Sideways",
    TrendStatus.WEAK_BEAR: "Weak downtrend",
    TrendStatus.BEAR: "Downtrend",
    TrendStatus.STRONG_BEAR: "Strong downtrend",
}
MACD_LABEL = {
    MACDStatus.GOLDEN_CROSS_ZERO: "MACD bullish cross above zero",
    MACDStatus.GOLDEN_CROSS: "MACD bullish cross",
    MACDStatus.BULLISH: "MACD bullish",
    MACDStatus.CROSSING_UP: "MACD crossing above zero",
    MACDStatus.CROSSING_DOWN: "MACD crossing below zero",
    MACDStatus.BEARISH: "MACD bearish",
    MACDStatus.DEATH_CROSS: "MACD bearish cross",
}
VOLUME_LABEL = {
    VolumeStatus.HEAVY_VOLUME_UP: "heavy-volume rally",
    VolumeStatus.HEAVY_VOLUME_DOWN: "heavy-volume selling",
    VolumeStatus.SHRINK_VOLUME_UP: "light-volume rally",
    VolumeStatus.SHRINK_VOLUME_DOWN: "light-volume pullback",
    VolumeStatus.NORMAL: "normal volume",
}

SETUPS = {
    # key: (label, action)
    "pullback": ("Pullback buy", "BUY"),
    "breakout": ("Breakout", "BUY"),
    "breakout_watch": ("Breakout watch", "WATCH"),
    "bounce": ("Oversold bounce (counter-trend)", "WATCH"),
    "extended": ("Extended: don't chase", "WAIT"),
    "neutral": ("No edge yet", "WAIT"),
    "avoid": ("Downtrend: avoid", "AVOID"),
}


# ---------------------------------------------------------------- helpers
def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _r(x: Any, nd: int = 2) -> Optional[float]:
    v = _f(x)
    return None if math.isnan(v) else round(v, nd)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a fetcher DataFrame: sorted, datetime index-free, numeric OHLCV + ATR."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    d["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return d


# ---------------------------------------------------------------- the playbook
@dataclass
class Plan:
    setup: str
    label: str
    action: str
    score: int
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    trigger: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    rr: Optional[float] = None
    reasons: Optional[List[str]] = None
    risks: Optional[List[str]] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup": self.setup, "label": self.label, "action": self.action,
            "score": self.score,
            "entry_low": _r(self.entry_low), "entry_high": _r(self.entry_high),
            "trigger": _r(self.trigger), "stop": _r(self.stop), "target": _r(self.target),
            "rr": _r(self.rr, 1), "reasons": self.reasons or [], "risks": self.risks or [],
            "note": self.note,
        }


def describe(t: TrendAnalysisResult, d: pd.DataFrame) -> Dict[str, List[str]]:
    """English reasons / risks built from numbers (the analyzer's own text is Chinese)."""
    reasons: List[str] = []
    risks: List[str] = []
    price = t.current_price

    reasons.append(f"{TREND_LABEL.get(t.trend_status, 'Sideways')} (MA5 {t.ma5:.2f} / MA10 {t.ma10:.2f} / MA20 {t.ma20:.2f})")
    if t.trend_status in (TrendStatus.BEAR, TrendStatus.STRONG_BEAR, TrendStatus.WEAK_BEAR):
        risks.append("Price structure is bearish: shorter averages are below longer ones")
    if t.ma60 and price > t.ma60:
        reasons.append(f"Above 60-day average ({t.ma60:.2f})")
    elif t.ma60:
        risks.append(f"Below 60-day average ({t.ma60:.2f})")

    bias = t.bias_ma5
    if bias > 5:
        risks.append(f"Stretched: {bias:.1f}% above MA5, so pullback risk is elevated")
    elif bias < -5:
        reasons.append(f"{abs(bias):.1f}% below MA5: deep pullback")
    elif -3 <= bias <= 2:
        reasons.append(f"Close to MA5 ({bias:+.1f}%): low-risk entry area")

    (reasons if t.macd_status in (MACDStatus.GOLDEN_CROSS_ZERO, MACDStatus.GOLDEN_CROSS, MACDStatus.BULLISH, MACDStatus.CROSSING_UP) else risks).append(MACD_LABEL.get(t.macd_status, "MACD neutral"))

    rsi = t.rsi_12
    if t.rsi_status == RSIStatus.OVERBOUGHT:
        risks.append(f"RSI {rsi:.0f}: overbought")
    elif t.rsi_status == RSIStatus.OVERSOLD:
        reasons.append(f"RSI {rsi:.0f}: oversold, bounce potential")
    else:
        reasons.append(f"RSI {rsi:.0f}: {'healthy momentum' if rsi >= 50 else 'soft momentum'}")

    vol = VOLUME_LABEL.get(t.volume_status, "normal volume")
    if t.volume_status == VolumeStatus.HEAVY_VOLUME_DOWN:
        risks.append(f"{vol.capitalize()} ({t.volume_ratio_5d:.1f}x 5-day average)")
    elif t.volume_status in (VolumeStatus.HEAVY_VOLUME_UP, VolumeStatus.SHRINK_VOLUME_DOWN):
        reasons.append(f"{vol.capitalize()} ({t.volume_ratio_5d:.1f}x 5-day average)")
    return {"reasons": reasons, "risks": risks}


def build_plan(d: pd.DataFrame, t: TrendAnalysisResult) -> Plan:
    """Turn the analyzer output + price structure into a concrete plan. `d` ends at 'today'."""
    price = _f(d["close"].iloc[-1])
    atr = _f(d["atr14"].iloc[-1], price * 0.02)
    hi20_prior = _f(d["high"].iloc[-21:-1].max())
    lo10 = _f(d["low"].iloc[-10:].min())
    hi60 = _f(d["high"].iloc[-60:].max())
    hi252 = _f(d["high"].iloc[-252:].max())

    bull = t.trend_status in (TrendStatus.STRONG_BULL, TrendStatus.BULL)
    weak_bull = t.trend_status == TrendStatus.WEAK_BULL
    bear = t.trend_status in (TrendStatus.BEAR, TrendStatus.STRONG_BEAR, TrendStatus.WEAK_BEAR)
    bias = _f(t.bias_ma5, 0.0)
    rsi12 = _f(t.rsi_12, 50.0)
    rsi6 = _f(t.rsi_6, 50.0)
    vr = _f(t.volume_ratio_5d, 1.0)
    ma5, ma10, ma20 = _f(t.ma5), _f(t.ma10), _f(t.ma20)
    score = int(t.signal_score or 0)

    txt = describe(t, d)
    setup = "neutral"
    entry_low = entry_high = trigger = None
    note = ""

    if not bear and price > hi20_prior and vr >= 1.2 and bias <= 8 and rsi12 < 80:
        setup = "breakout"
        entry_low, entry_high = max(hi20_prior, price - 0.5 * atr), price + 0.25 * atr
        note = f"Closed above the 20-day high ({hi20_prior:.2f}) on {vr:.1f}x volume. Buy only if it holds above that level."
    elif (bull or weak_bull) and (bias > 5 or rsi12 > 75):
        setup = "extended"
        entry_low, entry_high = ma10, max(ma5, ma10)
        note = f"Trend is strong but price is stretched. Wait for a dip toward MA10 ({ma10:.2f}) instead of chasing."
    elif (bull or (weak_bull and price >= ma20)) and -5 <= bias <= 3 and rsi12 < 70 and price >= ma20:
        setup = "pullback"
        entry_low, entry_high = price - 0.5 * atr, price + 0.25 * atr
        note = "Uptrend intact and price is near its short-term average: a low-risk place to buy the dip."
    elif not bear and price >= 0.97 * hi20_prior and price <= hi20_prior:
        setup = "breakout_watch"
        trigger = hi20_prior * 1.002
        entry_low, entry_high = trigger, trigger + 0.25 * atr
        note = f"Coiling just under the 20-day high. Buy only on a close above {trigger:.2f}, ideally on volume."
    elif bear and (rsi12 < 32 or rsi6 < 20):
        setup = "bounce"
        trigger = ma5
        entry_low, entry_high = ma5, ma5 + 0.25 * atr
        note = f"Oversold inside a downtrend. Counter-trend: only on a reclaim of MA5 ({ma5:.2f}), small size."
    elif bear:
        setup = "avoid"
        note = f"Downtrend. No long entry until price reclaims MA20 ({ma20:.2f}). If holding, review your stop."
    else:
        setup = "neutral"
        note = "No clean setup. Let the trend resolve before committing capital."

    label, action = SETUPS[setup]
    plan = Plan(setup, label, action, score, entry_low, entry_high, trigger,
                reasons=txt["reasons"], risks=txt["risks"], note=note)

    if entry_low is not None and setup != "avoid":
        ref = (entry_low + entry_high) / 2 if trigger is None else trigger
        stop_atr = ref - 2.0 * atr
        swing = lo10 - 0.25 * atr
        stop = max(stop_atr, min(swing, ref - 0.75 * atr))
        # Target: nearest overhead level at least 1 ATR above the entry
        levels = [x for x in [hi20_prior, hi60, hi252, *[float(v) for v in (t.resistance_levels or [])]]
                  if math.isfinite(x) and x > max(ref, price) + 1.0 * atr]
        target = min(levels) if levels else max(ref, price) + 3.0 * atr
        risk = ref - stop
        plan.stop, plan.target = stop, target
        plan.rr = (target - ref) / risk if risk > 0 else None
        if plan.action == "BUY" and (plan.rr is None or plan.rr < MIN_RR):
            plan.action = "WAIT"
            plan.label = plan.label + " (poor reward:risk)"
            plan.note = (f"Setup is valid but reward:risk is only {plan.rr:.1f}:1 (need {MIN_RR}+). "
                         "Wait for a cheaper entry or a higher target." if plan.rr else plan.note)
    return plan


# ---------------------------------------------------------------- history back-fill
def daily_plans(code: str, df: pd.DataFrame, analyzer: StockTrendAnalyzer, history_days: int = 25) -> List[Dict[str, Any]]:
    """Plan summaries for the last `history_days` bars (feeds the History tab). No look-ahead: bar i only sees bars[:i+1]."""
    d = prepare(df)
    n = len(d)
    out: List[Dict[str, Any]] = []
    for i in range(max(MIN_BARS, n - history_days), n):
        sl = d.iloc[: i + 1]
        try:
            t = analyzer.analyze(sl[["date", "open", "high", "low", "close", "volume"]].copy(), code)
        except Exception:     # a bad slice must not kill the whole run
            continue
        plan = build_plan(sl, t)
        out.append({
            "date": d["date"].iloc[i].strftime("%Y-%m-%d"),
            "price": _r(d["close"].iloc[i]),
            "chg_pct": _r((d["close"].iloc[i] / d["close"].iloc[i - 1] - 1) * 100, 2),
            **plan.to_dict(),
        })
    return out


# ---------------------------------------------------------------- snapshot pieces
def analyse_now(code: str, df: pd.DataFrame, analyzer: Optional[StockTrendAnalyzer] = None) -> Dict[str, Any]:
    """Current plan + indicator levels + sparkline for one ticker (today's card)."""
    analyzer = analyzer or StockTrendAnalyzer()
    d = prepare(df)
    t = analyzer.analyze(d[["date", "open", "high", "low", "close", "volume"]].copy(), code)
    plan = build_plan(d, t)
    price = _f(d["close"].iloc[-1])
    prev = _f(d["close"].iloc[-2], price)
    yr = d.iloc[-252:]
    return {
        "code": code,
        "asof": d["date"].iloc[-1].strftime("%Y-%m-%d"),
        "price": _r(price),
        "chg_pct": _r((price / prev - 1) * 100),
        "ret_5d": _r((price / _f(d["close"].iloc[-6]) - 1) * 100, 1) if len(d) > 6 else None,
        "ret_1m": _r((price / _f(d["close"].iloc[-22]) - 1) * 100, 1) if len(d) > 22 else None,
        "trend": TREND_LABEL.get(t.trend_status, "Sideways"),
        "trend_key": t.trend_status.name,
        **plan.to_dict(),
        "levels": {
            "ma5": _r(t.ma5), "ma10": _r(t.ma10), "ma20": _r(t.ma20), "ma60": _r(t.ma60),
            "atr": _r(d["atr14"].iloc[-1]), "rsi": _r(t.rsi_12, 0),
            "hi52": _r(yr["high"].max()), "lo52": _r(yr["low"].min()),
            "vol_ratio": _r(t.volume_ratio_5d, 1),
        },
        "spark": [_r(x) for x in d["close"].iloc[-60:].tolist()],
        "spark_dates": [x.strftime("%Y-%m-%d") for x in d["date"].iloc[-60:]],
    }


def market_regime(spy: Dict[str, Any], qqq: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple regime filter from SPY (and QQQ): bullish / neutral / bearish."""
    def grade(x):
        k = x["trend_key"]
        if k in ("STRONG_BULL", "BULL"):
            return 1
        if k in ("BEAR", "STRONG_BEAR", "WEAK_BEAR"):
            return -1
        return 0
    g = [grade(spy)] + ([grade(qqq)] if qqq else [])
    s = sum(g)
    regime = "bullish" if s > 0 else "bearish" if s < 0 else "neutral"
    notes = {
        "bullish": "Index trend supports long ideas. Normal sizing.",
        "neutral": "Index is mixed. Be selective and size down.",
        "bearish": "Index is in a downtrend. Long ideas are lower-probability; size down hard or wait.",
    }
    return {"regime": regime, "note": notes[regime]}


def conviction(score: int, plan: Dict[str, Any], regime: str, earnings_days: Optional[int]) -> Dict[str, Any]:
    """Adjusted conviction for today's card (regime + earnings), kept separate from the raw plan."""
    s = score
    flags: List[str] = []
    if plan["action"] in ("BUY", "WATCH"):
        if regime == "bearish":
            s -= 10
            flags.append("Market downtrend")
        elif regime == "neutral":
            s -= 4
        if plan.get("rr") and plan["rr"] >= 2.5:
            s += 5
    if earnings_days is not None and 0 <= earnings_days <= 7:
        s -= 15
        flags.append(f"Earnings in {earnings_days}d: gap risk")
    level = "High" if s >= 75 else "Medium" if s >= 60 else "Low"
    return {"conviction": level, "adj_score": int(s), "flags": flags}
