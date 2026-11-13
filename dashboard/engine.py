# -*- coding: utf-8 -*-
"""
Quant engine for the US watchlist dashboard.

Reuses the repo's own data layer (DataFetcherManager) and technical scorer
(StockTrendAnalyzer), then adds:
  * a rule-based "playbook" that turns indicators into a concrete plan
    (setup, entry zone / trigger, stop, target, reward:risk)
  * a walk-forward trade simulator used for the Backtest tab and for
    back-filling the History tab.

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
MAX_HOLD_BARS = 10      # time stop for simulated trades
TRIGGER_WINDOW = 5      # bars a trigger-type plan stays valid
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


# ---------------------------------------------------------------- trade simulation
def simulate_trade(d: pd.DataFrame, i: int, plan: Plan) -> Optional[Dict[str, Any]]:
    """
    Simulate acting on the plan generated at the close of bar i.
    Entry: next open (zone plans) or first cross of the trigger within TRIGGER_WINDOW bars.
    Exit: stop (stop first if both hit the same bar), target, or time stop at MAX_HOLD_BARS.
    Gaps through the stop exit at the open. No costs/slippage. Returns None if no trade.
    """
    n = len(d)
    if plan.stop is None or plan.target is None or i + 1 >= n:
        return None
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    atr = _f(d["atr14"].iloc[i], 0.0)

    entry = None
    start = None
    if plan.trigger is not None:
        for j in range(i + 1, min(i + 1 + TRIGGER_WINDOW, n)):
            if h[j] >= plan.trigger:
                entry, start = max(o[j], plan.trigger), j
                break
    else:
        j = i + 1
        if o[j] <= plan.entry_high + 0.5 * atr and o[j] > plan.stop:   # skip big gap-ups: don't chase
            entry, start = o[j], j
    if entry is None or entry <= plan.stop:
        return None

    risk = entry - plan.stop
    last = min(start + MAX_HOLD_BARS - 1, n - 1)
    exit_px, exit_idx, reason = c[last], last, "time"
    for k in range(start, last + 1):
        if k > start and o[k] <= plan.stop:
            exit_px, exit_idx, reason = o[k], k, "stop"
            break
        if l[k] <= plan.stop:
            exit_px, exit_idx, reason = plan.stop, k, "stop"
            break
        if h[k] >= plan.target:
            exit_px, exit_idx, reason = plan.target, k, "target"
            break
    if exit_idx == last and reason == "time" and last == n - 1 and (last - start + 1) < MAX_HOLD_BARS:
        return None   # still open at the end of data: not a closed trade
    return {
        "entry_date": d["date"].iloc[start].strftime("%Y-%m-%d"),
        "exit_date": d["date"].iloc[exit_idx].strftime("%Y-%m-%d"),
        "entry": float(entry), "exit": float(exit_px), "stop": float(plan.stop),
        "target": float(plan.target), "reason": reason,
        "ret_pct": float((exit_px / entry - 1) * 100),
        "r": float((exit_px - entry) / risk),
        "bars": int(exit_idx - start + 1),
        "exit_idx": int(exit_idx),
    }


def walk_forward(code: str, df: pd.DataFrame, analyzer: StockTrendAnalyzer, history_days: int = 30) -> Dict[str, Any]:
    """
    Run the playbook on every historical bar. Returns:
      trades:   closed, non-overlapping simulated trades for BUY/WATCH plans
      daily:    plan summaries for the last `history_days` bars (History back-fill)
      baseline: stats for entering every day at the next open and holding MAX_HOLD_BARS
    """
    d = prepare(df)
    n = len(d)
    trades: List[Dict[str, Any]] = []
    daily: List[Dict[str, Any]] = []
    busy_until = -1
    for i in range(MIN_BARS, n):
        sl = d.iloc[: i + 1]
        try:
            t = analyzer.analyze(sl[["date", "open", "high", "low", "close", "volume"]].copy(), code)
        except Exception:     # a bad slice must not kill the whole run
            continue
        plan = build_plan(sl, t)
        if i >= n - history_days:
            daily.append({
                "date": d["date"].iloc[i].strftime("%Y-%m-%d"),
                "price": _r(d["close"].iloc[i]),
                "chg_pct": _r((d["close"].iloc[i] / d["close"].iloc[i - 1] - 1) * 100, 2),
                **plan.to_dict(),
            })
        if plan.action in ("BUY", "WATCH") and plan.setup != "neutral" and i > busy_until:
            tr = simulate_trade(d, i, plan)
            if tr:
                tr.update({"code": code, "setup": plan.setup, "label": plan.label, "signal_date": d["date"].iloc[i].strftime("%Y-%m-%d")})
                trades.append(tr)
                busy_until = tr["exit_idx"]

    # Baseline: arbitrary entries (next open) with the SAME exit rules (2 ATR stop, 3 ATR target,
    # 10-bar time stop), non-overlapping. Answers: does picking these setups beat random timing?
    base_trades: List[Dict[str, Any]] = []
    b_busy = -1
    for i in range(MIN_BARS, n - 1):
        if i <= b_busy:
            continue
        px, a = _f(d["close"].iloc[i]), _f(d["atr14"].iloc[i], 0.0)
        if not (a > 0):
            continue
        gp = Plan("random", "Random entry", "BUY", 0, entry_low=px, entry_high=px, stop=px - 2 * a, target=px + 3 * a)
        tr = simulate_trade(d, i, gp)
        if tr:
            base_trades.append(tr)
            b_busy = tr["exit_idx"]
    return {"trades": trades, "daily": daily, "baseline_trades": base_trades,
            "first_date": d["date"].iloc[MIN_BARS].strftime("%Y-%m-%d") if n > MIN_BARS else None,
            "last_date": d["date"].iloc[-1].strftime("%Y-%m-%d")}


def summarise_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n": 0}
    rs = np.array([t["r"] for t in trades])
    rets = np.array([t["ret_pct"] for t in trades])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    return {
        "n": int(len(trades)),
        "win_rate": float((rs > 0).mean() * 100),
        "avg_ret": float(rets.mean()),
        "avg_r": float(rs.mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else None,
        "avg_loss_r": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "avg_bars": float(np.mean([t["bars"] for t in trades])),
        "target_hits": int(sum(t["reason"] == "target" for t in trades)),
        "stop_hits": int(sum(t["reason"] == "stop" for t in trades)),
        "time_exits": int(sum(t["reason"] == "time" for t in trades)),
    }


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
    """Adjusted conviction for today's card (regime + earnings), kept separate from the backtested raw plan."""
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
