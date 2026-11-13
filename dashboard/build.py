# -*- coding: utf-8 -*-
"""
Build the US watchlist dashboard (a single self-contained, mobile-friendly HTML file).

    python dashboard/build.py                 # watchlist from STOCK_LIST in .env
    python dashboard/build.py --stocks AAPL,NVDA
    python dashboard/build.py --ai on|off|auto   # LLM layer via the repo's own pipeline

Output (dashboard/site/):  index.html (open it, host it, or send it)
                           data/latest.json, data/history.json, digest.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)          # the repo's config uses relative paths (./data/...)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
logging.disable(logging.WARNING)  # the data layer is chatty; we print our own progress

from data_provider.base import DataFetcherManager  # noqa: E402
from src.stock_analyzer import StockTrendAnalyzer  # noqa: E402

from dashboard import engine as E  # noqa: E402

US_TICKER = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")
MAX_HISTORY_DAYS = 250
WATCHLIST_FILE = HERE / "watchlist.json"   # written by serve.py (Add/Remove ticker); overrides STOCK_LIST
BACKFILL_DAYS = 25
CHART_BARS = 960          # daily bars embedded per ticker for the Charts tab: 3Y (756) + MA200 warm-up
LLM_KEY_VARS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                "AIHUBMIX_KEY", "LITELLM_API_KEY", "MODELFLARE_KEY", "GEMINI_API_KEYS", "OPENAI_API_KEYS")


def log(msg: str) -> None:
    print(f"[dashboard] {msg}", flush=True)


# ------------------------------------------------------------------ inputs
def watchlist(arg: Optional[str]) -> List[str]:
    raw = arg
    if not raw and WATCHLIST_FILE.exists():
        try:
            raw = " ".join(json.loads(WATCHLIST_FILE.read_text(encoding="utf-8")))
        except Exception:
            log(f"could not read {WATCHLIST_FILE.name}; falling back to STOCK_LIST")
    raw = raw or os.getenv("STOCK_LIST", "")
    out, seen = [], set()
    for tok in re.split(r"[,\s;]+", raw.upper()):
        if not tok or tok in seen:
            continue
        if US_TICKER.match(tok):
            out.append(tok)
            seen.add(tok)
        else:
            log(f"skipping non-US / invalid symbol: {tok}")
    return out


def _num(x: Any, nd: int = 2) -> Optional[float]:
    try:
        v = float(x)
        return round(v, nd) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _analyst_positioning(info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyst consensus/targets and short-interest/ownership from Yahoo's quote summary (None when Yahoo has no data, e.g. ETFs)."""
    out: Dict[str, Any] = {"analyst": None, "positioning": None}
    rec = info.get("recommendationKey")
    a = {"rec": rec if rec and rec != "none" else None, "rec_mean": _num(info.get("recommendationMean")),
         "n": info.get("numberOfAnalystOpinions"), "target_mean": _num(info.get("targetMeanPrice")),
         "target_high": _num(info.get("targetHighPrice")), "target_low": _num(info.get("targetLowPrice"))}
    if a["rec"] or a["target_mean"]:
        out["analyst"] = a
    spf, ss, ssp = info.get("shortPercentOfFloat"), _num(info.get("sharesShort"), 0), _num(info.get("sharesShortPriorMonth"), 0)
    sdate = None
    try:
        if info.get("dateShortInterest"):
            sdate = datetime.fromtimestamp(int(info["dateShortInterest"]), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        pass
    pos = {"short_pct_float": _num(spf * 100 if spf is not None else None, 2), "days_to_cover": _num(info.get("shortRatio")),
           "short_chg_pct": _num((ss / ssp - 1) * 100, 1) if ss and ssp else None, "short_date": sdate,
           "inst_pct": _num(info.get("heldPercentInstitutions") * 100 if info.get("heldPercentInstitutions") is not None else None, 1),
           "insider_pct": _num(info.get("heldPercentInsiders") * 100 if info.get("heldPercentInsiders") is not None else None, 1)}
    if any(pos[k] is not None for k in ("short_pct_float", "inst_pct", "insider_pct")):
        out["positioning"] = pos
    return out


def fetch_macro() -> List[Dict[str, Any]]:
    """VIX and the 10-year Treasury yield for the market header (Yahoo ^VIX, ^TNX; ^TNX is quoted in percent)."""
    out: List[Dict[str, Any]] = []
    try:
        import yfinance as yf
    except Exception:
        return out
    for sym, label, kind in (("^VIX", "VIX", "vol"), ("^TNX", "US 10Y", "yield")):
        try:
            h = yf.Ticker(sym).history(period="3mo")["Close"].dropna()
            if len(h) < 2:
                continue
            px, prev = float(h.iloc[-1]), float(h.iloc[-2])
            out.append({"code": label, "kind": kind, "price": round(px, 2), "chg": round(px - prev, 2),
                        "chg_pct": round((px / prev - 1) * 100, 2), "spark": [round(float(x), 2) for x in h.iloc[-40:]]})
        except Exception as e:
            log(f"macro {sym} failed: {e}")
    return out


def _atm(chain: Any, spot: float) -> Optional[Dict[str, float]]:
    """ATM implied vol (avg of call+put at the strike nearest spot) and the straddle price. None if quotes look unusable."""
    calls, puts = chain.calls, chain.puts
    common = sorted(set(calls["strike"]) & set(puts["strike"]), key=lambda k: abs(k - spot))
    for k in common[:6]:
        c, p = calls[calls["strike"] == k].iloc[0], puts[puts["strike"] == k].iloc[0]
        ivs = [float(x) for x in (c["impliedVolatility"], p["impliedVolatility"]) if x and 0.03 < float(x) < 5]
        if len(ivs) < 2:
            continue

        def mid(r: Any) -> float:   # bid/ask are 0 after hours: fall back to the last trade
            b, a = float(r["bid"] or 0), float(r["ask"] or 0)
            return (b + a) / 2 if b > 0 and a > 0 else float(r["lastPrice"] or 0)
        m = mid(c) + mid(p)
        if m > 0:
            return {"iv": sum(ivs) / 2, "straddle": m, "strike": float(k)}
    return None


def earnings_moves(ed: Any, df: Any, limit: int = 8) -> List[Dict[str, Any]]:
    """Actual close-to-close reaction to each past earnings report (after-close reports react next session)."""
    moves: List[Dict[str, Any]] = []
    if ed is None or len(ed) == 0:
        return moves
    dates = [str(x)[:10] for x in df["date"]]
    closes = [float(x) for x in df["close"]]
    today = date.today()
    for ts, row in ed.iterrows():
        try:
            if ts.date() >= today or row.get("Reported EPS") != row.get("Reported EPS"):   # future, or not reported yet (NaN)
                continue
            rd = ts.date().isoformat()
            if ts.hour >= 12:      # after the close: base = that day's close, reaction = next session
                base = max((i for i, d in enumerate(dates) if d <= rd), default=None)
                if base is None or base + 1 >= len(closes):
                    continue
                mv = closes[base + 1] / closes[base] - 1
            else:                  # before the open: reaction = that session vs the previous close
                nxt = next((i for i, d in enumerate(dates) if d >= rd), None)
                if nxt is None or nxt == 0:
                    continue
                mv = closes[nxt] / closes[nxt - 1] - 1
            moves.append({"date": rd, "move_pct": round(mv * 100, 1)})
        except Exception:
            continue
    return moves[:limit]


def fetch_options(code: str, df: Any, earnings_date: Optional[str]) -> Optional[Dict[str, Any]]:
    """IV term structure + the implied move around the next earnings report, vs what past reports actually did."""
    try:
        import yfinance as yf
        t = yf.Ticker(code)
        exps = list(t.options or [])
        if not exps:
            return None
        spot = float(df["close"].iloc[-1])
        today = date.today()
        try:
            ed = t.get_earnings_dates(limit=14)
        except Exception:
            ed = None
        est = False
        if not earnings_date and ed is not None and len(ed):     # Yahoo's calendar has no confirmed date: use the projected one
            fut = sorted(x.date().isoformat() for x in ed.index if x.date() > today)
            if fut:
                earnings_date, est = fut[0], True
        dte = {e: (date.fromisoformat(e) - today).days for e in exps}
        ok = [e for e in exps if dte[e] >= 2]
        if not ok:
            return None
        pick = ok[:3]
        pre = post = None
        if earnings_date:
            post = next((e for e in ok if e > earnings_date), None)    # strictly after: an after-close report reacts next session
            pre = next((e for e in reversed(ok) if e < earnings_date), None)
            pick += [e for e in (pre, post) if e]
        pick += [e for e in ok if 30 <= dte[e] <= 220 and 15 <= int(e[8:10]) <= 21][:3]      # monthlies
        pick = sorted(set(pick))[:9]
        term = []
        for e in pick:
            a = _atm(t.option_chain(e), spot)
            if a:
                term.append({"exp": e, "dte": dte[e], "iv": round(a["iv"] * 100, 1), "move_pct": round(a["straddle"] / spot * 100, 1)})
        if not term:
            return None
        by = {x["exp"]: x for x in term}
        res: Dict[str, Any] = {"spot": round(spot, 2), "earnings_date": earnings_date,
                               "earnings_days": (date.fromisoformat(earnings_date) - today).days if earnings_date else None,
                               "earnings_est": est, "exp_pre": pre if pre in by else None, "exp_post": post if post in by else None,
                               "iv_pre": by[pre]["iv"] if pre in by else None, "iv_post": by[post]["iv"] if post in by else None,
                               "implied_move_pct": None, "move_quality": None, "term": term}
        # Event-day move: the post-earnings expiry carries extra variance versus a base vol with no event in it.
        # V_event = s_post^2*T_post - s_base^2*(T_post - 1 day);  expected |move| ~= 0.798*sqrt(V_event)
        base = pre if pre in by else next((e for e in by if post and e > post), None)
        if post in by and base:
            tp, tb = dte[post] / 365.0, dte[base] / 365.0
            sp, sb = by[post]["iv"] / 100.0, by[base]["iv"] / 100.0
            ev = sp * sp * tp - sb * sb * (tp - 1 / 365.0)
            if ev > 0:
                res["implied_move_pct"] = round(0.798 * math.sqrt(ev) * 100, 1)
                res["move_quality"] = "ok" if abs(dte[post] - dte[base]) <= 10 else "rough"   # far-apart expiries blur the event with the term structure
        hist = earnings_moves(ed, df)
        res["hist"] = hist
        res["hist_avg_abs"] = round(sum(abs(h["move_pct"]) for h in hist) / len(hist), 1) if hist else None
        return res
    except Exception as e:
        log(f"options {code} failed: {str(e)[:100]}")
        return None


def fetch_meta(code: str) -> Dict[str, Any]:
    """Company name, next earnings date and recent headlines from Yahoo (free)."""
    meta: Dict[str, Any] = {"name": code, "sector": None, "earnings_date": None, "earnings_days": None, "news": [],
                            "analyst": None, "positioning": None}
    try:
        import yfinance as yf
        t = yf.Ticker(code)
        try:
            info = t.get_info()
            meta["name"] = info.get("shortName") or info.get("longName") or code
            meta["sector"] = info.get("sector")
            meta.update(_analyst_positioning(info))
        except Exception:
            pass
        try:
            cal = t.calendar or {}
            ed = cal.get("Earnings Date")
            eds = ed if isinstance(ed, (list, tuple)) else [ed] if ed else []
            future = sorted(x for x in eds if isinstance(x, date) and x >= date.today())
            if future:
                meta["earnings_date"] = future[0].isoformat()
                meta["earnings_days"] = (future[0] - date.today()).days
        except Exception:
            pass
        try:
            for item in (t.news or [])[:5]:
                c = item.get("content", item)
                title = c.get("title")
                if not title:
                    continue
                url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link")
                meta["news"].append({
                    "title": title,
                    "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
                    "time": c.get("pubDate"),
                    "url": url,
                })
            meta["news"] = meta["news"][:4]
        except Exception:
            pass
    except Exception as e:
        log(f"meta failed for {code}: {e}")
    return meta


# ------------------------------------------------------------------ AI layer (optional)
def ai_available() -> bool:
    return any(os.getenv(k) for k in LLM_KEY_VARS)


def run_ai(codes: List[str], timeout: int = 1500) -> bool:
    """Run the repo's own full pipeline (LLM report) for the watchlist. Results land in its SQLite DB."""
    cmd = [sys.executable, "main.py", "--stocks", ",".join(codes), "--no-notify", "--no-market-review"]
    env = {**os.environ, "PYTHONUTF8": "1"}
    log(f"AI layer: running pipeline for {len(codes)} tickers (this calls your LLM)...")
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            log(f"pipeline exited {p.returncode}: {p.stderr[-300:]}")
        return p.returncode == 0
    except Exception as e:
        log(f"AI pipeline failed: {e}")
        return False


def read_ai(codes: List[str], max_age_days: int = 2) -> Dict[str, Dict[str, Any]]:
    """Latest saved LLM report per ticker (≤ max_age_days old)."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        from src.storage import DatabaseManager
        db = DatabaseManager.get_instance()
        for code in codes:
            rows = db.get_analysis_history(code=code, days=max_age_days, limit=1)
            if not rows:
                continue
            r = rows[0]
            raw: Dict[str, Any] = {}
            try:
                raw = json.loads(r.raw_result) if r.raw_result else {}
            except Exception:
                pass
            dash = raw.get("dashboard") or {}
            core = dash.get("core_conclusion") or {}
            intel = dash.get("intelligence") or {}
            pos = core.get("position_advice") or {}
            out[code] = {
                "advice": r.operation_advice, "score": r.sentiment_score, "trend": r.trend_prediction,
                "summary": r.analysis_summary, "one_sentence": core.get("one_sentence"),
                "no_position": pos.get("no_position"), "has_position": pos.get("has_position"),
                "risks": intel.get("risk_alerts") or [], "catalysts": intel.get("positive_catalysts") or [],
                "buy": r.ideal_buy, "buy2": r.secondary_buy, "stop": r.stop_loss, "target": r.take_profit,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
    except Exception as e:
        log(f"could not read AI reports: {e}")
    return out


# ------------------------------------------------------------------ snapshot assembly
def digest(snap: Dict[str, Any]) -> str:
    L = [f"US watchlist briefing - {snap['asof']}",
         f"Market: {snap['market']['regime'].upper()} - {snap['market']['note']}", ""]
    for grp, title in (("BUY", "BUY ZONE"), ("WATCH", "WATCH / TRIGGERS"), ("WAIT", "WAIT"), ("AVOID", "AVOID")):
        items = [i for i in snap["ideas"] if i["action"] == grp]
        if not items:
            continue
        L.append(title)
        for i in items:
            if i["action"] in ("BUY", "WATCH") and i.get("stop"):
                lvl = f"trigger {i['trigger']}" if i.get("trigger") else f"entry {i['entry_low']}-{i['entry_high']}"
                L.append(f"  {i['code']:<5} ${i['price']} ({i['chg_pct']:+.1f}%) {i['label']} | {lvl} stop {i['stop']} tgt {i['target']} R:R {i['rr']} | {i['conviction']}")
            else:
                L.append(f"  {i['code']:<5} ${i['price']} ({i['chg_pct']:+.1f}%) {i['label']}")
        L.append("")
    if snap.get("changes"):
        L.append("CHANGES vs previous briefing")
        L += [f"  {c['code']}: {c['from']} -> {c['to']}" for c in snap["changes"]]
    L.append("\nRule-based technical signals. Not financial advice.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stocks", help="comma list (default: dashboard/watchlist.json if present, else STOCK_LIST from .env; US tickers only)")
    ap.add_argument("--ai", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--no-options", action="store_true", help="skip option chains / implied-vol analysis (saves ~30s)")
    ap.add_argument("--notify", action="store_true",
                    help="push the text digest to the channels configured in .env (Telegram/Discord/Slack/email/...)")
    args = ap.parse_args()

    codes = watchlist(args.stocks)
    if not codes:
        log("no valid US tickers. Set STOCK_LIST in .env or pass --stocks")
        return 2
    t0 = time.time()
    log(f"watchlist: {', '.join(codes)}")

    mgr = DataFetcherManager()
    analyzer = StockTrendAnalyzer()

    # --- data
    frames: Dict[str, Any] = {}
    for code in codes + [c for c in ("SPY", "QQQ") if c not in codes]:
        try:
            df, _src = mgr.get_daily_data(code, days=900)
            if df is None or len(df) < E.MIN_BARS + 20:
                raise ValueError(f"only {0 if df is None else len(df)} bars")
            frames[code] = df
        except Exception as e:
            log(f"!! {code}: data fetch failed ({e})")
    codes = [c for c in codes if c in frames]
    if not codes:
        log("no data fetched for any ticker (network?)")
        return 3
    log(f"fetched {len(frames)} price histories ({time.time() - t0:.0f}s)")

    with ThreadPoolExecutor(max_workers=4) as ex:
        metas = dict(zip(codes, ex.map(fetch_meta, codes)))

    options: Dict[str, Any] = {}
    if not args.no_options:
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(lambda c: fetch_options(c, frames[c], metas[c].get("earnings_date")), codes))
        options = {c: r for c, r in zip(codes, res) if r}
        log(f"options for {len(options)}/{len(codes)} tickers ({time.time() - t1:.0f}s)")

    # --- current plans
    now = {c: E.analyse_now(c, frames[c], analyzer) for c in codes}
    spy = now.get("SPY") or E.analyse_now("SPY", frames["SPY"], analyzer)
    qqq = now.get("QQQ") or (E.analyse_now("QQQ", frames["QQQ"], analyzer) if "QQQ" in frames else None)
    market = E.market_regime(spy, qqq)
    market["indices"] = [{"code": x["code"], "price": x["price"], "chg_pct": x["chg_pct"], "trend": x["trend"],
                          "spark": x["spark"], "spark_dates": x["spark_dates"]} for x in (spy, qqq) if x]
    market["macro"] = fetch_macro()
    asof = spy["asof"]

    # --- history back-fill (replay the playbook over the last few bars)
    backfill: Dict[str, List[Dict[str, Any]]] = {}
    for c in codes:
        for row in E.daily_plans(c, frames[c], analyzer, history_days=BACKFILL_DAYS):
            backfill.setdefault(row["date"], []).append({"code": c, **row})

    # --- AI layer
    use_ai = args.ai == "on" or (args.ai == "auto" and ai_available())
    ai: Dict[str, Dict[str, Any]] = {}
    if use_ai:
        run_ai(codes)
        ai = read_ai(codes)
        log(f"AI reports found for {len(ai)}/{len(codes)} tickers")
    elif args.ai == "auto":
        log("no LLM key in .env: skipping AI commentary (quant layer only)")

    # --- compose ideas
    ideas = []
    for c in codes:
        n = now[c]
        m = metas[c]
        cv = E.conviction(n["score"], n, market["regime"], m["earnings_days"])
        ideas.append({**n, **m, **cv, "ai": ai.get(c)})
    order = {"BUY": 0, "WATCH": 1, "WAIT": 2, "HOLD": 3, "AVOID": 4}
    clevel = {"High": 0, "Medium": 1, "Low": 2}
    ideas.sort(key=lambda i: (order.get(i["action"], 9), clevel[i["conviction"]], -i["adj_score"]))

    # --- history merge
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data").mkdir(exist_ok=True)
    hist_path = SITE / "data" / "history.json"
    history: List[Dict[str, Any]] = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    by_date = {h["date"]: h for h in history}

    def compact(i):   # drop heavy fields for stored history
        return {k: v for k, v in i.items() if k not in ("spark", "spark_dates", "news", "levels", "analyst", "positioning")}

    for d_, rows in backfill.items():
        if d_ == asof:
            continue
        if d_ not in by_date:
            by_date[d_] = {"date": d_, "backfilled": True,
                           "items": [{**r, "conviction": None} for r in sorted(rows, key=lambda r: r["code"])]}
        else:   # day already stored: fill in tickers added since (their earlier days would otherwise stay empty)
            have = {it["code"] for it in by_date[d_]["items"]}
            by_date[d_]["items"] += [{**r, "conviction": None} for r in sorted(rows, key=lambda r: r["code"]) if r["code"] not in have]
    prev = [h for h in sorted(by_date.values(), key=lambda h: h["date"]) if h["date"] < asof]
    changes = []
    if prev:
        pmap = {i["code"]: i for i in prev[-1]["items"]}
        for i in ideas:
            p = pmap.get(i["code"])
            if p and p.get("setup") != i["setup"]:
                changes.append({"code": i["code"], "from": p.get("label"), "to": i["label"], "prev_date": prev[-1]["date"]})
    by_date[asof] = {"date": asof, "backfilled": False, "market": market["regime"],
                     "items": [compact(i) for i in ideas]}
    history = sorted(by_date.values(), key=lambda h: h["date"])[-MAX_HISTORY_DAYS:]
    hist_path.write_text(json.dumps(history, separators=(",", ":"), default=str), encoding="utf-8")

    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": asof, "watchlist": codes, "market": market, "ideas": ideas, "changes": changes,
        "ai_enabled": bool(ai), "ai_key_present": ai_available(),
    }
    (SITE / "data" / "latest.json").write_text(json.dumps(snap, indent=1, default=str), encoding="utf-8")
    (SITE / "digest.txt").write_text(digest(snap), encoding="utf-8")

    charts: Dict[str, List[List[Any]]] = {}
    for c in codes:
        d = frames[c].tail(CHART_BARS)
        rows = []
        for dt, o, h, l, cl, v in zip(d["date"], d["open"], d["high"], d["low"], d["close"], d["volume"]):
            try:
                vals = [float(o), float(h), float(l), float(cl), float(v or 0)]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(x) for x in vals):
                rows.append([str(dt)[:10], round(vals[0], 2), round(vals[1], 2), round(vals[2], 2), round(vals[3], 2), int(vals[4])])
        charts[c] = rows

    payload = {"latest": snap, "history": history, "charts": charts, "options": options}
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    blob = json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")
    (SITE / "index.html").write_text(tpl.replace("/*__DATA__*/null", blob), encoding="utf-8")

    if args.notify:
        try:
            from src.notification import NotificationService
            svc = NotificationService()
            url = os.getenv("DASHBOARD_URL")
            text = digest(snap) + (("\n\nFull dashboard: " + url) if url else "")
            if svc.is_available():
                log("digest sent" if svc.send(text) else "digest NOT sent (all channels failed)")
            else:
                log("--notify: no notification channel configured in .env (see .env.example)")
        except Exception as e:
            log(f"notify failed: {e}")

    log(f"done in {time.time() - t0:.0f}s -> {SITE / 'index.html'}")
    print("\n" + digest(snap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
