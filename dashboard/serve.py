# -*- coding: utf-8 -*-
"""
Local server for the dashboard: serves dashboard/site/ and powers the "Analyze" tab.

    python dashboard/serve.py                    # http://127.0.0.1:8765  (this PC only)
    python dashboard/serve.py --host 0.0.0.0     # also reachable from your phone on the same Wi-Fi

GET /api/analyze?symbol=AMD[&ai=1]  -> one idea card as JSON (live prices; AI report if ai=1 and a key is set)
POST /api/watchlist {action: add|remove, symbol}  -> edits dashboard/watchlist.json and rebuilds the site
There is no authentication: only use --host 0.0.0.0 on a network you trust.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import build, engine as E  # noqa: E402  (build loads .env and sets cwd)
from src.stock_analyzer import StockTrendAnalyzer  # noqa: E402

SITE = build.SITE
LOCK = threading.Lock()                     # the data layer isn't built for parallel hammering
_mgr = build.DataFetcherManager()
_an = StockTrendAnalyzer()
_regime_cache = {"t": 0.0, "v": None}


def _regime():
    if time.time() - _regime_cache["t"] > 600 or not _regime_cache["v"]:
        spy = E.analyse_now("SPY", _mgr.get_daily_data("SPY", days=400)[0], _an)
        qqq = E.analyse_now("QQQ", _mgr.get_daily_data("QQQ", days=400)[0], _an)
        _regime_cache.update(t=time.time(), v=E.market_regime(spy, qqq))
    return _regime_cache["v"]


def analyze(symbol: str, want_ai: bool):
    df, _ = _mgr.get_daily_data(symbol, days=900)
    if df is None or len(df) < E.MIN_BARS:
        raise ValueError("not enough price history for that symbol")
    now = E.analyse_now(symbol, df, _an)
    meta = build.fetch_meta(symbol)
    cv = E.conviction(now["score"], now, _regime()["regime"], meta["earnings_days"])
    ai = None
    if want_ai and build.ai_available():
        build.run_ai([symbol], timeout=600)
        ai = build.read_ai([symbol]).get(symbol)
    return {**now, **meta, **cv, "ai": ai}


def _current_list():
    return build.watchlist(None)


def _save_and_rebuild(codes):
    """Persist the list, then regenerate the site (about 5-10s; AI commentary is skipped, the scheduled build restores it)."""
    build.WATCHLIST_FILE.write_text(json.dumps(codes), encoding="utf-8")
    p = subprocess.run([sys.executable, str(Path(build.__file__)), "--ai", "off"], cwd=build.ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if p.returncode != 0:
        raise RuntimeError("rebuild failed: " + (p.stderr or p.stdout)[-200:])


def edit_watchlist(action: str, symbol: str):
    codes = _current_list()
    if action == "add":
        if symbol in codes:
            raise ValueError(f"{symbol} is already in the watchlist")
        df, _ = _mgr.get_daily_data(symbol, days=400)
        if df is None or len(df) < E.MIN_BARS:
            raise ValueError(f"no usable price history for {symbol}")
        codes.append(symbol)
    elif action == "remove":
        if symbol not in codes:
            raise ValueError(f"{symbol} is not in the watchlist")
        if len(codes) == 1:
            raise ValueError("keep at least one ticker")
        codes.remove(symbol)
    else:
        raise ValueError("unknown action")
    _save_and_rebuild(codes)
    return codes


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(SITE), **kw)

    def _json(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/watchlist":
            return self._json(200, {"watchlist": _current_list()})
        if u.path != "/api/analyze":
            return super().do_GET()
        q = parse_qs(u.query)
        sym = (q.get("symbol", [""])[0] or "").strip().upper()
        if not build.US_TICKER.match(sym):
            return self._json(400, {"error": "enter a US ticker like AAPL"})
        try:
            with LOCK:
                return self._json(200, analyze(sym, q.get("ai", ["0"])[0] == "1"))
        except Exception as e:  # surface the reason to the UI
            return self._json(502, {"error": str(e)[:200]})

    def do_POST(self):
        if urlparse(self.path).path != "/api/watchlist":
            return self._json(404, {"error": "not found"})
        # JSON content-type forces a CORS preflight, so other websites can't edit your list from the browser
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json(415, {"error": "expected application/json"})
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4096)) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad JSON"})
        sym = str(body.get("symbol", "")).strip().upper()
        if not build.US_TICKER.match(sym):
            return self._json(400, {"error": "enter a US ticker like AAPL"})
        try:
            with LOCK:
                return self._json(200, {"watchlist": edit_watchlist(str(body.get("action")), sym)})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            return self._json(502, {"error": str(e)[:200]})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not (SITE / "index.html").exists():
        sys.exit("No site yet. Run: python dashboard/build.py")
    print(f"Dashboard: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
