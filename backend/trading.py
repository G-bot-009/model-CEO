"""Disciplined rule-based trading engine for G Office.

Phases:
  1) paper  — simulate fills against the live price (no real money)
  2) live   — Binance / Bybit real orders (signed REST), opt-in only
  3) mt5    — Forex/Gold via MetaApi cloud bridge

The strategy is deterministic and rule-based (the AI authors/tunes the rules,
it does NOT predict the market). Every bot runs behind a safety layer:
default paper mode, per-day loss limit + profit target kill-switch, and a
max position size. Nothing trades on its own until the user starts the bot.

Honest limits: no bot can predict the market or be "risk-free". A stop-loss
caps risk on the remaining size; the first leg and slippage/gaps can still lose.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request

# ---------------------------------------------------------------- strategy ----
DEFAULT_CONFIG = {
    "symbol": "BTCUSDT",
    "side": "long",            # long | short  (manual direction for the demo)
    "size_usd": 50.0,          # notional per entry
    "tp_legs_pct": [0.5, 1.0, 1.5],   # 3 take-profit legs (% from entry)
    "sl_pct": 0.5,             # stop-loss %
    "move_sl_to_be_after_leg": 1,     # after leg N fills, move SL to breakeven
    "daily_target_usd": 100.0, # stop for the day once reached
    "daily_loss_limit_usd": 50.0,     # hard kill-switch for the day
    "max_size_usd": 200.0,     # never risk more than this per position
}


def normalize_config(cfg: dict) -> dict:
    c = dict(DEFAULT_CONFIG)
    c.update({k: v for k, v in (cfg or {}).items() if k in DEFAULT_CONFIG})
    c["symbol"] = str(c["symbol"]).upper().strip() or "BTCUSDT"
    c["side"] = "short" if str(c["side"]).lower().startswith("s") else "long"
    legs = c.get("tp_legs_pct") or [0.5, 1.0, 1.5]
    c["tp_legs_pct"] = [float(x) for x in legs][:3] or [1.0]
    for k in ("size_usd", "sl_pct", "daily_target_usd", "daily_loss_limit_usd", "max_size_usd"):
        c[k] = max(0.0, float(c[k]))
    c["size_usd"] = min(c["size_usd"], c["max_size_usd"]) or 10.0
    c["move_sl_to_be_after_leg"] = int(c["move_sl_to_be_after_leg"])
    return c


def new_state() -> dict:
    return {"pos": None, "day": time.strftime("%Y-%m-%d"), "daily_pnl": 0.0, "halted": False}


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def step(cfg: dict, state: dict, price: float) -> tuple:
    """Advance one tick at ``price``. Returns (new_state, events) — pure logic.

    events: list of {"kind","text","pnl"} describing fills/opens/closes.
    """
    cfg = normalize_config(cfg)
    st = json.loads(json.dumps(state or new_state()))     # deep copy
    ev: list = []
    if price is None or price <= 0:
        return st, ev

    # daily reset
    if st.get("day") != _today():
        st.update({"day": _today(), "daily_pnl": 0.0, "halted": False})

    long = cfg["side"] == "long"
    sign = 1 if long else -1

    # --- manage an open position ------------------------------------------
    pos = st.get("pos")
    if pos:
        e = pos["entry"]
        for leg in pos["legs"]:
            if leg["filled"]:
                continue
            hit = (price >= leg["tp"]) if long else (price <= leg["tp"])
            if hit:
                leg["filled"] = True
                pnl = leg["qty"] * (leg["tp"] - e) * sign
                st["daily_pnl"] += pnl
                ev.append({"kind": "tp", "text": f"TP{leg['i']} @ {leg['tp']:.4f}", "pnl": pnl})
                if leg["i"] >= cfg["move_sl_to_be_after_leg"]:
                    pos["sl"] = e          # move stop to breakeven
                    ev.append({"kind": "sl_move", "text": "เลื่อน SL มาที่ทุน (breakeven)", "pnl": 0.0})
        # stop-loss / all legs done
        rem = [l for l in pos["legs"] if not l["filled"]]
        stop_hit = (price <= pos["sl"]) if long else (price >= pos["sl"])
        if rem and stop_hit:
            qty = sum(l["qty"] for l in rem)
            pnl = qty * (pos["sl"] - e) * sign
            st["daily_pnl"] += pnl
            kind = "be" if abs(pos["sl"] - e) < 1e-9 else "sl"
            ev.append({"kind": kind, "text": f"ปิดส่วนที่เหลือ @ {pos['sl']:.4f}", "pnl": pnl})
            rem = []
        if not rem:
            st["pos"] = None
            pos = None

    # --- daily kill-switch ------------------------------------------------
    if not st.get("halted"):
        if cfg["daily_target_usd"] and st["daily_pnl"] >= cfg["daily_target_usd"]:
            st["halted"] = True
            ev.append({"kind": "halt", "text": f"ถึงเป้ากำไร/วัน (+{st['daily_pnl']:.2f}) — หยุดถึงพรุ่งนี้", "pnl": 0.0})
        elif cfg["daily_loss_limit_usd"] and st["daily_pnl"] <= -cfg["daily_loss_limit_usd"]:
            st["halted"] = True
            ev.append({"kind": "halt", "text": f"ถึงลิมิตขาดทุน/วัน ({st['daily_pnl']:.2f}) — หยุด", "pnl": 0.0})

    # --- open a new position when flat ------------------------------------
    if st.get("pos") is None and not st.get("halted"):
        e = price
        qty = cfg["size_usd"] / e
        n = len(cfg["tp_legs_pct"])
        legs = []
        for i, tp_pct in enumerate(cfg["tp_legs_pct"], start=1):
            tp = e * (1 + sign * tp_pct / 100.0)
            legs.append({"i": i, "tp": tp, "qty": qty / n, "filled": False})
        sl = e * (1 - sign * cfg["sl_pct"] / 100.0)
        st["pos"] = {"entry": e, "qty": qty, "sl": sl, "legs": legs,
                     "side": cfg["side"], "opened_at": int(time.time())}
        ev.append({"kind": "open", "text": f"เปิด {cfg['side'].upper()} {cfg['symbol']} @ {e:.4f} (SL {sl:.4f})", "pnl": 0.0})

    return st, ev


# ----------------------------------------------------------------- brokers ----
_UA = "GOffice-trader/1.0"


def _http(url, data=None, headers=None, method=None, timeout=10):
    h = {"User-Agent": _UA, "Accept": "application/json"}
    if headers:
        h.update(headers)
    body = data.encode() if isinstance(data, str) else data
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_price(exchange: str, symbol: str, creds: dict | None = None) -> float:
    """Public last price. Used by paper mode and as a sanity check for live."""
    ex = (exchange or "").lower()
    sym = symbol.upper()
    if ex == "bybit":
        d = _http(f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={sym}")
        return float(d["result"]["list"][0]["lastPrice"])
    if ex == "mt5":
        # MetaApi price requires account creds; handled in live path. No public price.
        raise RuntimeError("MT5 price ต้องใช้ MetaApi (โหมด live)")
    # default: binance futures public ticker
    d = _http(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}")
    return float(d["price"])


def _binance_order(creds: dict, symbol: str, side: str, qty: float) -> dict:
    """Place a market order on Binance USDT-M futures (signed)."""
    base = "https://fapi.binance.com"
    ts = int(time.time() * 1000)
    params = {"symbol": symbol.upper(), "side": "BUY" if side == "long" else "SELL",
              "type": "MARKET", "quantity": qty, "timestamp": ts}
    q = urllib.parse.urlencode(params)
    sig = hmac.new(creds["api_secret"].encode(), q.encode(), hashlib.sha256).hexdigest()
    return _http(f"{base}/fapi/v1/order?{q}&signature={sig}", data=b"",
                 headers={"X-MBX-APIKEY": creds["api_key"]}, method="POST")


def _bybit_order(creds: dict, symbol: str, side: str, qty: float) -> dict:
    """Place a market order on Bybit v5 (signed)."""
    base = "https://api.bybit.com"
    ts = str(int(time.time() * 1000))
    body = json.dumps({"category": "linear", "symbol": symbol.upper(),
                       "side": "Buy" if side == "long" else "Sell",
                       "orderType": "Market", "qty": str(qty)})
    recv = "5000"
    pre = ts + creds["api_key"] + recv + body
    sig = hmac.new(creds["api_secret"].encode(), pre.encode(), hashlib.sha256).hexdigest()
    headers = {"X-BAPI-API-KEY": creds["api_key"], "X-BAPI-TIMESTAMP": ts,
               "X-BAPI-RECV-WINDOW": recv, "X-BAPI-SIGN": sig, "Content-Type": "application/json"}
    return _http(f"{base}/v5/order/create", data=body, headers=headers, method="POST")


def _metaapi_order(creds: dict, symbol: str, side: str, volume: float) -> dict:
    """Place a market order via MetaApi (MT5). creds: {token, account_id}."""
    region = creds.get("region", "new-york")
    base = f"https://mt-client-api-v1.{region}.agiliumtrade.ai"
    acc = creds["account_id"]
    body = json.dumps({"actionType": "ORDER_TYPE_BUY" if side == "long" else "ORDER_TYPE_SELL",
                       "symbol": symbol, "volume": volume})
    headers = {"auth-token": creds["token"], "Content-Type": "application/json"}
    return _http(f"{base}/users/current/accounts/{acc}/trade", data=body, headers=headers, method="POST")


def place_market(exchange: str, creds: dict, symbol: str, side: str, qty: float) -> dict:
    """Live market order. Raises on failure (caller logs + halts)."""
    ex = (exchange or "").lower()
    if ex == "bybit":
        return _bybit_order(creds, symbol, side, qty)
    if ex == "mt5":
        return _metaapi_order(creds, symbol, side, qty)
    return _binance_order(creds, symbol, side, qty)
