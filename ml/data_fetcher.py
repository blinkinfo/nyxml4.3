"""MEXC data fetcher for BTC/USDT OHLCV + CVD — BLUEPRINT sections 3.1-3.5.

All data sourced from MEXC only (spot + futures). NO Binance, NO Coinbase.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

import ccxt
import httpx
import pandas as pd

log = logging.getLogger(__name__)

MEXC_CVD_URL = "https://contract.mexc.com/api/v1/contract/kline/BTC_USDT"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    """Convert ccxt OHLCV list to a clean DataFrame."""
    df = pd.DataFrame(ohlcv, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def _paginate_ohlcv(exchange, symbol: str, timeframe: str, start_ms: int, end_ms: int, batch: int = 500) -> pd.DataFrame:
    """Paginate ccxt fetch_ohlcv calls from start_ms to end_ms.

    MEXC spot caps at 500 candles per request; futures may allow more.
    We probe the actual page size from the first response and stop when
    returned count < that size (meaning we hit the end of history).
    """
    all_rows = []
    since = start_ms
    actual_page_size = None  # determined from first successful response

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch)
        except Exception as e:
            log.warning("fetch_ohlcv error (%s %s since=%d): %s", symbol, timeframe, since, e)
            break
        if not ohlcv:
            break
        all_rows.extend(ohlcv)

        # Determine effective page size from first batch
        if actual_page_size is None:
            actual_page_size = len(ohlcv)

        last_ts = ohlcv[-1][0]
        # Stop if we reached end of requested range or got a partial page
        if last_ts >= end_ms or len(ohlcv) < actual_page_size:
            break
        since = last_ts + 1
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = _ohlcv_to_df(all_rows)
    # Deduplicate on timestamp, sort ascending
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Filter to [start_ms, end_ms)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Section 3.1 — 5m candles (spot)
# ---------------------------------------------------------------------------

def fetch_5m(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT 5m spot candles from MEXC."""
    exchange = ccxt.mexc()
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT", "5m", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.2 — 15m candles (swap/futures)
# ---------------------------------------------------------------------------

def fetch_15m(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT:USDT 15m futures candles from MEXC."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT:USDT", "15m", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.3 — 1h candles (swap/futures)
# ---------------------------------------------------------------------------

def fetch_1h(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT:USDT 1h futures candles from MEXC."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT:USDT", "1h", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.4 — Funding rate history
# ---------------------------------------------------------------------------

MEXC_FUNDING_URL = "https://contract.mexc.com/api/v1/contract/funding_rate/history"


def _funding_records_to_df(records: list, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Deduplicate, filter to [start_ms, end_ms], sort, and return clean DataFrame."""
    if not records:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].reset_index(drop=True)
    return df


def _fetch_funding_ccxt(exchange, start_ms: int, end_ms: int) -> list:
    """Try to paginate funding rate history via ccxt with stall detection.

    Returns a list of raw record dicts with keys 'timestamp' and 'funding_rate'.
    Stops if: batch is empty, last_ts stalls (two consecutive pages same), or last_ts >= end_ms.
    Does NOT stop just because len(batch) < 100.
    """
    records = []
    since = start_ms
    prev_last_ts: int | None = None

    while since < end_ms:
        try:
            batch = exchange.fetch_funding_rate_history("BTC/USDT:USDT", since=since, limit=100)
        except Exception as e:
            log.warning("ccxt fetch_funding_rate_history error since=%d: %s", since, e)
            break
        if not batch:
            break
        for r in batch:
            ts = r.get("timestamp")
            rate = r.get("fundingRate")
            if ts is not None and rate is not None:
                records.append({
                    "timestamp": pd.Timestamp(ts, unit="ms", tz="UTC"),
                    "funding_rate": float(rate),
                })
        last_ts = batch[-1].get("timestamp", 0) or 0
        # Stall detection: same last_ts as previous page means exchange ignores `since`
        if last_ts == prev_last_ts:
            log.info("ccxt funding pagination stalled at ts=%d — stopping ccxt strategy", last_ts)
            break
        if last_ts >= end_ms:
            break
        prev_last_ts = last_ts
        since = last_ts + 1
        time.sleep(0.1)

    return records


def _fetch_funding_rest(start_ms: int, end_ms: int) -> list:
    """Fall back to MEXC direct REST API for funding rate history.

    Paginates GET https://contract.mexc.com/api/v1/contract/funding_rate/history
    with page_num (1-based) and page_size=100.
    Each record has settleTime (ms) and fundingRate.
    Stops when a page returns no data or all records are older than start_ms.
    """
    records = []
    page_num = 1

    with httpx.Client(timeout=30) as client:
        while True:
            params = {
                "symbol": "BTC_USDT",
                "page_num": page_num,
                "page_size": 100,
            }
            try:
                resp = client.get(MEXC_FUNDING_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.warning("MEXC REST funding page %d error: %s", page_num, e)
                break

            page_data = data.get("data", {})
            # The API returns {"data": {"resultList": [...], ...}} or similar
            if isinstance(page_data, dict):
                items = page_data.get("resultList", page_data.get("result", []))
            elif isinstance(page_data, list):
                items = page_data
            else:
                items = []

            if not items:
                log.info("MEXC REST funding page %d returned empty — stopping", page_num)
                break

            page_records = []
            for item in items:
                ts = item.get("settleTime")
                rate = item.get("fundingRate")
                if ts is not None and rate is not None:
                    page_records.append({
                        "timestamp": pd.Timestamp(int(ts), unit="ms", tz="UTC"),
                        "funding_rate": float(rate),
                    })

            if not page_records:
                break

            records.extend(page_records)

            # REST returns newest-first; if oldest record on this page is before start_ms, we have enough
            oldest_ts = min(r["timestamp"] for r in page_records)
            start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            if oldest_ts <= start_dt:
                log.info("MEXC REST funding: reached start boundary at page %d", page_num)
                break

            page_num += 1
            time.sleep(0.1)

    log.info("MEXC REST funding strategy: fetched %d raw records across %d pages", len(records), page_num)
    return records


def fetch_funding(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT:USDT funding rate history from MEXC.

    Strategy 1: ccxt with stall detection (stops if two consecutive pages return
    the same last timestamp, meaning the exchange ignores the `since` param).
    Strategy 2: If ccxt coverage < 80% of the requested window, fall back to
    MEXC's direct REST endpoint which supports page_num pagination.
    """
    window_ms = end_ms - start_ms
    coverage_threshold = 0.80

    # --- Strategy 1: ccxt ---
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    exchange.load_markets()

    ccxt_records = _fetch_funding_ccxt(exchange, start_ms, end_ms)
    ccxt_df = _funding_records_to_df(ccxt_records, start_ms, end_ms)

    if not ccxt_df.empty:
        covered_ms = (
            ccxt_df["timestamp"].max() - ccxt_df["timestamp"].min()
        ).total_seconds() * 1000
        coverage = covered_ms / window_ms if window_ms > 0 else 0.0
    else:
        coverage = 0.0

    log.info(
        "fetch_funding ccxt: %d records, coverage=%.1f%% (threshold=%.0f%%)",
        len(ccxt_df), coverage * 100, coverage_threshold * 100,
    )

    if coverage >= coverage_threshold:
        log.info("fetch_funding: ccxt strategy sufficient — returning %d records", len(ccxt_df))
        return ccxt_df

    # --- Strategy 2: MEXC direct REST fallback ---
    log.info(
        "fetch_funding: ccxt coverage %.1f%% < %.0f%% — falling back to MEXC REST API",
        coverage * 100, coverage_threshold * 100,
    )

    rest_records = _fetch_funding_rest(start_ms, end_ms)
    rest_df = _funding_records_to_df(rest_records, start_ms, end_ms)

    if not rest_df.empty:
        log.info("fetch_funding: REST strategy returned %d records", len(rest_df))
        return rest_df

    # Both failed — return whatever ccxt gave us (may be empty)
    log.warning(
        "fetch_funding: both strategies failed or returned insufficient data — "
        "returning %d ccxt records", len(ccxt_df)
    )
    return ccxt_df if not ccxt_df.empty else pd.DataFrame(columns=["timestamp", "funding_rate"])


# ---------------------------------------------------------------------------
# Section 3.5 — CVD via direct MEXC futures REST (NOT ccxt)
# ---------------------------------------------------------------------------

def _cvd_proxy(open_: float, high: float, low: float, close: float, vol: float):
    """Compute buy_vol/sell_vol proxy from OHLCV."""
    if (high - low) > 0:
        buy_vol = vol * (close - low) / (high - low)
        sell_vol = vol * (high - close) / (high - low)
    else:
        buy_vol = vol * 0.5
        sell_vol = vol * 0.5
    return buy_vol, sell_vol


def fetch_cvd(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 5m CVD data directly from MEXC futures REST endpoint."""
    # 100 candles * 5min = 500min window per request
    window_sec = 100 * 5 * 60
    start_sec = start_ms // 1000
    end_sec = end_ms // 1000

    records = []
    cursor = start_sec

    with httpx.Client(timeout=30) as client:
        while cursor < end_sec:
            batch_end = min(cursor + window_sec, end_sec)
            params = {
                "interval": "Min5",
                "start": cursor,
                "end": batch_end,
            }
            try:
                resp = client.get(MEXC_CVD_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.warning("CVD fetch error cursor=%d: %s", cursor, e)
                break

            candle_data = data.get("data", {})
            times = candle_data.get("time", [])
            opens = candle_data.get("open", [])
            highs = candle_data.get("high", [])
            lows = candle_data.get("low", [])
            closes = candle_data.get("close", [])
            vols = candle_data.get("vol", [])

            if not times:
                break

            for i in range(len(times)):
                try:
                    ts_sec = int(times[i])
                    o = float(opens[i])
                    h = float(highs[i])
                    lo = float(lows[i])
                    c = float(closes[i])
                    v = float(vols[i])
                    bv, sv = _cvd_proxy(o, h, lo, c, v)
                    records.append({
                        "timestamp": pd.Timestamp(ts_sec, unit="s", tz="UTC"),
                        "open": o,
                        "high": h,
                        "low": lo,
                        "close": c,
                        "volume": v,
                        "buy_vol": bv,
                        "sell_vol": sv,
                    })
                except (IndexError, ValueError, TypeError):
                    continue

            last_time = int(times[-1]) if times else cursor
            if last_time >= end_sec or len(times) < 100:
                break
            cursor = last_time + 300  # next 5m window
            time.sleep(0.05)

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol"])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Filter to range
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# fetch_all — fetch last N months of all 5 sources
# ---------------------------------------------------------------------------

def fetch_all(months: int = 5) -> dict:
    """Fetch all 5 data sources for the last `months` months.

    Returns dict with keys: df5, df15, df1h, funding, cvd
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=months * 30)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    log.info("fetch_all: start=%s end=%s", start.isoformat(), now.isoformat())

    print(f"  Fetching 5m candles ({months} months)...")
    df5 = fetch_5m(start_ms, end_ms)
    print(f"  -> {len(df5)} 5m candles")

    print("  Fetching 15m candles...")
    df15 = fetch_15m(start_ms, end_ms)
    print(f"  -> {len(df15)} 15m candles")

    print("  Fetching 1h candles...")
    df1h = fetch_1h(start_ms, end_ms)
    print(f"  -> {len(df1h)} 1h candles")

    print("  Fetching funding rate history...")
    funding = fetch_funding(start_ms, end_ms)
    print(f"  -> {len(funding)} funding records")

    print("  Fetching CVD data...")
    cvd = fetch_cvd(start_ms, end_ms)
    print(f"  -> {len(cvd)} CVD candles")

    return {"df5": df5, "df15": df15, "df1h": df1h, "funding": funding, "cvd": cvd}


# ---------------------------------------------------------------------------
# Live fetchers (for MLStrategy real-time inference)
# ---------------------------------------------------------------------------

def fetch_live_5m(limit: int = 400) -> pd.DataFrame:
    """Fetch last `limit` 5m candles from MEXC spot."""
    exchange = ccxt.mexc()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_15m(limit: int = 100) -> pd.DataFrame:
    """Fetch last `limit` 15m candles from MEXC futures."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="15m", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_1h(limit: int = 60) -> pd.DataFrame:
    """Fetch last `limit` 1h candles from MEXC futures."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_funding() -> float | None:
    """Fetch the current funding rate for BTC/USDT:USDT. Returns single float."""
    try:
        exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
        result = exchange.fetch_funding_rate("BTC/USDT:USDT")
        return float(result.get("fundingRate", 0.0))
    except Exception as e:
        log.warning("fetch_live_funding error: %s", e)
        return None


def fetch_live_funding_history(n_periods: int = 24) -> list[float]:
    """Fetch the last `n_periods` historical funding rates for seeding the zscore buffer.

    MEXC funding is every 8 hours (3/day). 24 periods = 8 days of history.
    Returns a list of floats sorted oldest-first, ready to populate a deque(maxlen=24).
    """
    # 24 periods * 8h = 192h back; add margin to ensure we get enough records
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=(n_periods + 4) * 8)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    try:
        df = fetch_funding(start_ms, end_ms)
        if df.empty:
            return []
        # Return the last n_periods values, oldest-first
        rates = df["funding_rate"].tail(n_periods).tolist()
        log.info("fetch_live_funding_history: fetched %d records for buffer seed", len(rates))
        return rates
    except Exception as e:
        log.warning("fetch_live_funding_history error: %s", e)
        return []


def fetch_live_cvd(n_candles: int = 400) -> pd.DataFrame:
    """Fetch last `n_candles` 5m candles from MEXC futures REST and compute CVD proxy."""
    end_sec = int(time.time())
    # Fetch a bit more to ensure we get n_candles
    start_sec = end_sec - (n_candles + 10) * 300
    params = {
        "interval": "Min5",
        "start": start_sec,
        "end": end_sec,
    }
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(MEXC_CVD_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("fetch_live_cvd error: %s", e)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol"])

    candle_data = data.get("data", {})
    times = candle_data.get("time", [])
    opens = candle_data.get("open", [])
    highs = candle_data.get("high", [])
    lows = candle_data.get("low", [])
    closes = candle_data.get("close", [])
    vols = candle_data.get("vol", [])

    records = []
    for i in range(len(times)):
        try:
            ts_sec = int(times[i])
            o = float(opens[i])
            h = float(highs[i])
            lo = float(lows[i])
            c = float(closes[i])
            v = float(vols[i])
            bv, sv = _cvd_proxy(o, h, lo, c, v)
            records.append({
                "timestamp": pd.Timestamp(ts_sec, unit="s", tz="UTC"),
                "open": o, "high": h, "low": lo, "close": c,
                "volume": v, "buy_vol": bv, "sell_vol": sv,
            })
        except (IndexError, ValueError, TypeError):
            continue

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol"])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df.tail(n_candles).reset_index(drop=True)
