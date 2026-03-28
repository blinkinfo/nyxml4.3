"""BTC 5-min slot helpers — compute slot boundaries & fetch prices from Gamma + CLOB APIs."""

from __future__ import annotations

import logging
import json
import math
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import config as cfg

log = logging.getLogger(__name__)

SLOT_DURATION = 300  # 5 minutes in seconds


# ---------------------------------------------------------------------------
# Slot boundary helpers
# ---------------------------------------------------------------------------

def _slot_start_ts(dt: datetime) -> int:
    """Return the unix timestamp of the current 5-min slot start for *dt*."""
    epoch = int(dt.timestamp())
    return epoch - (epoch % SLOT_DURATION)


def get_current_slot_info() -> dict[str, Any]:
    """Compute current slot N boundaries.

    Returns dict with:
      slot_start_dt, slot_end_dt, slot_start_ts, slug,
      slot_start_str ("HH:MM"), slot_end_str ("HH:MM")
    """
    now = datetime.now(timezone.utc)
    start_ts = _slot_start_ts(now)
    end_ts = start_ts + SLOT_DURATION
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    slug = f"btc-updown-5m-{start_ts}"
    return {
        "slot_start_dt": start_dt,
        "slot_end_dt": end_dt,
        "slot_start_ts": start_ts,
        "slug": slug,
        "slot_start_str": start_dt.strftime("%H:%M"),
        "slot_end_str": end_dt.strftime("%H:%M"),
        "slot_start_full": start_dt.strftime("%Y-%m-%d %H:%M"),
        "slot_end_full": end_dt.strftime("%Y-%m-%d %H:%M"),
    }


def get_next_slot_info() -> dict[str, Any]:
    """Compute next slot N+1 boundaries."""
    now = datetime.now(timezone.utc)
    start_ts = _slot_start_ts(now) + SLOT_DURATION
    end_ts = start_ts + SLOT_DURATION
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    slug = f"btc-updown-5m-{start_ts}"
    return {
        "slot_start_dt": start_dt,
        "slot_end_dt": end_dt,
        "slot_start_ts": start_ts,
        "slug": slug,
        "slot_start_str": start_dt.strftime("%H:%M"),
        "slot_end_str": end_dt.strftime("%H:%M"),
        "slot_start_full": start_dt.strftime("%Y-%m-%d %H:%M"),
        "slot_end_full": end_dt.strftime("%Y-%m-%d %H:%M"),
    }


def slot_info_from_ts(start_ts: int) -> dict[str, Any]:
    """Build slot info dict from an arbitrary start timestamp."""
    end_ts = start_ts + SLOT_DURATION
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    slug = f"btc-updown-5m-{start_ts}"
    return {
        "slot_start_dt": start_dt,
        "slot_end_dt": end_dt,
        "slot_start_ts": start_ts,
        "slug": slug,
        "slot_start_str": start_dt.strftime("%H:%M"),
        "slot_end_str": end_dt.strftime("%H:%M"),
        "slot_start_full": start_dt.strftime("%Y-%m-%d %H:%M"),
        "slot_end_full": end_dt.strftime("%Y-%m-%d %H:%M"),
    }


# ---------------------------------------------------------------------------
# CLOB API — best ask price fetcher
# ---------------------------------------------------------------------------

async def get_clob_best_ask(token_id: str, client: httpx.AsyncClient) -> float | None:
    """Fetch the best ask price for a token from the CLOB order book.

    GET https://clob.polymarket.com/book?token_id={token_id}

    The ask side represents what a buyer pays. We return asks[0]["price"]
    which is the lowest ask (best ask) — this matches what Polymarket UI shows.

    Returns float price or None on error / empty book.
    """
    url = f"{cfg.CLOB_HOST}/book"
    params = {"token_id": token_id}
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        log.exception("CLOB /book request failed for token_id=%s", token_id)
        return None

    asks = data.get("asks", [])
    if not asks:
        log.warning("CLOB /book returned empty asks for token_id=%s", token_id)
        return None

    try:
        # asks are sorted DESCENDING (highest price first) — last entry is best (lowest) ask
        best_ask = float(asks[-1]["price"])
        log.debug("CLOB best ask for token_id=%s: %.4f (book range: %.4f–%.4f, %d levels)",
                  token_id, best_ask, float(asks[-1]["price"]), float(asks[0]["price"]), len(asks))
        return best_ask
    except (KeyError, ValueError, IndexError):
        log.exception("Failed to parse CLOB asks for token_id=%s", token_id)
        return None


# ---------------------------------------------------------------------------
# Two-step price fetcher: Gamma (token IDs) + CLOB (real ask prices)
# ---------------------------------------------------------------------------

async def get_slot_prices(slug: str) -> dict[str, Any] | None:
    """Fetch live ask prices & token IDs for a BTC 5-min slot.

    Step 1 — Gamma API: get token IDs for Up and Down outcomes.
    Step 2 — CLOB API: get best ask price for each token (what you actually pay).

    Using CLOB best ask instead of Gamma outcomePrices (mid price) ensures the
    threshold check matches what Polymarket UI displays to buyers.

    Returns dict:
      up_price, down_price, up_token_id, down_token_id
    or None on error / empty response.
    """
    # --- Step 1: Gamma API — get token IDs ---
    gamma_url = f"{cfg.GAMMA_API_HOST}/markets"
    params = {"slug": slug}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(gamma_url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        log.exception("Gamma API request failed for slug=%s", slug)
        return None

    if not data or not isinstance(data, list) or len(data) == 0:
        log.warning("Gamma API returned empty response for slug=%s", slug)
        return None

    market = data[0]

    try:
        outcomes_raw = market["outcomes"]
        token_ids_raw = market["clobTokenIds"]

        # Gamma API may return these fields as JSON-encoded strings
        if isinstance(outcomes_raw, str):
            outcomes_raw = json.loads(outcomes_raw)
        if isinstance(token_ids_raw, str):
            token_ids_raw = json.loads(token_ids_raw)

        up_idx = outcomes_raw.index("Up")
        down_idx = outcomes_raw.index("Down")

        token_ids = [str(t) for t in token_ids_raw]
        up_token_id = token_ids[up_idx]
        down_token_id = token_ids[down_idx]

    except (KeyError, ValueError, IndexError):
        log.exception("Failed to parse Gamma market data for slug=%s", slug)
        return None

    # --- Step 2: CLOB API — get real best ask prices for both tokens ---
    async with httpx.AsyncClient(timeout=15) as client:
        up_ask = await get_clob_best_ask(up_token_id, client)
        down_ask = await get_clob_best_ask(down_token_id, client)

    if up_ask is None or down_ask is None:
        log.error(
            "CLOB ask fetch failed for slug=%s  up_token=%s up_ask=%s  down_token=%s down_ask=%s",
            slug, up_token_id, up_ask, down_token_id, down_ask,
        )
        return None

    log.debug(
        "slug=%s  Up ask=%.4f (token=%s)  Down ask=%.4f (token=%s)",
        slug, up_ask, up_token_id, down_ask, down_token_id,
    )

    return {
        "up_price": up_ask,
        "down_price": down_ask,
        "up_token_id": up_token_id,
        "down_token_id": down_token_id,
    }
