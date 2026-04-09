"""ML strategy using a trained LightGBM model for BTC/USDT 5-min binary prediction.

Returns the IDENTICAL signal dict schema as PatternStrategy.
Uses get_next_slot_info() + get_slot_prices() exactly as PatternStrategy does.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

import numpy as np

from core.strategies.base import BaseStrategy
from ml import data_fetcher
from ml import features as feat_eng
from ml import model_store
from db import queries
from polymarket.markets import get_next_slot_info, get_slot_prices
import config as cfg

log = logging.getLogger(__name__)

FEATURE_COLS = feat_eng.FEATURE_COLS  # 22 features in exact order

# Module-level reload flag so cmd_promote_model can signal a reload
_RELOAD_REQUESTED = False


def request_model_reload() -> None:
    """Signal that the model should be reloaded on the next check_signal call."""
    global _RELOAD_REQUESTED
    _RELOAD_REQUESTED = True


class MLStrategy(BaseStrategy):
    """LightGBM-based signal strategy. Replaces PatternStrategy as the default."""

    def __init__(self):
        self._model = None
        self._funding_buffer: deque = deque(maxlen=24)
        self._model_slot = "current"
        self._load_model()
        self._seed_funding_buffer()

    def _seed_funding_buffer(self) -> None:
        """Seed the funding buffer with historical data on startup.

        Without seeding, the buffer starts empty and zscore is undefined for the
        first 8 days of operation (24 periods * 8h each). This pre-fills the buffer
        from MEXC historical funding so zscore is valid from the very first inference.
        """
        try:
            history = data_fetcher.fetch_live_funding_history(n_periods=24)
            if history:
                for rate in history:
                    self._funding_buffer.append(rate)
                log.info(
                    "MLStrategy: seeded funding_buffer with %d historical records",
                    len(self._funding_buffer),
                )
            else:
                log.warning("MLStrategy: could not seed funding_buffer — no historical data returned")
        except Exception as exc:
            log.warning("MLStrategy: funding_buffer seed failed: %s", exc)

    def _load_model(self) -> None:
        """Load the current model — use preloaded model if available, else load from disk."""
        global _RELOAD_REQUESTED, _PRELOADED_MODEL
        if _PRELOADED_MODEL is not None:
            self._model = _PRELOADED_MODEL
            _PRELOADED_MODEL = None
            _RELOAD_REQUESTED = False
            log.info("MLStrategy: model set from preloaded instance")
        else:
            self._model = model_store.load_model("current")
            _RELOAD_REQUESTED = False
            if self._model is None:
                log.warning("MLStrategy: no trained model found at models/model_current.lgb")
            else:
                log.info("MLStrategy: model loaded successfully")

    async def _get_threshold(self) -> float:
        """Read threshold from ml_config table, fall back to cfg default."""
        try:
            val = await queries.get_ml_threshold()
            return val
        except Exception:
            pass
        # Legacy fallback: check settings table
        try:
            val = await queries.get_setting("ml_threshold")
            if val is not None:
                return float(val)
        except Exception:
            pass
        return cfg.ML_DEFAULT_THRESHOLD

    async def check_signal(self) -> dict[str, Any] | None:
        """Generate an ML-based signal for slot N+1.

        Called at T-85s before the current slot ends.

        Returns the same signal dict schema as PatternStrategy:
          - skipped=True dict when no trade (below threshold or data issues)
          - skipped=False dict with full trade fields when model fires
          - None on hard failure
        """
        global _RELOAD_REQUESTED

        # Reload model if requested (e.g., after promote_model)
        if _RELOAD_REQUESTED:
            self._load_model()

        # Get next slot info — identical pattern to PatternStrategy
        slot_n1 = get_next_slot_info()

        # Standard base fields used in all return dicts (matches PatternStrategy exactly)
        base_fields: dict[str, Any] = {
            "skipped": True,
            "pattern": None,
            "candles_used": 50,
            "slot_n1_start_full": slot_n1["slot_start_full"],
            "slot_n1_end_full":   slot_n1["slot_end_full"],
            "slot_n1_start_str":  slot_n1["slot_start_str"],
            "slot_n1_end_str":    slot_n1["slot_end_str"],
            "slot_n1_ts":         slot_n1["slot_start_ts"],
            "slot_n1_slug":       slot_n1["slug"],
        }

        if self._model is None:
            self._load_model()
            if self._model is None:
                log.error("MLStrategy: no model loaded, skipping slot %s", slot_n1["slug"])
                return {**base_fields, "reason": "No model loaded"}

        try:
            # Fetch live data in parallel using executor (blocking ccxt calls)
            loop = asyncio.get_event_loop()
            df5, df15, df1h, funding_rate, cvd_live = await asyncio.gather(
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_5m(50)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_15m(30)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_1h(20)),
                loop.run_in_executor(None, data_fetcher.fetch_live_funding),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_cvd(25)),
            )

            # Update funding rolling buffer
            if funding_rate is not None:
                self._funding_buffer.append(funding_rate)

            # Build feature row
            feature_row = feat_eng.build_live_features(
                df5, df15, df1h, funding_rate, self._funding_buffer, cvd_live
            )
            if feature_row is None:
                log.warning("MLStrategy: insufficient data for features, skipping")
                return {**base_fields, "reason": "Insufficient data for features"}

            # Model inference
            prob = float(self._model.predict(feature_row)[0])
            threshold = await self._get_threshold()

            # Determine direction per BLUEPRINT Section 11.1 Step 5:
            #   UP   if prob >= threshold          (class 1: price goes up)
            #   DOWN if (1 - prob) >= threshold    (class 0: price goes down, high confidence)
            #   No trade if neither condition is met
            up_confidence   = prob
            down_confidence = 1.0 - prob

            if up_confidence >= threshold:
                side = "Up"
            elif down_confidence >= threshold:
                side = "Down"
            else:
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}<{threshold:.3f}",
                    "reason": f"Below threshold (p={prob:.4f}, 1-p={down_confidence:.4f})",
                }

            log.info(
                "MLStrategy: side=%s prob=%.4f 1-prob=%.4f threshold=%.3f slot=%s",
                side, prob, down_confidence, threshold, slot_n1["slug"],
            )

            # Fetch Polymarket prices — identical to PatternStrategy
            prices = await get_slot_prices(slot_n1["slug"])
            if prices is None:
                log.warning(
                    "MLStrategy: no Polymarket prices for slug=%s, skipping", slot_n1["slug"]
                )
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}",
                    "reason": "Market data unavailable",
                }

            entry_price    = prices["up_price"]    if side == "Up" else prices["down_price"]
            opposite_price = prices["down_price"]  if side == "Up" else prices["up_price"]
            token_id       = prices["up_token_id"] if side == "Up" else prices["down_token_id"]

            return {
                **base_fields,
                "skipped":        False,
                "side":           side,
                "entry_price":    entry_price,
                "opposite_price": opposite_price,
                "token_id":       token_id,
                "pattern":        f"p={prob:.4f}",
            }

        except Exception as exc:
            log.exception("MLStrategy.check_signal failed: %s", exc)
            return None
