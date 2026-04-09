"""LightGBM trainer — BLUEPRINT sections 7, 8, 9.

CRITICAL: NO data shuffling (time-series order must be preserved).
Threshold sweep ONLY on validation set, never on test set.
"""

from __future__ import annotations

import logging
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from ml import model_store
from ml.features import FEATURE_COLS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LightGBM hyperparameters — exact blueprint spec
# ---------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": 2,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


# ---------------------------------------------------------------------------
# Threshold sweep (val set only — never test set)
# ---------------------------------------------------------------------------

def sweep_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    lo: float = 0.50,
    hi: float = 0.80,
    step: float = 0.005,
) -> tuple[float, float, float]:
    """Sweep thresholds on val set and select best.

    Selection criteria:
      - If any threshold achieves WR >= 0.59: pick the one with most trades.
      - Otherwise: pick threshold with maximum WR.

    Returns:
      (best_threshold, best_wr, trades_per_day)
      trades_per_day = trades / (len(probs) * 5 / 1440)
    """
    periods_per_day = 1440 / 5  # 5-min candles per day

    best_threshold = lo
    best_wr = 0.0
    best_trades = 0
    best_trades_per_day = 0.0

    # First pass: find candidates with WR >= 0.59 and max trades
    candidates_above = []

    thresh = lo
    while thresh <= hi + 1e-9:
        mask = probs >= thresh
        trades = int(mask.sum())
        if trades > 0:
            wr = float(y_true[mask].mean())
            tpd = trades / (len(probs) * 5 / 1440)
            if wr >= 0.59:
                candidates_above.append((thresh, wr, trades, tpd))
        thresh = round(thresh + step, 4)

    if candidates_above:
        # Pick maximum trades among WR >= 0.59 candidates
        best = max(candidates_above, key=lambda x: x[2])
        best_threshold, best_wr, best_trades, best_trades_per_day = best
        log.info(
            "sweep_threshold: WR>=0.59 candidates=%d, best thresh=%.3f WR=%.4f trades/day=%.1f",
            len(candidates_above), best_threshold, best_wr, best_trades_per_day,
        )
    else:
        # No candidate >= 0.59: pick max WR
        best_wr_val = 0.0
        thresh = lo
        while thresh <= hi + 1e-9:
            mask = probs >= thresh
            trades = int(mask.sum())
            if trades > 0:
                wr = float(y_true[mask].mean())
                tpd = trades / (len(probs) * 5 / 1440)
                if wr > best_wr_val or (wr == best_wr_val and trades > best_trades):
                    best_wr_val = wr
                    best_threshold = thresh
                    best_wr = wr
                    best_trades = trades
                    best_trades_per_day = tpd
            thresh = round(thresh + step, 4)
        log.warning(
            "sweep_threshold: no threshold achieves WR>=0.59, best=%.3f WR=%.4f",
            best_threshold, best_wr,
        )

    return best_threshold, best_wr, best_trades_per_day


# ---------------------------------------------------------------------------
# Evaluate at a single threshold
# ---------------------------------------------------------------------------

def evaluate_at_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate model at a specific threshold.

    Returns dict: wr, precision, trades, trades_per_day, recall, f1
    """
    mask = probs >= threshold
    trades = int(mask.sum())

    if trades == 0:
        return {
            "wr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "trades": 0,
            "trades_per_day": 0.0,
        }

    y_pred = mask.astype(int)
    y_sel = y_true[mask]

    wr = float(y_sel.mean())
    trades_per_day = trades / (len(probs) * 5 / 1440)

    try:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        precision = wr
        recall = 0.0
        f1 = 0.0

    return {
        "wr": wr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trades": trades,
        "trades_per_day": trades_per_day,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(df_features: pd.DataFrame, slot: str = "current") -> dict:
    """Train LightGBM model and save to model store.

    Args:
        df_features: DataFrame with FEATURE_COLS + 'target'. NOT shuffled.
        slot: 'current' or 'candidate'

    Returns:
        dict with model, threshold, test_metrics, val_wr, val_trades
    """
    n = len(df_features)
    if n < 100:
        raise ValueError(f"Too few samples to train: {n}")

    # Time-series split: DO NOT SHUFFLE
    train_end = int(n * 0.75)
    val_start = int(train_end * 0.80)

    log.info("train: n=%d train=[0:%d] val=[%d:%d] test=[%d:%d]",
             n, val_start, val_start, train_end, train_end, n)

    X = df_features[FEATURE_COLS].values
    y = df_features["target"].values

    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:train_end], y[val_start:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    log.info("train: X_train=%s X_val=%s X_test=%s", X_train.shape, X_val.shape, X_test.shape)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    log.info("train: best_iteration=%d", model.best_iteration)

    # Threshold sweep on VALIDATION set only — never test set
    val_probs = model.predict(X_val)
    best_threshold, best_wr, best_trades_per_day = sweep_threshold(val_probs, y_val)

    # Evaluate on test set using threshold chosen from val set
    test_probs = model.predict(X_test)
    test_metrics = evaluate_at_threshold(test_probs, y_test, best_threshold)

    log.info(
        "train: val_wr=%.4f threshold=%.3f | test_wr=%.4f test_trades=%d",
        best_wr, best_threshold, test_metrics["wr"], test_metrics["trades"],
    )

    # Save model and metadata
    metadata = {
        "train_date": datetime.utcnow().isoformat(),
        "threshold": best_threshold,
        "val_wr": best_wr,
        "val_trades_per_day": best_trades_per_day,
        "test_wr": test_metrics["wr"],
        "test_precision": test_metrics["precision"],
        "test_trades": test_metrics["trades"],
        "test_trades_per_day": test_metrics["trades_per_day"],
        "sample_count": n,
        "train_size": val_start,
        "val_size": train_end - val_start,
        "test_size": n - train_end,
        "feature_cols": FEATURE_COLS,
        "best_iteration": model.best_iteration,
    }
    model_store.save_model(model, slot, metadata)

    return {
        "model": model,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "val_wr": best_wr,
        "val_trades": best_trades_per_day,
    }
