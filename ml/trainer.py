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

# ---------------------------------------------------------------------------
# Deployment gate — Blueprint Rule 10
# ---------------------------------------------------------------------------

class DeploymentBlockedError(Exception):
    """Raised when the trained model fails to meet the minimum test-set WR.

    Blueprint Rule 10: ALWAYS validate that test set WR >= 59% before
    deploying. If a new retrain fails to hit 59% on test, do not deploy.
    """

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
    "n_jobs": 1,  # 1 avoids multiprocess overhead on single-vCPU Railway instances
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# Walk-forward validation constants
# ---------------------------------------------------------------------------
WF_FOLDS = 5          # number of walk-forward folds
WF_INITIAL_PCT = 0.60 # fraction of data used as train+val in fold 1
# Each fold expands the train+val window by WF_STEP_PCT.
# WF_STEP_PCT = (1.0 - WF_INITIAL_PCT) / WF_FOLDS = 0.08
WF_STEP_PCT = (1.0 - WF_INITIAL_PCT) / WF_FOLDS


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
      - If any threshold achieves WR >= 0.59: pick the one that maximizes
        (WR - 0.5) * trades_per_day  (edge-weighted activity score).
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

    # First pass: find candidates with WR >= 0.59
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
        # Pick maximum daily edge = (WR - 0.5) * trades_per_day among WR >= 0.59 candidates.
        # This balances win-rate quality against trade frequency rather than
        # blindly picking the highest WR or the most trades.
        # Example: 59.5% WR / 5 tpd -> edge = (0.595-0.5)*5 = 0.475, which beats
        # 65% WR / 2 tpd -> edge = (0.65-0.5)*2 = 0.30, even though 65% WR > 59.5%.
        # The metric correctly favours the more active threshold here.
        best = max(candidates_above, key=lambda x: (x[1] - 0.5) * x[3])
        best_threshold, best_wr, best_trades, best_trades_per_day = best
        log.info(
            "sweep_threshold: WR>=0.59 candidates=%d, best thresh=%.3f WR=%.4f "
            "trades/day=%.1f edge/day=%.4f",
            len(candidates_above), best_threshold, best_wr, best_trades_per_day,
            (best_wr - 0.5) * best_trades_per_day,
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
# Walk-forward validation (evaluation only — does NOT save any model)
# ---------------------------------------------------------------------------

def walk_forward_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """Run 5-fold walk-forward validation on the full dataset.

    Each fold uses an expanding training window:
      Fold 1: train+val = first 60%, test = next 8%  (rows [0:60%], test [60%:68%])
      Fold 2: train+val = first 68%, test = next 8%
      Fold 3: train+val = first 76%, test = next 8%
      Fold 4: train+val = first 84%, test = next 8%
      Fold 5: train+val = first 92%, test = last 8%  (remainder goes to last fold)

    Within each fold, the train+val block is further split 80/20 (train/val)
    and the same threshold sweep used in the main training path is applied to
    the val set. The resulting threshold is then used to evaluate WR on the
    held-out test window, which was never seen during training or threshold
    selection.

    This is PURELY for reporting. No model is saved here.

    Returns:
        dict with keys:
          fold_results  -- list of per-fold dicts (fold, train_size, val_size,
                           test_size, threshold, val_wr, test_wr, test_trades)
          avg_wr        -- mean test WR across all folds
          std_wr        -- std dev of test WR across all folds
          min_wr        -- minimum per-fold test WR
          max_wr        -- maximum per-fold test WR
    """
    n = len(y)
    fold_results = []

    log.info(
        "walk_forward_validation: starting %d-fold walk-forward on n=%d samples "
        "(initial_pct=%.0f%%, step_pct=%.0f%%)",
        WF_FOLDS, n, WF_INITIAL_PCT * 100, WF_STEP_PCT * 100,
    )

    for fold_idx in range(WF_FOLDS):
        # ---------------------------------------------------------------------------
        # Compute slice boundaries — all in absolute row indices.
        # train+val window expands by WF_STEP_PCT each fold.
        # test window is always the NEXT sequential chunk after train+val.
        # No row ever appears in both train+val and test for the same fold.
        # ---------------------------------------------------------------------------
        trainval_end = int(n * (WF_INITIAL_PCT + fold_idx * WF_STEP_PCT))
        if fold_idx < WF_FOLDS - 1:
            test_end = int(n * (WF_INITIAL_PCT + (fold_idx + 1) * WF_STEP_PCT))
        else:
            test_end = n  # last fold uses all remaining rows

        test_start = trainval_end  # test immediately follows train+val — no gap, no overlap

        # 80/20 split of the train+val block (same ratio as main training path)
        fold_val_start = int(trainval_end * 0.80)

        # Slice arrays — time order is strictly preserved (no shuffling)
        X_fold_train = X[:fold_val_start]
        y_fold_train = y[:fold_val_start]
        X_fold_val = X[fold_val_start:trainval_end]
        y_fold_val = y[fold_val_start:trainval_end]
        X_fold_test = X[test_start:test_end]
        y_fold_test = y[test_start:test_end]

        fold_num = fold_idx + 1
        log.info(
            "walk_forward fold %d/%d: train=[0:%d] val=[%d:%d] test=[%d:%d]",
            fold_num, WF_FOLDS,
            fold_val_start, fold_val_start, trainval_end,
            test_start, test_end,
        )

        if len(X_fold_train) < 50 or len(X_fold_val) < 10 or len(X_fold_test) < 10:
            log.warning(
                "walk_forward fold %d: insufficient samples "
                "(train=%d val=%d test=%d) — skipping fold",
                fold_num, len(X_fold_train), len(X_fold_val), len(X_fold_test),
            )
            continue

        # Train a fold model (evaluation only — not saved to disk)
        fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train, feature_name=FEATURE_COLS)
        fold_val_data = lgb.Dataset(
            X_fold_val, label=y_fold_val, feature_name=FEATURE_COLS, reference=fold_train_data
        )
        fold_callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),  # suppress per-iteration logging for fold models
        ]
        fold_model = lgb.train(
            LGBM_PARAMS,
            fold_train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[fold_val_data],
            callbacks=fold_callbacks,
        )

        # Threshold sweep on fold val set (same logic as main training path)
        fold_val_probs = fold_model.predict(X_fold_val)
        fold_threshold, fold_val_wr, fold_val_tpd = sweep_threshold(fold_val_probs, y_fold_val)

        # Evaluate on strictly held-out test window using threshold from val
        fold_test_probs = fold_model.predict(X_fold_test)
        fold_test_metrics = evaluate_at_threshold(fold_test_probs, y_fold_test, fold_threshold)

        log.info(
            "walk_forward fold %d/%d: threshold=%.3f val_wr=%.4f "
            "test_wr=%.4f test_trades=%d",
            fold_num, WF_FOLDS,
            fold_threshold, fold_val_wr,
            fold_test_metrics["wr"], fold_test_metrics["trades"],
        )

        fold_results.append({
            "fold": fold_num,
            "train_size": fold_val_start,
            "val_size": trainval_end - fold_val_start,
            "test_size": test_end - test_start,
            "threshold": fold_threshold,
            "val_wr": fold_val_wr,
            "test_wr": fold_test_metrics["wr"],
            "test_trades": fold_test_metrics["trades"],
            "test_trades_per_day": fold_test_metrics["trades_per_day"],
        })

    # Aggregate results
    if fold_results:
        wrs = [r["test_wr"] for r in fold_results]
        avg_wr = float(np.mean(wrs))
        std_wr = float(np.std(wrs))
        min_wr = float(np.min(wrs))
        max_wr = float(np.max(wrs))
    else:
        avg_wr = std_wr = min_wr = max_wr = 0.0

    log.info(
        "walk_forward_validation SUMMARY: folds=%d avg_wr=%.4f std_wr=%.4f "
        "min_wr=%.4f max_wr=%.4f",
        len(fold_results), avg_wr, std_wr, min_wr, max_wr,
    )
    for r in fold_results:
        log.info(
            "  fold %d: train_size=%d val_size=%d test_size=%d "
            "thresh=%.3f val_wr=%.4f test_wr=%.4f test_trades=%d",
            r["fold"], r["train_size"], r["val_size"], r["test_size"],
            r["threshold"], r["val_wr"], r["test_wr"], r["test_trades"],
        )

    return {
        "fold_results": fold_results,
        "avg_wr": avg_wr,
        "std_wr": std_wr,
        "min_wr": min_wr,
        "max_wr": max_wr,
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
        dict with model, threshold, val_wr, val_trades, wf_results, and metadata
    """
    n = len(df_features)
    # Minimum n is derived from the walk-forward fold skip guard:
    # fold 1 train = int(int(n*0.60)*0.80) must be >= 50.
    # Empirically this requires n >= 123; use 130 as a conservative buffer.
    if n < 130:
        raise ValueError(f"Too few samples to train: {n} (minimum 130 required)")

    X = df_features[FEATURE_COLS].values
    y = df_features["target"].values

    # ---------------------------------------------------------------------------
    # Walk-forward validation — purely for evaluation/reporting.
    # Runs BEFORE the final model fit so results are logged early.
    # Does NOT save any model to disk. Does NOT influence the final threshold.
    # ---------------------------------------------------------------------------
    log.info("train: running walk-forward validation (%d folds) before final fit", WF_FOLDS)
    wf_results = walk_forward_validation(X, y)
    log.info(
        "train: walk-forward done — avg_wr=%.4f std_wr=%.4f (across %d folds)",
        wf_results["avg_wr"], wf_results["std_wr"], len(wf_results["fold_results"]),
    )

    # ---------------------------------------------------------------------------
    # Final model: train on ALL data using the same 80/20 val split for early
    # stopping, then sweep threshold on that val set.
    #
    # Time-series split: DO NOT SHUFFLE
    # split_boundary = index where validation ends and test begins (75% of data).
    # val_start      = index where training ends and validation begins (80% of split_boundary).
    # Layout: [0 : val_start] = train, [val_start : split_boundary] = val, [split_boundary :] = test
    # ---------------------------------------------------------------------------
    split_boundary = int(n * 0.75)
    val_start = int(split_boundary * 0.80)

    log.info("train: n=%d train=[0:%d] val=[%d:%d] test=[%d:%d]",
             n, val_start, val_start, split_boundary, split_boundary, n)

    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:split_boundary], y[val_start:split_boundary]
    X_test, y_test = X[split_boundary:], y[split_boundary:]

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

    # ---------------------------------------------------------------------------
    # DOWN threshold sweep — validate independently on the same val set.
    # P(DOWN) = 1 - P(UP).  Labels are inverted: 1 means price actually went DOWN.
    # This is NOT a symmetric assumption — the sweep finds the actual best
    # threshold for the inverted probability and checks whether WR >= 59%.
    # down_enabled=True only when the DOWN side passes the same deployment gate.
    # ---------------------------------------------------------------------------
    down_probs_val = 1.0 - val_probs
    y_val_down = 1 - y_val  # label=1 means price went DOWN
    down_threshold, down_val_wr, down_val_tpd = sweep_threshold(down_probs_val, y_val_down)
    down_enabled = down_val_wr >= 0.59

    log.info(
        "train: DOWN sweep — down_threshold=%.3f down_val_wr=%.4f down_val_tpd=%.1f down_enabled=%s",
        down_threshold, down_val_wr, down_val_tpd, down_enabled,
    )
    if not down_enabled:
        log.warning(
            "train: DOWN side did NOT pass deployment gate (down_val_wr=%.4f < 0.59). "
            "DOWN trades will be disabled for this model.",
            down_val_wr,
        )

    # Evaluate on test set using threshold chosen from val set
    test_probs = model.predict(X_test)
    test_metrics = evaluate_at_threshold(test_probs, y_test, best_threshold)

    # DOWN test set evaluation — confirms DOWN threshold holds on held-out data.
    # If DOWN test WR < 59%, override down_enabled to False regardless of val result.
    down_test_metrics = evaluate_at_threshold(
        1.0 - test_probs,  # P(DOWN) on test set
        1 - y_test,        # DOWN labels on test set
        down_threshold,
    )
    if down_enabled and down_test_metrics["wr"] < 0.59:
        log.warning(
            "train: DOWN passed val gate but FAILED test gate "
            "(down_test_wr=%.4f < 0.59). Disabling DOWN.",
            down_test_metrics["wr"],
        )
        down_enabled = False

    log.info(
        "train: val_wr=%.4f threshold=%.3f | test_wr=%.4f test_trades=%d",
        best_wr, best_threshold, test_metrics["wr"], test_metrics["trades"],
    )
    log.info(
        "train: down_val_wr=%.4f down_threshold=%.3f | down_test_wr=%.4f down_test_trades=%d down_enabled=%s",
        down_val_wr, down_threshold, down_test_metrics["wr"], down_test_metrics["trades"], down_enabled,
    )

    # -----------------------------------------------------------------------
    # Deployment gate — Blueprint Rule 10
    # ALWAYS validate test WR >= 59% before auto-deploying.
    # If the model fails this gate we still save it to the candidate slot
    # so the user can inspect it and decide whether to promote or discard.
    # We return blocked=True so the caller can surface the decision to the
    # user rather than silently keeping or discarding the model.
    # -----------------------------------------------------------------------
    MIN_DEPLOY_WR = 0.59
    blocked = test_metrics["wr"] < MIN_DEPLOY_WR
    if blocked:
        log.warning(
            "DEPLOYMENT BLOCKED: test_wr=%.4f is below minimum %.2f "
            "(Blueprint Rule 10). Model saved to candidate slot — "
            "user must decide whether to promote or discard.",
            test_metrics["wr"], MIN_DEPLOY_WR,
        )

    # Save model and metadata to candidate slot regardless of gate result.
    # The caller decides what to do with a blocked candidate.
    metadata = {
        "train_date": datetime.utcnow().isoformat(),
        # UP side
        "threshold": best_threshold,
        "val_wr": best_wr,
        "val_trades_per_day": best_trades_per_day,
        "test_wr": test_metrics["wr"],
        "test_precision": test_metrics["precision"],
        "test_trades": test_metrics["trades"],
        "test_trades_per_day": test_metrics["trades_per_day"],
        # DOWN side — independently swept and validated
        "down_threshold": down_threshold,
        "down_enabled": down_enabled,
        "down_val_wr": down_val_wr,
        "down_val_tpd": down_val_tpd,
        "down_test_wr": down_test_metrics["wr"],
        "down_test_trades": down_test_metrics["trades"],
        "down_test_tpd": down_test_metrics["trades_per_day"],
        # Walk-forward validation summary
        "wf_avg_wr": wf_results["avg_wr"],
        "wf_std_wr": wf_results["std_wr"],
        "wf_min_wr": wf_results["min_wr"],
        "wf_max_wr": wf_results["max_wr"],
        "wf_folds": len(wf_results["fold_results"]),
        "wf_fold_results": wf_results["fold_results"],
        # Common
        "sample_count": n,
        "train_size": val_start,
        "val_size": split_boundary - val_start,
        "test_size": n - split_boundary,
        "feature_cols": FEATURE_COLS,
        "best_iteration": model.best_iteration,
        "blocked": blocked,
    }
    model_store.save_model(model, slot, metadata)

    return {
        "model": model,
        "threshold": best_threshold,
        "down_threshold": down_threshold,
        "down_enabled": down_enabled,
        "down_val_wr": down_val_wr,
        "down_val_tpd": down_val_tpd,
        "down_test_metrics": down_test_metrics,
        "test_metrics": test_metrics,
        "val_wr": best_wr,
        "val_trades": best_trades_per_day,
        "best_iteration": model.best_iteration,
        "blocked": blocked,
        "wf_results": wf_results,
        "warning_reason": (
            f"Test WR {test_metrics['wr']*100:.2f}% is below the 59% deployment gate "
            f"(Blueprint Rule 10). Candidate saved but NOT auto-promoted."
        ) if blocked else None,
    }
