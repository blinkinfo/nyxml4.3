"""Model store — save/load/promote LightGBM models with metadata JSON."""

from __future__ import annotations

import json
import logging
import os
import shutil

import lightgbm as lgb

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _ensure_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _model_path(slot: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{slot}.lgb")


def _meta_path(slot: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{slot}_meta.json")


def save_model(model: lgb.Booster, slot: str, metadata: dict) -> None:
    """Save model to models/model_{slot}.lgb and metadata JSON."""
    _ensure_dir()
    model_path = _model_path(slot)
    meta_path = _meta_path(slot)

    model.save_model(model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("save_model: saved slot=%s path=%s", slot, model_path)


def load_model(slot: str = "current") -> lgb.Booster | None:
    """Load and return model. Returns None if file doesn't exist."""
    path = _model_path(slot)
    if not os.path.exists(path):
        log.debug("load_model: no model file at %s", path)
        return None
    try:
        model = lgb.Booster(model_file=path)
        log.info("load_model: loaded slot=%s", slot)
        return model
    except Exception as e:
        log.error("load_model: failed to load %s: %s", path, e)
        return None


def load_metadata(slot: str = "current") -> dict | None:
    """Load and return metadata JSON. Returns None if not found."""
    path = _meta_path(slot)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.error("load_metadata: failed to load %s: %s", path, e)
        return None


def promote_candidate() -> None:
    """Copy candidate model to current (overwrites current)."""
    _ensure_dir()
    src_model = _model_path("candidate")
    src_meta = _meta_path("candidate")
    dst_model = _model_path("current")
    dst_meta = _meta_path("current")

    if not os.path.exists(src_model):
        raise FileNotFoundError(f"Candidate model not found: {src_model}")

    shutil.copy2(src_model, dst_model)
    if os.path.exists(src_meta):
        shutil.copy2(src_meta, dst_meta)

    log.info("promote_candidate: copied candidate -> current")


def has_model(slot: str = "current") -> bool:
    """Return True if model file exists for the given slot."""
    return os.path.exists(_model_path(slot))


async def save_model_to_db(model: lgb.Booster, slot: str, metadata: dict) -> None:
    """Serialize model to bytes and upsert into model_blobs table in SQLite."""
    import tempfile, os, json
    import aiosqlite
    import config as cfg
    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save_model(tmp_path)
        with open(tmp_path, "rb") as f:
            blob = f.read()
    finally:
        os.unlink(tmp_path)
    meta_json = json.dumps(metadata)
    async with aiosqlite.connect(cfg.DB_PATH) as db:
        await db.execute("""
            INSERT INTO model_blobs (slot, blob, metadata)
            VALUES (?, ?, ?)
            ON CONFLICT(slot) DO UPDATE SET
                blob=excluded.blob,
                metadata=excluded.metadata,
                updated_at=CURRENT_TIMESTAMP
        """, (slot, blob, meta_json))
        await db.commit()
    log.info("save_model_to_db: saved slot=%s (%d bytes)", slot, len(blob))


async def load_model_from_db(slot: str = "current") -> "lgb.Booster | None":
    """Load model blob from SQLite and write to temp disk path for LightGBM to load."""
    import tempfile, os
    import aiosqlite
    import config as cfg
    async with aiosqlite.connect(cfg.DB_PATH) as db:
        cursor = await db.execute(
            "SELECT blob FROM model_blobs WHERE slot = ?", (slot,)
        )
        row = await cursor.fetchone()
    if not row:
        log.info("load_model_from_db: no blob found for slot=%s", slot)
        return None
    blob = row[0]
    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        model = lgb.Booster(model_file=tmp_path)
        log.info("load_model_from_db: loaded slot=%s (%d bytes)", slot, len(blob))
        return model
    except Exception as e:
        log.error("load_model_from_db: failed to load slot=%s: %s", slot, e)
        return None
    finally:
        os.unlink(tmp_path)
