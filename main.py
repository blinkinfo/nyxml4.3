"""AutoPoly entry point — init DB, start Telegram bot + scheduler."""

from __future__ import annotations

import asyncio
import logging
import sys

from telegram import BotCommand
from telegram.ext import Application

import config as cfg
from bot import handlers
from core.scheduler import recover_unresolved, start_scheduler
from db.models import init_db, migrate_db, cleanup_bad_redemptions
from polymarket.client import PolymarketClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
log = logging.getLogger("autopoly")


def _validate_config() -> bool:
    ok = True
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                 "POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS"):
        if not getattr(cfg, name, None):
            log.error("Missing required env var: %s", name)
            ok = False

    # Warn (but don't block startup) if POLYGON_RPC_URL is missing
    if not getattr(cfg, "POLYGON_RPC_URL", None):
        log.warning(
            "POLYGON_RPC_URL is not set — on-chain redemptions will fail. "
            "Set this env var to enable /redeem and auto-redeem."
        )

    return ok


async def _startup_safe_sanity_check() -> None:
    """Validate Safe/EOA config on startup. Logs warnings, does not block startup."""
    import config as cfg

    sig_type = cfg.POLYMARKET_SIGNATURE_TYPE
    private_key = cfg.POLYMARKET_PRIVATE_KEY
    funder = cfg.POLYMARKET_FUNDER_ADDRESS
    rpc_url = cfg.POLYGON_RPC_URL

    log.info("=== Wallet Sanity Check ===")
    log.info("Signature type: %s", sig_type)

    if not private_key:
        log.warning("POLYMARKET_PRIVATE_KEY not set — skipping sanity check")
        return
    if not funder:
        log.warning("POLYMARKET_FUNDER_ADDRESS not set — skipping sanity check")
        return

    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url)) if rpc_url else None

        from eth_account import Account
        eoa_address = Account.from_key(private_key).address
        log.info("EOA (signer) address:  %s", eoa_address)
        log.info("Safe/proxy address:    %s", funder)

        if eoa_address.lower() == funder.lower():
            if sig_type == 2:
                log.error(
                    "MISCONFIGURATION: EOA == FUNDER but SIGNATURE_TYPE=2. "
                    "With sig type 2 the EOA and Safe must be different addresses. "
                    "Redemptions will fail."
                )
            else:
                log.info("EOA == FUNDER (sig type %s — direct EOA mode, OK)", sig_type)
        else:
            log.info("EOA != FUNDER (sig type %s)", sig_type)
            if sig_type != 2:
                log.warning(
                    "EOA != FUNDER but SIGNATURE_TYPE=%s (not 2). "
                    "This may be a misconfiguration.", sig_type
                )

        # On-chain ownership check (only if RPC available and sig_type==2)
        if sig_type == 2 and w3 and w3.is_connected():
            try:
                from core.redeemer import _SAFE_ABI
                safe = w3.eth.contract(
                    address=Web3.to_checksum_address(funder),
                    abi=_SAFE_ABI,
                )
                owners = safe.functions.getOwners().call()
                threshold = safe.functions.getThreshold().call()
                owners_lower = [o.lower() for o in owners]
                log.info("Safe owners: %s", owners)
                log.info("Safe threshold: %s/%s", threshold, len(owners))
                if eoa_address.lower() in owners_lower:
                    log.info("OK: EOA is a listed owner of the Safe")
                else:
                    log.error(
                        "MISCONFIGURATION: EOA %s is NOT listed as an owner of Safe %s. "
                        "Redemptions will fail (EOA cannot sign Safe transactions).",
                        eoa_address, funder,
                    )
            except Exception as exc:
                log.warning("Could not verify Safe ownership on-chain: %s", exc)
        elif sig_type == 2 and (not w3 or not w3.is_connected()):
            log.warning("POLYGON_RPC_URL not available — skipping on-chain Safe ownership check")

    except Exception as exc:
        log.warning("Startup sanity check failed (non-fatal): %s", exc)

    log.info("=== End Wallet Sanity Check ===")


def main() -> None:
    if not _validate_config():
        sys.exit(1)

    # 1. Init DB synchronously first (create tables, seed defaults)
    asyncio.run(init_db())
    log.info("Database initialised at %s", cfg.DB_PATH)


    # 2. Init Polymarket client (synchronous — derives creds)
    poly_client: PolymarketClient | None = None
    try:
        poly_client = PolymarketClient(cfg)
        log.info("Polymarket client ready.")
    except Exception:
        log.exception("Failed to initialise Polymarket client — trading disabled")

    # 3. Build Telegram Application with post_init hook
    async def post_init(application: Application) -> None:
        """Called after the Application is initialised but before polling starts.

        Order matters:
          1. start_scheduler() first — creates and starts the AsyncIOScheduler,
             setting the module-level SCHEDULER global.
          2. recover_unresolved() second — reads unresolved signals from the DB
             and schedules immediate resolution jobs onto SCHEDULER.  If called
             before start_scheduler(), SCHEDULER is still None and every
             recovery job is silently dropped (the `if SCHEDULER is not None`
             guard in recover_unresolved fires False for every signal).
        """
        await migrate_db()
        cleaned = await cleanup_bad_redemptions()
        if cleaned:
            log.info("Startup: cleaned %d incorrectly recorded redemption row(s)", cleaned)
        await _startup_safe_sanity_check()

        # Load ML model: try DB first, fall back to disk
        from core.strategies import ml_strategy
        from ml import model_store
        loaded = await model_store.load_model_from_db("current")
        if loaded:
            ml_strategy.set_model(loaded)
            log.info("Startup: ML model loaded from DB")
        else:
            disk_model = model_store.load_model("current")
            if disk_model:
                ml_strategy.set_model(disk_model)
                log.info("Startup: ML model loaded from disk (fallback)")
            else:
                log.warning("Startup: No ML model found — signals will be skipped until retrain")

        start_scheduler(application, poly_client)
        await recover_unresolved()

        # Register bot commands so they appear in Telegram menu
        await application.bot.set_my_commands([
            BotCommand("status",      "Portfolio overview & bot health"),
            BotCommand("signals",     "Recent trading signals"),
            BotCommand("trades",      "Open & recent trades"),
            BotCommand("redeem",      "Scan & redeem winning positions"),
            BotCommand("redemptions", "Redemption history"),
            BotCommand("settings",    "View/adjust bot settings"),
            BotCommand("demo",        "Demo trading & virtual bankroll"),
            BotCommand("help",        "Show available commands"),
        ])

    application = (
        Application.builder()
        .token(cfg.TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # 4. Register handlers & inject poly client
    handlers.set_poly_client(poly_client)
    handlers.set_start_time()
    handlers.register(application)

    # 5. Run polling (blocks until stopped)
    #    Both scheduler and bot run in the same async event loop.
    log.info("Starting Telegram bot polling...")
    try:
        application.run_polling(drop_pending_updates=True)
    except Exception as exc:
        log.critical("Telegram bot exited: %s", exc)


if __name__ == "__main__":
    main()
