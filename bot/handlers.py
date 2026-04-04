"""Telegram command and callback-query handlers."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any

import openpyxl
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config as cfg
from bot.formatters import (
    format_demo_recent_trades,
    format_demo_stats,
    format_error_alert,
    format_help,
    format_pattern_stats,
    format_recent_signals,
    format_recent_trades,
    format_redeem_preview,
    format_redeem_results,
    format_redemption_history,
    format_signal_stats,
    format_status,
    format_trade_stats,
)
from bot.keyboards import (
    back_to_menu,
    download_keyboard,
    main_menu,
    pattern_filter_row,
    pattern_keyboard,
    redeem_confirm_keyboard,
    redeem_done_keyboard,
    settings_keyboard,
    signal_filter_row,
    trade_filter_row,
)
from bot.middleware import auth_check
from db import queries
from polymarket import account as pm_account

log = logging.getLogger(__name__)

# Set at startup by main.py
_start_time: datetime = datetime.now(timezone.utc)
_poly_client: Any = None


def set_poly_client(client: Any) -> None:
    global _poly_client
    _poly_client = client


def set_start_time() -> None:
    global _start_time
    _start_time = datetime.now(timezone.utc)


def _uptime() -> str:
    delta = datetime.now(timezone.utc) - _start_time
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


# ---------------------------------------------------------------------------
# Safe edit helper — silently ignores 'Message is not modified' errors
# ---------------------------------------------------------------------------

async def _safe_edit(query, text, reply_markup=None, parse_mode="HTML"):
    """Edit a message, silently ignoring 'not modified' errors."""
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            pass  # Content unchanged — not an error
        else:
            raise


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@auth_check
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "\U0001f916 <b>Welcome to AutoPoly!</b>\n\n"
        "BTC Up/Down 5-min trading bot for Polymarket.\n"
        "Select an option below:"
    )
    await update.message.reply_text(text, reply_markup=main_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@auth_check
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    connected = False
    balance = None
    positions = []
    if _poly_client:
        connected = await pm_account.get_connection_status(_poly_client)
        balance = await pm_account.get_balance(_poly_client)
        positions = await pm_account.get_open_positions(_poly_client)

    autotrade = await queries.is_autotrade_enabled()
    auto_redeem = await queries.is_auto_redeem_enabled()
    trade_amount = await queries.get_trade_amount()
    last_sig = await queries.get_last_signal()
    last_sig_str = None
    if last_sig:
        ss = last_sig["slot_start"].split(" ")[-1] if " " in last_sig["slot_start"] else last_sig["slot_start"]
        last_sig_str = f"{ss} UTC ({last_sig['side']})"

    demo_trade = await queries.is_demo_trade_enabled()
    demo_bankroll = await queries.get_demo_bankroll() if demo_trade else None
    text = format_status(
        connected=connected,
        balance=balance,
        autotrade=autotrade,
        trade_amount=trade_amount,
        open_positions=len(positions),
        uptime_str=_uptime(),
        last_signal=last_sig_str,
        auto_redeem=auto_redeem,
        demo_trade_enabled=demo_trade,
        demo_bankroll=demo_bankroll,
    )
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        target = update.message
        if target is None:
            return
        await target.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /signals
# ---------------------------------------------------------------------------

async def _render_signals(update: Update, limit: int | None, active: str) -> None:
    stats = await queries.get_signal_stats(limit=limit)
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_signal_stats(stats, label)
    recent = await queries.get_recent_signals(10)
    text += format_recent_signals(recent)
    kb = signal_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_signals(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# /trades
# ---------------------------------------------------------------------------

async def _render_trades(update: Update, limit: int | None, active: str) -> None:
    stats = await queries.get_trade_stats(limit=limit)
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_trade_stats(stats, label)
    recent = await queries.get_recent_trades(10)
    text += format_recent_trades(recent)
    kb = trade_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_trades(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# /settings
# ---------------------------------------------------------------------------

@auth_check
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    autotrade = await queries.is_autotrade_enabled()
    auto_redeem = await queries.is_auto_redeem_enabled()
    trade_amount = await queries.get_trade_amount()
    demo_trade = await queries.is_demo_trade_enabled()
    demo_bankroll = await queries.get_demo_bankroll()
    text = "\u2699\ufe0f <b>Settings</b>\n\nTap a button to change:"
    kb = settings_keyboard(autotrade, trade_amount, auto_redeem, demo_trade, demo_bankroll)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

@auth_check
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = format_help()
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        await update.message.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /redeem — manual redemption (dry-run preview then confirm)
# ---------------------------------------------------------------------------

@auth_check
async def cmd_redeem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Step 1: dry-run scan — show what would be redeemed, await confirmation."""
    from core.redeemer import scan_and_redeem

    wallet = cfg.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        text = "\u274c <b>Redeem Error</b>\n\nPOLYMARKET_FUNDER_ADDRESS is not configured."
        if update.callback_query:
            await update.callback_query.answer()
            await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
        else:
            await update.message.reply_text(text, parse_mode="HTML", reply_markup=back_to_menu())
        return

    # Show scanning message first
    scanning_text = "\U0001f50d <b>Scanning wallet for redeemable positions...</b>"
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, scanning_text)
        sent = None  # edits are done via callback_query
    else:
        sent = await update.message.reply_text(scanning_text, parse_mode="HTML")

    try:
        results = await scan_and_redeem(wallet, dry_run=True)
    except Exception:
        log.exception("cmd_redeem: scan_and_redeem raised unexpectedly")
        error_text = "\u274c <b>Scan failed</b>\n\nCould not fetch positions. Please try again."
        if update.callback_query:
            await _safe_edit(update.callback_query, error_text, reply_markup=back_to_menu())
        else:
            await sent.edit_text(error_text, parse_mode="HTML", reply_markup=back_to_menu())
        return

    # Store dry-run results in user_data for the confirm step
    context.user_data["redeem_preview"] = results

    text = format_redeem_preview(results)
    kb = redeem_confirm_keyboard() if results else back_to_menu()

    if update.callback_query:
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await sent.edit_text(text, parse_mode="HTML", reply_markup=kb)


# ---------------------------------------------------------------------------
# /redemptions — history dashboard
# ---------------------------------------------------------------------------

@auth_check
async def cmd_redemptions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = await queries.get_redemption_stats()
    recent = await queries.get_recent_redemptions(10)
    text = format_redemption_history(stats, recent)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        await update.message.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# Download handlers
# ---------------------------------------------------------------------------

@auth_check
async def cmd_download_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export all signals as a CSV file (filter_blocked column removed — always was 0)."""
    query = update.callback_query
    await query.answer("Preparing CSV...")
    rows = await queries.get_all_signals_for_export()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "slot_start", "side", "entry_price", "is_win", "pattern"])
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    await query.message.reply_document(
        document=io.BytesIO(buf.getvalue().encode()),
        filename="signals.csv",
        caption="\U0001f4e5 All signals export (CSV)",
    )


@auth_check
async def cmd_download_excel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export all signals as an Excel file (filter_blocked column removed — always was 0)."""
    query = update.callback_query
    await query.answer("Preparing Excel...")
    rows = await queries.get_all_signals_for_export()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Signals"
    ws.append(["id", "slot_start", "side", "entry_price", "is_win", "pattern"])
    for r in rows:
        ws.append([r["id"], r["slot_start"], r["side"], r["entry_price"], r["is_win"], r.get("pattern", "")])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    await query.message.reply_document(
        document=buf,
        filename="signals.xlsx",
        caption="\U0001f4e5 All signals export (Excel)",
    )



# ---------------------------------------------------------------------------
# /patterns — per-pattern performance dashboard
# ---------------------------------------------------------------------------

@auth_check
async def cmd_patterns(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = await queries.get_pattern_stats()
    text = format_pattern_stats(rows)
    kb = pattern_keyboard()
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_download_pattern_excel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export per-pattern stats as an Excel file."""
    query = update.callback_query
    await query.answer("Preparing Excel...")
    rows = await queries.get_pattern_stats_for_export()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Patterns"
    ws.append([
        "Pattern", "Total Trades", "Wins", "Losses",
        "Win%", "W/L Ratio", "Deployed USDC", "Net PnL", "ROI%", "Last Seen",
    ])
    for r in rows:
        ws.append([
            r["pattern"], r["total_trades"], r["wins"], r["losses"],
            r["win_pct"],
            r["wl_ratio"] if r["wl_ratio"] != float("inf") else "inf",
            r["total_deployed"], r["net_pnl"], r["roi_pct"], r["last_seen"],
        ])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    await query.message.reply_document(
        document=buf,
        filename="pattern_performance.xlsx",
        caption="\U0001f4e5 Per-pattern stats export (Excel)",
    )


# ---------------------------------------------------------------------------
# Callback query router
# ---------------------------------------------------------------------------

@auth_check
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data

    if data == "cmd_menu":
        await query.answer()
        text = "\U0001f916 <b>AutoPoly Menu</b>\n\nSelect an option:"
        await _safe_edit(query, text, reply_markup=main_menu())

    elif data == "cmd_status":
        await cmd_status(update, context)

    elif data == "cmd_signals":
        await _render_signals(update, limit=None, active="all")

    elif data == "cmd_trades":
        await _render_trades(update, limit=None, active="all")

    elif data == "cmd_settings":
        await cmd_settings(update, context)

    elif data == "cmd_help":
        await cmd_help(update, context)

    elif data == "cmd_redeem":
        await cmd_redeem(update, context)

    elif data == "cmd_redemptions":
        await cmd_redemptions(update, context)

    # Signal filters
    elif data == "signals_10":
        await _render_signals(update, limit=10, active="10")
    elif data == "signals_50":
        await _render_signals(update, limit=50, active="50")
    elif data == "signals_all":
        await _render_signals(update, limit=None, active="all")

    # Trade filters
    elif data == "trades_10":
        await _render_trades(update, limit=10, active="10")
    elif data == "trades_50":
        await _render_trades(update, limit=50, active="50")
    elif data == "trades_all":
        await _render_trades(update, limit=None, active="all")

    # Settings toggles
    elif data == "toggle_autotrade":
        current = await queries.is_autotrade_enabled()
        await queries.set_setting("autotrade_enabled", "false" if current else "true")
        await cmd_settings(update, context)

    elif data == "toggle_auto_redeem":
        current = await queries.is_auto_redeem_enabled()
        await queries.set_setting("auto_redeem_enabled", "false" if current else "true")
        new_state = "ON" if not current else "OFF"
        await query.answer(f"Auto-Redeem {new_state}")
        await cmd_settings(update, context)

    elif data == "change_amount":
        await query.answer()
        await _safe_edit(
            query,
            "\U0001f4b5 <b>Set Trade Amount</b>\n\n"
            "Type the new amount in USDC (e.g. <code>2.50</code>):",
        )
        context.user_data["awaiting_amount"] = True

    elif data == "download_csv":
        await cmd_download_csv(update, context)

    elif data == "download_xlsx":
        await cmd_download_excel(update, context)

    # Redeem confirm / cancel
    elif data == "redeem_confirm":
        await _handle_redeem_confirm(update, context)

    elif data == "redeem_cancel":
        context.user_data.pop("redeem_preview", None)
        await query.answer("Cancelled.")
        await _safe_edit(
            query,
            "\u274c Redemption cancelled.",
            reply_markup=back_to_menu(),
        )

    elif data == "toggle_demo_trade":
        current = await queries.is_demo_trade_enabled()
        await queries.set_setting("demo_trade_enabled", "false" if current else "true")
        new_state = "OFF" if current else "ON"
        await query.answer(f"Demo Trade {new_state}")
        await cmd_settings(update, context)

    elif data == "set_demo_bankroll":
        await query.answer()
        demo_bankroll = await queries.get_demo_bankroll()
        await _safe_edit(
            query,
            f"\U0001f4b0 <b>Set Demo Bankroll</b>\n\n"
            f"Current balance: <b>${demo_bankroll:.2f}</b>\n\n"
            "Type the new bankroll amount in USDC (e.g. <code>500.00</code>):",
        )
        context.user_data["awaiting_demo_bankroll"] = True

    elif data == "reset_demo_bankroll":
        await queries.reset_demo_bankroll(1000.00)
        await query.answer("Demo bankroll reset to $1000.00")
        await cmd_settings(update, context)

    elif data == "cmd_demo":
        await _render_demo_stats(update, active="all")

    elif data == "demo_10":
        await _render_demo_stats(update, limit=10, active="10")

    elif data == "demo_50":
        await _render_demo_stats(update, limit=50, active="50")

    elif data == "demo_all":
        await _render_demo_stats(update, limit=None, active="all")

    elif data == "cmd_patterns":
        await cmd_patterns(update, context)

    elif data == "download_pattern_xlsx":
        await cmd_download_pattern_excel(update, context)

    else:
        await query.answer("Unknown action")


async def _handle_redeem_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute redemptions after user confirmed the dry-run preview."""
    from core.redeemer import redeem_position

    query = update.callback_query
    await query.answer("Executing redemptions...")

    # Retrieve and immediately clear the stored preview
    preview = context.user_data.pop("redeem_preview", None)

    if not preview:
        await _safe_edit(
            query,
            "\u274c <b>Nothing to redeem</b>\n\nThe preview has expired or no positions were found. "
            "Run /redeem again to rescan.",
            reply_markup=back_to_menu(),
        )
        return

    await _safe_edit(
        query,
        f"\u23f3 <b>Executing {len(preview)} redemption(s) on-chain...</b>\n\nThis may take up to 2 minutes.",
    )

    wallet = cfg.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        await _safe_edit(
            query,
            "\u274c POLYMARKET_FUNDER_ADDRESS not configured.",
            reply_markup=back_to_menu(),
        )
        return

    results: list[dict] = []
    for pos in preview:
        result = await redeem_position(pos["condition_id"])
        merged = {**pos, **result, "dry_run": False}
        results.append(merged)

        # Persist each result to DB immediately (even if failed)
        try:
            await queries.insert_redemption(
                condition_id=pos["condition_id"],
                outcome_index=pos["outcome_index"],
                size=pos["size"],
                title=pos.get("title"),
                tx_hash=result.get("tx_hash"),
                status="success" if result.get("success") else "failed",
                error=result.get("error"),
                gas_used=result.get("gas_used"),
                dry_run=False,
            )
        except Exception:
            log.exception("Failed to persist redemption record for condition=%s", pos.get("condition_id"))

    text = format_redeem_results(results)
    await _safe_edit(query, text, reply_markup=redeem_done_keyboard())


# ---------------------------------------------------------------------------
# Text handler (for trade amount input)
# ---------------------------------------------------------------------------

@auth_check
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # -- Demo bankroll input -------------------------------------------------------
    if context.user_data.get("awaiting_demo_bankroll"):
        context.user_data["awaiting_demo_bankroll"] = False
        raw = update.message.text.strip().replace("$", "")
        try:
            amount = float(raw)
            if amount < 0:
                raise ValueError("negative")
        except ValueError:
            await update.message.reply_text(
                "\u274c Invalid amount. Please enter a non-negative number (e.g. 500.00)."
            )
            return
        amount = round(amount, 2)
        await queries.set_demo_bankroll(amount)
        await update.message.reply_text(
            f"\u2705 Demo bankroll set to <b>${amount:.2f}</b>",
            parse_mode="HTML",
        )
        # Refresh settings panel
        autotrade = await queries.is_autotrade_enabled()
        auto_redeem = await queries.is_auto_redeem_enabled()
        demo_trade = await queries.is_demo_trade_enabled()
        kb = settings_keyboard(autotrade, await queries.get_trade_amount(), auto_redeem, demo_trade, amount)
        await update.message.reply_text(
            "\u2699\ufe0f <b>Settings</b>",
            reply_markup=kb,
            parse_mode="HTML",
        )
        return

    # -- Trade amount input --------------------------------------------------------
    if not context.user_data.get("awaiting_amount"):
        return

    context.user_data["awaiting_amount"] = False
    raw = update.message.text.strip().replace("$", "")
    try:
        amount = float(raw)
        if amount <= 0:
            raise ValueError("non-positive")
    except ValueError:
        await update.message.reply_text(
            "\u274c Invalid amount. Please enter a positive number (e.g. 2.50)."
        )
        return

    amount = round(amount, 2)
    await queries.set_setting("trade_amount_usdc", str(amount))
    await update.message.reply_text(
        f"\u2705 Trade amount updated to <b>${amount:.2f}</b>",
        parse_mode="HTML",
    )
    # Show settings panel again
    autotrade = await queries.is_autotrade_enabled()
    auto_redeem = await queries.is_auto_redeem_enabled()
    demo_trade = await queries.is_demo_trade_enabled()
    demo_bankroll = await queries.get_demo_bankroll()
    kb = settings_keyboard(autotrade, amount, auto_redeem, demo_trade, demo_bankroll)
    await update.message.reply_text(
        "\u2699\ufe0f <b>Settings</b>",
        reply_markup=kb,
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /demo — Demo trade performance dashboard
# ---------------------------------------------------------------------------

async def _render_demo_stats(update: Update, limit: int | None = None, active: str = "all") -> None:
    from bot.keyboards import demo_filter_row
    stats = await queries.get_demo_trade_stats(limit=limit)
    bankroll = await queries.get_demo_bankroll()
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_demo_stats(stats, bankroll, label)
    recent = await queries.get_recent_demo_trades(10)
    text += format_demo_recent_trades(recent)
    kb = demo_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_demo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_demo_stats(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# Register all handlers
# ---------------------------------------------------------------------------

def register(application) -> None:
    """Attach all command and callback handlers to the Telegram Application."""
    application.add_handler(CommandHandler("start",       cmd_start))
    application.add_handler(CommandHandler("status",      cmd_status))
    application.add_handler(CommandHandler("signals",     cmd_signals))
    application.add_handler(CommandHandler("trades",      cmd_trades))
    application.add_handler(CommandHandler("settings",    cmd_settings))
    application.add_handler(CommandHandler("help",        cmd_help))
    application.add_handler(CommandHandler("redeem",      cmd_redeem))
    application.add_handler(CommandHandler("redemptions", cmd_redemptions))
    application.add_handler(CommandHandler("demo",        cmd_demo))
    application.add_handler(CommandHandler("patterns",   cmd_patterns))
    application.add_handler(CallbackQueryHandler(callback_router))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    async def _error_handler(update, context):
        import traceback
        err_text = "".join(traceback.format_exception(type(context.error), context.error, context.error.__traceback__))
        log.error("Unhandled Telegram error:\n%s", err_text)
        try:
            if cfg.TELEGRAM_CHAT_ID:
                short = err_text[-800:] if len(err_text) > 800 else err_text
                await context.bot.send_message(
                    chat_id=int(cfg.TELEGRAM_CHAT_ID),
                    text=f"&#x26A0;&#xFE0F; <b>Unhandled Bot Error</b>\n<pre>{short}</pre>",
                    parse_mode="HTML",
                )
        except Exception:
            log.exception("Failed to send error notification to Telegram")

    application.add_error_handler(_error_handler)
