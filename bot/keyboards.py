"""Inline keyboard layouts for the Telegram bot."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# ---------------------------------------------------------------------------
# Consistent filter button helper
# ---------------------------------------------------------------------------

def _filter_btn(label: str, callback_data: str, active: str) -> InlineKeyboardButton:
    """Return a filter tab button with ✅ prefix when active."""
    prefix = "✅ " if callback_data.endswith(f"_{active}") else ""
    return InlineKeyboardButton(f"{prefix}{label}", callback_data=callback_data)


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4ca Status", callback_data="cmd_status"),
            InlineKeyboardButton("\U0001f4e1 Signals", callback_data="cmd_signals"),
        ],
        [
            InlineKeyboardButton("\U0001f4b9 Trades", callback_data="cmd_trades"),
            InlineKeyboardButton("\U0001f9e9 Patterns", callback_data="cmd_patterns"),
        ],
        [
            InlineKeyboardButton("\U0001f4b8 Redeem", callback_data="cmd_redeem"),
            InlineKeyboardButton("\U0001f4dc Redemptions", callback_data="cmd_redemptions"),
        ],
        [
            InlineKeyboardButton("\u2699\ufe0f Settings", callback_data="cmd_settings"),
            InlineKeyboardButton("\U0001f9ea Demo", callback_data="cmd_demo"),
        ],
        [
            InlineKeyboardButton("\u2753 Help", callback_data="cmd_help"),
            InlineKeyboardButton("\U0001f3e0 Home", callback_data="cmd_menu"),
        ],
    ])


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def settings_keyboard(
    autotrade_on: bool,
    trade_amount: float,
    auto_redeem_on: bool = False,
    demo_trade_on: bool = False,
    demo_bankroll: float = 1000.00,
    trade_mode: str = "fixed",
    trade_pct: float = 5.0,
) -> InlineKeyboardMarkup:
    # Row 1: paired toggles — related switches side-by-side
    at_label = f"\U0001f916 AutoTrade: {'ON' if autotrade_on else 'OFF'}"
    ar_label = f"{'\U0001f4b0' if auto_redeem_on else '\U0001f4e6'} Auto-Redeem: {'ON' if auto_redeem_on else 'OFF'}"

    # Row 2: mode toggle + value input
    if trade_mode == "pct":
        mode_label = "\U0001f4ca Mode: PCT"
        value_label = f"\U0001f4b5 {trade_pct:.1f}%"
    else:
        mode_label = "\U0001f4ca Mode: FIXED"
        value_label = f"\U0001f4b5 ${trade_amount:.2f}"

    # Row 3: demo toggle + bankroll display
    dt_label = f"\U0001f9ea Demo: {'ON' if demo_trade_on else 'OFF'}"
    db_label = f"\U0001f4b5 Bankroll: ${demo_bankroll:.2f}"

    return InlineKeyboardMarkup([
        # Paired toggles
        [
            InlineKeyboardButton(at_label, callback_data="toggle_autotrade"),
            InlineKeyboardButton(ar_label, callback_data="toggle_auto_redeem"),
        ],
        # Mode + value (side by side)
        [
            InlineKeyboardButton(mode_label, callback_data="toggle_trade_mode"),
            InlineKeyboardButton(value_label, callback_data="change_amount"),
        ],
        # Demo section
        [
            InlineKeyboardButton(dt_label, callback_data="toggle_demo_trade"),
            InlineKeyboardButton(db_label, callback_data="set_demo_bankroll"),
        ],
        # Destructive action — full-width, alone
        [InlineKeyboardButton("\U0001f504 Reset Bankroll", callback_data="reset_demo_bankroll")],
        # Back
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Filter rows (Last 10 / Last 50 / All Time)
# ---------------------------------------------------------------------------

def signal_filter_row(active: str = "all") -> InlineKeyboardMarkup:
    buttons = [
        _filter_btn("Last 10", "signals_10", active),
        _filter_btn("Last 50", "signals_50", active),
        _filter_btn("All Time", "signals_all", active),
    ]
    return InlineKeyboardMarkup([
        buttons,
        [
            InlineKeyboardButton("\U0001f4c4 CSV", callback_data="download_csv"),
            InlineKeyboardButton("\U0001f4ca Excel", callback_data="download_xlsx"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


def trade_filter_row(active: str = "all") -> InlineKeyboardMarkup:
    buttons = [
        _filter_btn("Last 10", "trades_10", active),
        _filter_btn("Last 50", "trades_50", active),
        _filter_btn("All Time", "trades_all", active),
    ]
    return InlineKeyboardMarkup([
        buttons,
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Back button only
# ---------------------------------------------------------------------------

def back_to_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Download keyboard (standalone)
# ---------------------------------------------------------------------------

def download_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4c4 CSV", callback_data="download_csv"),
            InlineKeyboardButton("\U0001f4ca Excel", callback_data="download_xlsx"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Redeem keyboards
# ---------------------------------------------------------------------------

def redeem_confirm_keyboard() -> InlineKeyboardMarkup:
    """Shown after a dry-run scan — lets user confirm or cancel."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("\u2705 Confirm & Redeem All", callback_data="redeem_confirm")],
        [
            InlineKeyboardButton("\u274c Cancel", callback_data="redeem_cancel"),
            InlineKeyboardButton("\U0001f519 Menu", callback_data="cmd_menu"),
        ],
    ])


def redeem_done_keyboard() -> InlineKeyboardMarkup:
    """Shown after redemptions complete."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4dc History", callback_data="cmd_redemptions"),
            InlineKeyboardButton("\U0001f519 Menu", callback_data="cmd_menu"),
        ],
    ])


def demo_filter_row(active: str = "all") -> InlineKeyboardMarkup:
    """Filter row for the /demo dashboard."""
    return InlineKeyboardMarkup([
        [
            _filter_btn("Last 10", "demo_10", active),
            _filter_btn("Last 50", "demo_50", active),
            _filter_btn("All Time", "demo_all", active),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


def pattern_filter_row() -> InlineKeyboardMarkup:
    """Keyboard for the /patterns dashboard — kept for backward compat."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4ca Excel", callback_data="download_pattern_xlsx"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Pattern performance keyboard
# ---------------------------------------------------------------------------

def pattern_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for the /patterns dashboard."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("\U0001f4ca Excel", callback_data="download_pattern_xlsx")],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])
