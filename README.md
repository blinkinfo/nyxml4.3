# NyxBot - BTC 5-Min Pattern Trading Bot

<div align="center">

Automated Polymarket trading bot for BTC 5-minute Up/Down binary options.
Matches **multi-depth candle patterns** (5-7 candles) against a table of historically-validated sequences,
executes **Fill-Or-Kill orders** via Telegram, and **auto-redeems** winning positions on-chain.

[Features](#-features) • [How It Works](#-how-it-works) • [Setup](#-setup) • [Commands](#-telegram-bot-commands) • [Architecture](#-architecture) • [Deploy](#-deployment)

</div>

---

## ✨ Features

| | |
|---|---|
| **Multi-Depth Pattern Matching** | Greedy 7/6/5-candle scan -- longest match wins |
| **Live Telegram Dashboard** | Real-time signals, P&L stats, inline analytics |
| **Auto-Trading (FOK)** | Fill-Or-Kill orders with retry + time-fence guards |
| **Demo Mode** | Paper-trade with a virtual $1,000 bankroll, zero risk |
| **Auto-Redeem** | Scans & reclaims resolved positions on-chain via web3.py |
| **Gnosis Safe Support** | Sig-type-2 redemptions routed through Safe.execTransaction |
| **Trade Sizing Modes** | Fixed USDC amount or percentage of balance |
| **Hour Filter** | Block trading during specific UTC hours |
| **Data Export** | Download signals & pattern stats as CSV or Excel |
| **SQLite Persistence** | All data survives restarts; unresolved signals auto-recover |
| **Single-Chat Auth** | Locked to your authorized Telegram chat ID |
| **Pluggable Strategies** | Registry-based -- add new strategies without touching core |
| **Railway-Ready** | One-click deploy with Procfile |

---

## 🔍 How It Works

### The Pattern Strategy

Every 5 minutes, Polymarket opens a binary market: **"Will BTC go Up or Down in the next 5 minutes?"**

NyxBot watches BTC/USD candles on Coinbase, matches the most recent closed candles against a table of historically-validated patterns, and predicts the next candle's direction.

**At T-85s** (85 seconds before the slot closes), the bot:

1. **Fetches** 12 recent 5-minute BTC-USD candles from Coinbase (~25-hour window)
2. **Drops** the tail candle (still-forming at T-85s) -- ensures only confirmed-closed data
3. **Builds pattern strings** at depths 7, 6, and 5 (longest-first, greedy match)
   - `U` = candle closed >= opened (green)
   - `D` = candle closed < opened (red)
4. **Looks up** each pattern in the pattern table -- first (longest) match wins
5. **If matched** -> fires a signal with the predicted direction (Up or Down)
6. **If no match** -> skips the slot, logs it, notifies you on Telegram

### Multi-Depth Pattern Table

The bot scans at **three depths simultaneously**. Longer patterns are preferred (greedy match), giving higher specificity.

**7-Candle Patterns (4 patterns)**

| Pattern | Prediction |
|---------|------------|
| `DDDUDDD` | UP |
| `DUDDDDD` | UP |
| `UDUUUUU` | UP |
| `UDUUUUD` | DOWN |

**6-Candle Patterns (15 patterns)**

| Pattern | Prediction | |
|---------|------------|-|
| `DDDDDD` | UP | `DUUUDU` -> DOWN |
| `DUUUUD` | DOWN | `DUUUDD` -> DOWN |
| `UDDUUD` | UP | `DUUDDD` -> UP |
| `DUDUDU` | DOWN | `UUDUUU` -> DOWN |
| `DUDUUD` | UP | `DDUDDU` -> UP |
| `DDDUUD` | DOWN | `UUUDUD` -> DOWN |
| `UDUUDU` | DOWN | `DDUDDD` -> UP |
| `DUDDDU` | DOWN |

**5-Candle Patterns (1 pattern)**

| Pattern | Prediction |
|---------|------------|
| `DDDUU` | DOWN |

> **20 total patterns** across 3 depths. All are hardcoded in `core/strategies/pattern_strategy.py` and can be extended.

### Signal Flow

```
[Every 5 min at T-85s]
  ├─ 1. PatternStrategy fetches 12 candles from Coinbase
  ├─ 2. Drops tail candle (safety), keeps 11 confirmed-closed
  ├─ 3. Greedy scan: depth 7 -> 6 -> 5, first match wins
  ├─ 4. Match? fetch Polymarket prices, return signal dict
  ├─ 5. No match? return skip, log shallowest pattern
  ├─ 6. Hour filter checks: is this a blocked UTC hour?
  ├─ 7. Signal persisted to DB
  ├─ 8. TradeManager gate (passthrough) proceeds
  ├─ 9. AutoTrade ON? place FOK order with retry logic
  ├─ 10. Demo mode ON? simulate trade, deduct bankroll
  └─ 11. Schedule resolution for slot_end + 30s

[Resolution: slot_end + 30s]
  ├─ 1. Poll Coinbase for the slot's candle (up to 5 retries)
  ├─ 2. close >= open => "Up", else "Down"
  ├─ 3. P&L: win = amount * (1/entry - 1), loss = -amount
  ├─ 4. Update signal + trade in DB
  └─ 5. Send resolution notification on Telegram

[Background Jobs]
  ├─ Reconciler: every 5 min, retries stuck resolutions from persistent queue
  ├─ Auto-Redeem: scans wallet for resolved positions, redeems on-chain
  └─ Startup Recovery: resolves any unresolved signals from previous run
```

### Order Execution (AutoTrade)

| Setting | Default | Description |
|---------|---------|-------------|
| `FOK_MAX_RETRIES` | `3` | Max retry attempts per slot |
| `FOK_RETRY_DELAY_BASE` | `2.0s` | Initial backoff delay |
| `FOK_RETRY_DELAY_MAX` | `5.0s` | Maximum backoff ceiling |
| `FOK_SLOT_CUTOFF_SECONDS` | `30` | Abort if < 30s remain in slot |

- Uses **Fill-Or-Kill (FOK)** market orders via `py-clob-client`
- Exponential backoff: 2s -> 4s -> 5s between retries
- **Time fence**: aborts if too close to slot end
- **Duplicate guard**: checks DB before each retry to prevent double-fills

### On-Chain Redemption

NyxBot supports two wallet types:

**EOA (Direct)**
- Private key directly controls the wallet
- `POLYMARKET_SIGNATURE_TYPE` = 1
- `POLYMARKET_FUNDER_ADDRESS` = the EOA address

**Gnosis Safe (Proxy)**
- EOA signs, Safe acts as msg.sender on CTF contract
- `POLYMARKET_SIGNATURE_TYPE` = 2
- `POLYMARKET_FUNDER_ADDRESS` = the Safe/proxy address
- Redemptions routed through `Safe.execTransaction()`
- Startup sanity check verifies EOA is a Safe owner

The redeemer:
1. Scans the Polymarket Data API for positions with `redeemable=true`
2. Filters settled markets only (`curPrice >= 0.99` or `<= 0.01`)
3. Calls `CTF.redeemPositions()` with `indexSets=[1,2]` (handles both won/lost)
4. Verifies post-tx position balances are zero
5. Records results in the DB

---

## ⚙ Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `POLYMARKET_PRIVATE_KEY` | Ethereum private key (signer EOA) |
| `POLYMARKET_FUNDER_ADDRESS` | Wallet address (EOA or Safe) |
| `TELEGRAM_BOT_TOKEN` | Bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Your authorized Telegram chat ID |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYMARKET_SIGNATURE_TYPE` | `2` | `2` = Gnosis Safe, `1` = direct EOA |
| `TRADE_AMOUNT_USDC` | `1.0` | Fixed trade size in USDC |
| `TRADE_MODE` | `fixed` | `fixed` or `pct` |
| `TRADE_PCT` | `5.0` | Trade as % of balance (when mode is `pct`) |
| `FOK_MAX_RETRIES` | `3` | Max FOK order retry attempts |
| `FOK_RETRY_DELAY_BASE` | `2.0` | Base retry delay (seconds) |
| `FOK_RETRY_DELAY_MAX` | `5.0` | Max retry delay (seconds) |
| `FOK_SLOT_CUTOFF_SECONDS` | `30` | Abort if less than this time remains in slot |
| `AUTO_REDEEM_INTERVAL_MINUTES` | `5` | How often auto-redeem scans |
| `POLYGON_RPC_URL` | `https://polygon-rpc.com` | RPC for on-chain redemptions |
| `DB_PATH` | `autopoly.db` | SQLite database file |
| `STRATEGY_NAME` | `pattern` | Active strategy module name |
| `BLOCKED_TRADE_HOURS_UTC` | `3,17` | UTC hours to skip trading (comma-separated) |

### Trade Sizing Modes

**Fixed Mode** (`TRADE_MODE=fixed`)
- Each trade uses `TRADE_AMOUNT_USDC` (default $1.00)
- Configurable from Telegram: `/settings` -> Change Amount

**Percentage Mode** (`TRADE_MODE=pct`)
- Each trade uses `TRADE_PCT`% of demo or real balance
- Minimum trade is always $1.00 (Polymarket limit)
- Configurable from Telegram: `/settings` -> Toggle Trade Mode

### Hour Filter

Trading is blocked during the UTC hours defined in `BLOCKED_TRADE_HOURS_UTC`.
Default blocks `03:00-03:59 UTC` and `17:00-17:59 UTC`.
Change without redeploying: set the env var and restart.

---

## 🚀 Setup

### Prerequisites

- **Python 3.10+**
- **Polymarket account** with funded Polygon wallet
- **Telegram bot token** from [@BotFather](https://t.me/BotFather)
- **Ethereum private key** for your Polymarket wallet

### Local Development

```bash
# Clone the repo
git clone https://github.com/your-org/nyx.git
cd nyx

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Run the bot
python main.py
```

### Telegram Bot Setup

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, follow prompts
2. Copy the bot token -> `TELEGRAM_BOT_TOKEN`
3. Message your new bot (any text)
4. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Find your `chat.id` in the JSON -> `TELEGRAM_CHAT_ID`

---

## 📱 Telegram Bot Commands

### Navigation

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and main menu |
| `/status` | Portfolio overview, balance, uptime, last signal |
| `/settings` | Toggle autotrade, change sizing, manage demo |
| `/help` | Command reference and strategy explanation |

### Analytics

| Command | Description |
|---------|-------------|
| `/signals` | Signal win rate, streaks, recent history |
| `/trades` | Trade P&L, deployed capital, ROI |
| `/patterns` | Per-pattern performance dashboard |
| `/demo` | Simulated bankroll and virtual trade history |
| `/redemptions` | On-chain redemption history |

### Actions

| Command | Description |
|---------|-------------|
| `/redeem` | Scan and redeem resolved positions (dry-run preview then confirm) |

### Interactive Features

The bot provides inline keyboards for:
- **Time Filters**: Last 10 / Last 50 / All Time (signals, trades, demo)
- **Pattern Filters**: Last 50 / Last 200 / All Time (pattern stats)
- **Toggles**: AutoTrade ON/OFF, Auto-Redeem ON/OFF, Demo ON/OFF, Trade Mode (Fixed/Pct)
- **Input Modes**: Set trade amount ($), set trade percentage (%), set demo bankroll, reset demo
- **Exports**: Download CSV or Excel of signals and pattern stats

---

## 🏗 Architecture

### Project Structure

```
nyx/
├── main.py                      # Entry point: DB init, bot startup, sanity checks
├── config.py                    # Env vars + constants (endpoints, chain IDs)
├── requirements.txt             # Python dependencies
├── Procfile                     # Railway: worker: python main.py
├── .env.example                 # Environment variable template
├── README.md                    # You are here
│
├── bot/                         # Telegram bot layer
│   ├── handlers.py              # All commands and callback query router
│   ├── keyboards.py             # Inline keyboard layouts
│   ├── formatters.py            # Message formatting utilities
│   └── middleware.py             # Chat ID auth guard
│
├── core/                        # Trading engine
│   ├── strategy.py              # Strategy orchestrator (registry-based)
│   ├── scheduler.py             # APScheduler: trading loop, resolution, reconciler
│   ├── trader.py                # FOK order execution and retry logic
│   ├── resolver.py              # Slot resolution via Coinbase candles
│   ├── trade_manager.py         # Pre-trade gate (hour filter, passthrough)
│   ├── redeemer.py              # On-chain CTF redemption (EOA + Safe)
│   ├── pending_queue.py         # Persistent JSON-backed retry queue
│   └── strategies/              # Pluggable strategy plugins
│       ├── __init__.py          # Registry: "pattern" -> PatternStrategy
│       ├── base.py              # Abstract BaseStrategy interface
│       └── pattern_strategy.py   # Multi-depth pattern matching (THE active strategy)
│
├── db/                          # Database layer
│   ├── models.py                # SQLite schema + init + migrations
│   └── queries.py               # All CRUD + analytics helpers
│
└── polymarket/                  # Polymarket API layer
    ├── client.py                # ClobClient wrapper (L2 credential derivation)
    ├── markets.py               # Slot boundaries, Gamma + CLOB price fetching
    └── account.py               # Balance, positions, connection status
```

### Database Schema

**4 Tables:**

| Table | Purpose |
|-------|---------|
| `signals` | Every signal check -- side, price, pattern, win/loss |
| `trades` | Executed orders -- amount, P&L, retries, status, demo flag |
| `settings` | Key-value runtime config -- autotrade, sizing, demo |
| `redemptions` | On-chain redemption records -- tx hash, gas, verified |

**Default Settings** (seeded on first run):

| Key | Default | Description |
|-----|---------|-------------|
| `autotrade_enabled` | `false` | Auto-execute trades |
| `demo_trade_enabled` | `false` | Virtual paper trading |
| `demo_bankroll_usdc` | `1000.00` | Starting virtual balance |
| `auto_redeem_enabled` | `false` | Automatic on-chain redemption |
| `trade_mode` | `fixed` | `fixed` or `pct` |
| `trade_pct` | `5.0` | Percent sizing (pct mode) |
| `trade_amount_usdc` | from env | Fixed sizing amount |

### Key Technical Decisions

- **Async-first**: `aiosqlite` and `httpx.AsyncClient` throughout; sync `py-clob-client` wrapped in `asyncio.to_thread()`
- **Strategy Registry**: Pluggable via `core/strategies/__init__.py` -- add new strategies by registering them
- **Graceful Degradation**: Coinbase API failure -> skip slot, retry next cycle
- **Persistent Queue**: Unresolved slots in `data/pending_slots.json`, retried by reconciler every 5 min
- **Startup Recovery**: Immediately resolves any stale signals from previous run
- **Gnosis Safe Support**: Sig-type-2 redemptions use Safe.execTransaction with EOA-signed hashes
- **Post-Tx Verification**: Balance check confirms position tokens are actually cleared after redemption
- **Config Validation**: Required env vars checked at startup, with clear error messages

---

## ⚡ Quick Start

1. **Clone and install**: `pip install -r requirements.txt`
2. **Configure**: `cp .env.example .env` and fill in required credentials
3. **Run**: `python main.py`
4. **Test in demo mode**: Open Telegram -> `/settings` -> Toggle Demo ON
5. **Monitor**: `/status`, `/signals`, `/demo`
6. **Go live**: Toggle AutoTrade ON when ready

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `py-clob-client>=0.34.0` | Polymarket CLOB order execution |
| `python-telegram-bot>=20.0` | Telegram bot framework (async v20) |
| `httpx>=0.25.0` | Async HTTP client for APIs |
| `apscheduler>=3.10.0` | Task scheduling |
| `python-dotenv>=1.0.0` | Environment variable loading |
| `aiosqlite>=0.19.0` | Async SQLite driver |
| `openpyxl>=3.1.0` | Excel file export |
| `web3>=6.0.0` | On-chain redemption (CTF + Safe) |

---

## 🔧 Extending: Adding New Strategies

The bot supports pluggable strategies via a registry pattern:

1. Create a new class in `core/strategies/your_strategy.py` extending `BaseStrategy`
2. Implement `async def check_signal() -> dict[str, Any] | None`
3. Register in `core/strategies/__init__.py`: `STRATEGIES["your_strategy"] = YourStrategy`
4. Set `STRATEGY_NAME=your_strategy` in your environment

See `core/strategies/pattern_strategy.py` for a complete reference implementation.

---

## ⚠ Risk Warning

This is **experimental software** for educational purposes.

Trading binary options carries **significant risk of loss**. The pattern strategy is based on historical analysis and **does not guarantee future results**.

- Always test in **demo mode** first: `/settings` -> Toggle Demo ON
- Monitor signal accuracy via `/signals` and `/demo` before enabling real trading
- The bot makes **autonomous trading decisions** -- review your strategy regularly
- **Only trade with funds you can afford to lose**

---

## 📄 License

MIT
