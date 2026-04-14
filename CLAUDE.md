# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install core dependencies
pip install yfinance pandas pyyaml requests ta

# Optional: IBKR integration
pip install ib_insync nest-asyncio

# Optional: visualization
pip install matplotlib numpy

# Run (simulated mode, no API keys required)
python cli.py status
python cli.py analyze
python cli.py analyze AAPL
python cli.py buy AAPL --yes
python cli.py sell AAPL --qty 50 --yes

# Scheduled routines
python cli.py premarket
python cli.py intraday
python cli.py postmarket
python cli.py daemon

# Memory & research
python cli.py memory list
python cli.py memory add "lesson text" --category lesson --importance 7
python cli.py review --days 7
python cli.py research "AI semiconductors"
```

Test suite: `pytest --tb=short -q` (337 tests, all passing). Config in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Architecture

The system is orchestrated by `agent.py` (`StockAgentAgent`), which wires together four subsystems. `cli.py` is the sole entry point — it instantiates the agent and delegates to command handlers.

### Request flow

```
cli.py -> StockAgentAgent -> [Data Sources, Analysis Engine, Risk Engine, Broker]
                                        |
                               Memory System (SQLite)
```

### Broker abstraction (`core/broker/`)

All market interaction goes through `BaseBroker` (ABC in `base.py`). `factory.py`'s `create_broker(config)` selects the implementation based on `config.trading_mode`:

- `SimulatedBroker` - GBM price simulation, state persisted to `data/sim_state.json`
- `IBKRBroker` - real IBKR via `ib_insync`; requires TWS/Gateway running on port 7497 (paper) or 7496 (live)

Never import a concrete broker directly - always go through the factory or the `BaseBroker` interface.

### Analysis engine (`analysis/`)

Four analyzers each return a signal score from -1 (bearish) to +1 (bullish):

| Analyzer | Module | Input |
|---|---|---|
| `TechnicalAnalyzer` | `analysis/technical/technical.py` | OHLCV price history |
| `FundamentalAnalyzer` | `analysis/fundamental/fundamental.py` | Financial ratios + sector |
| `SentimentAnalyzer` | `analysis/sentiment/sentiment.py` | News headlines |
| `CompositeAnalyzer` | `analysis/composite/composite.py` | All three above |

`CompositeAnalyzer.analyze()` is the final step; it applies configurable weights (default: technical 0.35, fundamental 0.35, sentiment 0.15, momentum 0.15) and outputs a `CompositeSignal` with a signal string (`strong_buy` / `buy` / `hold` / `sell` / `strong_sell`), confidence score, risk level, and human-readable recommendation.

### Execution layer (`execution/`)

`RiskEngine` (`risk_engine.py`) runs pre-trade validation. BLOCKER rules (position concentration, trade size, daily loss limit, daily trade count) abort the trade before it reaches the broker. Non-blocking rules generate warnings. The engine also calculates maximum safe trade quantities when a requested size exceeds limits.

`TradeExecutor` (`executor.py`) orchestrates the full lifecycle: risk check -> order creation -> broker submission -> journal recording -> notification.

### Three-tier memory system (`core/memory/`)

All tiers use SQLite. Paths are configured under `memory:` in `config.yaml` (default: `data/`).

| Tier | Class | Storage | Purpose |
|---|---|---|---|
| Long-term memory | `LongTermMemory` | `data/memory.db` | User preferences, corrections, lessons - injected (~2KB cap) into every reasoning turn |
| Research corpus | `ResearchCorpus` | `data/research_corpus/` | FTS5-indexed reports and analyses - queried on demand per ticker |
| Trade journal | `TradeJournal` | `data/trade_journal.db` | Full decision audit log with outcome tracking |

### Configuration (`core/config.py`)

`AppConfig` is a nested set of typed dataclasses. `load_config("config.yaml")` -> `AppConfig.from_yaml()` -> YAML merged with `STOCKAGENT_*` environment variable overrides. All subsystems receive the relevant sub-config (e.g., `RiskEngine` receives `RiskConfig`).

### Data sources (`data/sources/`)

- `MarketDataProvider` - yfinance (OHLCV, company info)
- `FundamentalsProvider` - yfinance (income statement, balance sheet, cash flow -> ratios)
- `NewsProvider` - Yahoo Finance RSS; optional enhanced feeds via Finnhub/NewsAPI API keys
- `SECFilingsProvider` - EDGAR API (10-K, 10-Q, 8-K, Form 4)

### Notifications (`push/notifier.py`)

`Notifier` supports `cli` (stdout), Feishu/Lark webhook, and Telegram bot. Channels are configured in `config.yaml` under `push:`.

## Repository & branch policy

This repo uses a two-remote setup (see `REPO_STRUCTURE.md`):

| Remote | Repo | Purpose |
|--------|------|---------|
| `origin` | `stock-agent` (public) | Publish releases via `git push origin main` |
| `dev-origin` | `stock-agent-dev` (private) | Back up all branches via `git push dev-origin dev` |

**Branch rules:**
- `main` - user-facing code, README, LICENSE. Exists in both repos.
- `dev` - private notes, design process, experiments. Private repo only.

**Critical:** `docs/` under the `dev` branch contains private content. **Never sync `docs/` to `main`** when merging or cherry-picking from `dev`.

## Development workflow

After completing each implementation phase:

1. **Update the roadmap** — `docs/RESUME_AND_ROADMAP.md`: mark completed items with ✅, update test count, update "当前进度" and "待完成" sections.
2. **Commit** — include all changed files in a single commit with a descriptive message.
3. **Sync repos** — push to remotes per branch policy below.

## Key design constraints

- **Default is safe**: `trading_mode: simulated` in `config.yaml`; IBKR broker defaults to `readonly: true`. Live trading requires explicit mode change and confirmation prompts.
- **Risk engine is a hard gate**: trades that fail BLOCKER rules are recorded in the journal as blocked decisions but never reach the broker.
- **Memory paths are relative to CWD**: run `cli.py` from the project root so SQLite paths (`data/memory.db`, etc.) resolve correctly.
