# StockAgent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

An autonomous stock investment agent that connects to your brokerage account, analyzes your portfolio using technical, fundamental, and sentiment signals, and manages trades with built-in risk controls.

> **Status**: Alpha. Core analysis and simulated trading are functional. IBKR integration requires a running TWS/Gateway instance.

---

## Features

**Multi-dimensional Analysis**

| Dimension | What it measures | Key indicators |
|-----------|-----------------|----------------|
| Technical | Price patterns & momentum | SMA/EMA crossovers, RSI, MACD, Bollinger Bands, volume |
| Fundamental | Financial health & valuation | P/E, PEG, margins, ROE, debt/equity, revenue growth |
| Sentiment | Market mood | News NLP scoring, article volume, VIX |
| Composite | Weighted signal aggregation | Configurable weights, confidence scoring |

**Risk Management**

- Position concentration limits (default: 20% max per stock)
- Single trade size caps (default: 5% of portfolio)
- Daily loss circuit breaker (default: 3%)
- Automated stop-loss (8%) and take-profit (20%)
- Daily trade count limits
- Large trade confirmation thresholds

**Persistent Memory**

The agent remembers context across sessions using a three-tier memory system:

- **Long-term memory** — user preferences, corrections, and lessons learned; injected into every decision
- **Research corpus** — indexed research reports and analyses; searchable on demand
- **Trade journal** — complete record of every decision with full context and outcome tracking

**Broker Integration**

| Mode | Description | Risk |
|------|-------------|------|
| `simulated` | Local mock with simulated price dynamics | None |
| `paper` | IBKR paper trading account | None |
| `advisory` | Analysis only, no execution | None |
| `live` | Real trading (extra confirmation required) | Real money |

**Notifications**

Multi-channel alerts via CLI, Feishu/Lark webhooks, Telegram bot, or local file reports.

---

## Quick Start

```bash
# Clone
git clone https://github.com/tortoiseknightma/stock-agent.git
cd stock-agent

# Install dependencies
pip install yfinance pandas pyyaml requests ta

# Run (uses simulated broker by default)
python cli.py status
python cli.py analyze
python cli.py analyze AAPL
```

No API keys required for simulated mode. The agent ships with a default demo portfolio and generates realistic price data locally.

---

## Installation

### Requirements

- Python 3.10+
- pip

### Core dependencies

```bash
pip install yfinance pandas pyyaml requests ta
```

### Optional: IBKR integration

```bash
pip install ib_insync nest-asyncio
```

Requires [TWS](https://www.interactivebrokers.com/en/trading/tws.php) or [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway.php) running with API access enabled.

### Optional: visualization

```bash
pip install matplotlib numpy
```

---

## Configuration

All settings live in `config.yaml`. Override any value with environment variables using the `STOCKAGENT_` prefix.

```yaml
# config.yaml
trading_mode: simulated  # simulated | paper | advisory | live

risk:
  max_position_pct: 0.20    # max 20% in one stock
  stop_loss_pct: 0.08       # 8% stop-loss
  take_profit_pct: 0.20     # 20% take-profit
  daily_loss_limit_pct: 0.03

analysis:
  weight_technical: 0.35
  weight_fundamental: 0.35
  weight_sentiment: 0.15
  weight_momentum: 0.15
```

**Environment overrides:**

```bash
export STOCKAGENT_TRADING_MODE=paper
export STOCKAGENT_IBKR_HOST=127.0.0.1
export STOCKAGENT_IBKR_PORT=7497          # 7497=paper, 7496=live
export STOCKAGENT_FEISHU_WEBHOOK=https://...
```

---

## Usage

### Portfolio overview

```bash
$ python cli.py status

  STOCKAGENT STATUS
  =======================================================

  Trading Mode:   SIMULATED
  Connected:      True
  Holdings:       8 tickers

  Portfolio:
    Net Value:    $105,234.56
    Cash:         $42,150.00
    Day P&L:      +$1,234.56
    Unrealized:   +$3,084.56

  Positions:
    AAPL      100   $178.50    $17,850.00    +$850
    NVDA       40   $892.30    $35,692.00  +$6,972
    ...
```

### Analyze a stock

```bash
$ python cli.py analyze AAPL

  🟢🟢 AAPL:     STRONG_BUY
  Score:    +0.723
  Confidence: 78%
  Risk:     low

  Strong BUY — Technicals: strong_buy; Fundamentals: Attractive
  valuation; strong growth; high profitability; Sentiment: positive
  news flow (8 articles)
```

### Execute trades

```bash
python cli.py buy AAPL              # buy with default sizing
python cli.py buy AAPL --pct 5      # target 5% of portfolio
python cli.py sell AAPL             # sell entire position
python cli.py sell AAPL --qty 50    # sell 50 shares
```

### Scheduled routines

```bash
python cli.py premarket     # pre-market analysis & briefing
python cli.py intraday      # stop-loss / take-profit check
python cli.py postmarket    # end-of-day review & report
python cli.py daemon        # run all routines on schedule
```

### Memory management

```bash
python cli.py memory list                               # view all memories
python cli.py memory add "prefer long-term holds" --category preference
python cli.py memory add "don't sell NVDA on dips" --category correction
python cli.py review --days 7                           # trade journal review
python cli.py research "AI semiconductor earnings"      # search corpus
```

---

## Project Structure

```
stock-agent/
├── cli.py                       # CLI entry point (16 commands)
├── agent.py                     # Main orchestrator
├── config.yaml                  # Default configuration
├── core/
│   ├── config.py                # Dataclass config + YAML + env vars
│   ├── broker/
│   │   ├── base.py              # Abstract broker interface
│   │   ├── simulated.py         # Mock broker (GBM price model)
│   │   ├── ibkr_broker.py       # IBKR via ib_insync
│   │   └── factory.py           # Factory pattern
│   └── memory/
│       ├── long_term.py         # Persistent memory (SQLite)
│       ├── research_corpus.py   # Indexed research (FTS5)
│       └── trade_journal.py     # Decision & trade history
├── data/sources/
│   ├── market_data.py           # Yahoo Finance
│   ├── news.py                  # News aggregation + sentiment
│   ├── fundamentals.py          # Financial ratios
│   └── sec_filings.py           # SEC EDGAR
├── analysis/
│   ├── technical/technical.py   # RSI, MACD, Bollinger, MAs
│   ├── fundamental/fundamental.py
│   ├── sentiment/sentiment.py
│   └── composite/composite.py   # Weighted signal combination
├── execution/
│   ├── risk_engine.py           # Pre-trade risk checks
│   └── executor.py              # Trade orchestration
├── push/notifier.py             # Multi-channel alerts
├── docs/ARCHITECTURE.md         # Technical architecture
├── tests/
└── requirements.txt
```

---

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical documentation covering:

- Broker abstraction layer design
- Three-tier memory system
- Analysis engine signal pipeline
- Risk management rules
- Data source integration

---

## IBKR Setup

1. Install [TWS](https://www.interactivebrokers.com/en/trading/tws.php) or [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway.php)
2. Enable API: **Configure → API → Settings → Enable ActiveX and Socket Clients**
3. Set port: **7497** (paper) or **7496** (live)
4. Set in config:
   ```yaml
   trading_mode: paper
   ibkr:
     host: 127.0.0.1
     port: 7497
   ```
5. Verify: `python cli.py status`

---

## Safety

- Default mode is `simulated` — no real money at risk
- Live trading requires explicit opt-in and confirmation
- Risk engine blocks trades that violate limits before they reach the broker
- All decisions are logged to the trade journal for audit

**This is not financial advice. Use at your own risk.**

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
# Fork and clone
git clone https://github.com/your-username/stock-agent.git
cd stock-agent

# Create a branch
git checkout -b feature/your-feature

# Make changes and test
python cli.py status  # basic smoke test

# Submit PR
git push origin feature/your-feature
```

---

## License

[MIT](LICENSE)
