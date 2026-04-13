# StockAgent — AI-Powered Investment Agent

An intelligent stock investment agent that combines technical, fundamental, and sentiment analysis to generate trading signals. Supports Interactive Brokers for live/paper trading with a built-in simulated broker for development.

## Features

- **Multi-Dimensional Analysis**: Technical (RSI, MACD, Bollinger), Fundamental (P/E, margins, ROE), Sentiment (news NLP)
- **Safety-First Risk Engine**: Position limits, daily loss limits, stop-loss/take-profit automation
- **Three-Tier Memory System**: Long-term memory, research corpus, trade journal
- **IBKR Integration**: Paper and live trading via Interactive Brokers
- **Simulated Broker**: Geometric Brownian Motion price simulation for backtesting
- **Multi-Channel Alerts**: CLI, Feishu/Lark, Telegram, email
- **Scheduled Operations**: Pre-market, intraday, post-market routines

## Quick Start

```bash
# Install dependencies
pip install yfinance pandas pyyaml requests

# Check status (uses simulated broker by default)
python cli.py status

# Analyze portfolio
python cli.py analyze

# Analyze specific stock
python cli.py analyze AAPL

# Generate daily report
python cli.py report
```

## Configuration

Edit `config.yaml` or use environment variables:

```bash
# Use IBKR paper trading
export STOCKAGENT_TRADING_MODE=paper
export STOCKAGENT_IBKR_PORT=7497

# Add Feishu webhook for alerts
export STOCKAGENT_FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
```

## Trading Modes

| Mode | Description | Risk |
|------|-------------|------|
| `simulated` | Local mock broker, no real money | None |
| `paper` | IBKR paper trading account | None |
| `advisory` | Analysis only, no execution | None |
| `live` | Real money trading | HIGH |

**Always start with `simulated` or `paper` mode.**

## CLI Commands

```bash
stockagent status          # Portfolio and agent status
stockagent analyze         # Full portfolio analysis
stockagent analyze AAPL    # Single ticker analysis
stockagent buy AAPL        # Execute buy (with confirmation)
stockagent sell AAPL       # Execute sell
stockagent report          # Generate daily report
stockagent review --days 7 # Review trade journal
stockagent memory list     # View long-term memory
stockagent memory add "text" --category lesson
stockagent research "query" # Search research corpus
stockagent premarket       # Pre-market routine
stockagent postmarket      # Post-market routine
stockagent daemon          # Run as scheduled daemon
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design documentation.

```
stock-agent/
├── agent.py              # Main orchestrator
├── cli.py                # CLI entry point
├── config.yaml           # Configuration
├── core/
│   ├── config.py         # Configuration dataclasses
│   ├── broker/           # Broker abstraction (Simulated + IBKR)
│   └── memory/           # Long-term memory, corpus, journal
├── data/sources/         # Market data, news, fundamentals, SEC
├── analysis/             # Technical, fundamental, sentiment, composite
├── execution/            # Risk engine, trade executor
└── push/                 # Multi-channel notifications
```

## IBKR Setup

1. Install [TWS](https://www.interactivebrokers.com/en/trading/tws.php) or [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway.php)
2. Enable API: Configure → API → Settings → Enable ActiveX and Socket Clients
3. Set port: 7497 (paper) or 7496 (live)
4. Set `trading_mode: paper` in config.yaml
5. Run `python cli.py status` to verify connection

## License

MIT
