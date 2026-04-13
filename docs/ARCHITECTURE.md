# StockAgent — AI-Powered Investment Agent

## Architecture Design Document

> **Project Purpose**: A production-grade autonomous stock investment agent that connects to Interactive Brokers, performs multi-dimensional market analysis, manages risk, executes trades, and maintains persistent memory for continuous improvement.

> **Resume Angle**: Demonstrates end-to-end agent architecture design — from broker abstraction and data pipeline to AI-driven analysis, risk management, and memory systems inspired by production agent frameworks.

---

## 1. Design Philosophy

### 1.1 Core Principles

StockAgent is designed around three principles that distinguish it from naive trading bots:

1. **Multi-Dimensional Intelligence** — No single indicator tells the whole story. StockAgent combines technical analysis (price patterns), fundamental analysis (financials), and sentiment analysis (news/social) into a composite signal with configurable weights.

2. **Safety-First Execution** — The risk engine is the last line of defense. No trade happens without passing position limits, daily loss limits, buying power checks, and trade size validation. The system defaults to read-only mode.

3. **Persistent Memory** — Inspired by production agent memory architectures, StockAgent maintains three memory tiers that enable the agent to learn from experience and prevent repeated mistakes.

### 1.2 Design Process

Each layer was designed to be:
- **Testable independently** (each module has clear inputs/outputs)
- **Replaceable** (swap IBKR for Alpaca by implementing one interface)
- **Configurable** (all thresholds in YAML, overridable by env vars)

---

## 2. System Architecture

```
                    +---------------------------------------------+
                    |          StockAgent Agent                    |
                    |         (agent.py - Orchestrator)           |
                    +------+------+------+------+----------------+
                           |      |      |      |
              +------------+      |      |      +------------+
              v                   v      v                   v
     +----------------+  +----------+ +--------------+ +-----------+
     |  Data Sources  |  | Analysis | |  Execution   | |  Memory   |
     |                |  |  Engine  | |    Layer     | |  System   |
     | - Market Data  |  |          | |              | |           |
     | - News/Sentim. |  | - Techn. | | - Risk Eng.  | | - LTM     |
     | - Fundamentals |  | - Fundam.| | - Executor   | | - Corpus  |
     | - SEC Filings  |  | - Sentim.| | - Journal    | | - Journal |
     +-------+--------+  +----+-----+ +------+-------+ +-----+-----+
             |                |               |               |
             +----------------+---------------+---------------+
                                     |
                            +--------v--------+
                            |  Broker Layer   |
                            |  (Abstract API) |
                            +-----------------+
                            | SimulatedBroker | <- Default (mock)
                            | IBKRBroker      | <- Real trading
                            +-----------------+
```

---

## 3. Memory System (Key Innovation)

The memory system is the most architecturally significant component. It's inspired by how production agents manage persistent context, adapted for financial decision-making.

### 3.1 Three-Tier Memory Architecture

```
+--------------------------------------------------------------+
|                    MEMORY ARCHITECTURE                        |
+-----------------+------------------+-------------------------+
|   TIER 1: LTM   |  TIER 2: CORPUS  |  TIER 3: TRADE JOURNAL|
|  (Injected into |  (Queried on-    |  (Complete decision    |
|   every turn)   |   demand)        |   history)             |
+-----------------+------------------+-------------------------+
| User prefs      | Research reports | Decision records       |
| Corrections     | Earnings calls   | (what was known)       |
| Strategies      | SEC filings      | Executed trades        |
| Lessons learned | Agent analyses   | (outcome tracking)     |
| Environment     | Industry reports | Performance metrics    |
| (API quirks)    |                  | Accuracy tracking      |
+-----------------+------------------+-------------------------+
| SQLite FTS5     | SQLite + files   | SQLite                 |
| ~2KB injection  | ~5KB per query   | Unlimited              |
| Always-on       | On-demand        | Always recording       |
+-----------------+------------------+-------------------------+
```

### 3.2 Why This Design?

**Tier 1 (Long-Term Memory)**: Injected into every reasoning turn, capped at ~2KB. This prevents the user from having to repeat preferences and corrections.

**Tier 2 (Research Corpus)**: Indexed but NOT automatically injected. Prevents information overload while keeping deep context accessible.

**Tier 3 (Trade Journal)**: Records every decision with full context — what the agent knew at the time, what it decided, and how it turned out.

---

## 4. Module Details

### 4.1 Configuration (`core/config.py`)

Dataclass-based configuration with:
- YAML file loading
- Environment variable overrides (prefix: STOCKAGENT_)
- Type-safe access
- 4 trading modes: simulated, paper, advisory, live

### 4.2 Broker Abstraction (`core/broker/`)

ALL market interaction goes through BaseBroker abstract class. The rest of the system is completely broker-agnostic.

Two implementations:
- SimulatedBroker: Geometric Brownian Motion price simulation, persistent state
- IBKRBroker: Real Interactive Brokers via ib_insync

### 4.3 Analysis Engine (`analysis/`)

Four sub-analyzers, each producing a score from -1 to 1:

| Analyzer | Key Indicators | Weight |
|----------|----------------|--------|
| Technical | SMA/EMA, RSI, MACD, Bollinger, Volume | 35% |
| Fundamental | P/E, PEG, margins, ROE, debt/equity | 35% |
| Sentiment | NLP sentiment, news volume, VIX | 15% |
| Momentum | Trend slope, relative strength | 15% |

### 4.4 Risk Engine

Pre-trade risk checks:
- Max position concentration: 20% of portfolio
- Max single trade size: 5% of portfolio
- Daily loss limit: 3% of portfolio
- Max trades per day: 10
- Stop-loss: 8% below cost
- Take-profit: 20% above cost

---

## 5. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Ecosystem for finance + AI |
| Broker API | ib_insync | Pythonic IBKR wrapper |
| Market Data | yfinance | Free, reliable, no API key |
| Database | SQLite + FTS5 | Embedded, zero-config, full-text search |
| Config | YAML + dataclasses | Human-readable, type-safe |
| Notifications | HTTP webhooks | Feishu, Telegram, extensible |

---

## 6. Project Structure

```
stock-agent/
├── __init__.py
├── agent.py                    # Main orchestrator
├── cli.py                      # CLI entry point
├── config.yaml                 # Configuration
├── core/
│   ├── __init__.py
│   ├── config.py               # Configuration dataclasses
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract broker interface
│   │   ├── simulated.py        # Mock broker (GBM prices)
│   │   ├── ibkr_broker.py      # IBKR via ib_insync
│   │   └── factory.py          # Broker factory
│   └── memory/
│       ├── __init__.py
│       ├── long_term.py        # Persistent memory (SQLite)
│       ├── research_corpus.py  # Indexed research (FTS5)
│       └── trade_journal.py    # Decision & trade history
├── data/
│   └── sources/
│       ├── __init__.py
│       ├── market_data.py      # Yahoo Finance data
│       ├── news.py             # News aggregation
│       ├── fundamentals.py     # Financial ratios
│       └── sec_filings.py      # SEC EDGAR filings
├── analysis/
│   ├── __init__.py
│   ├── technical/
│   │   └── technical.py        # Technical indicators
│   ├── fundamental/
│   │   └── fundamental.py      # Fundamental scoring
│   ├── sentiment/
│   │   └── sentiment.py        # News sentiment
│   └── composite/
│       └── composite.py        # Signal combination
├── execution/
│   ├── __init__.py
│   ├── risk_engine.py          # Pre-trade risk checks
│   └── executor.py             # Trade orchestration
├── push/
│   └── notifier.py             # Multi-channel alerts
└── docs/
    └── ARCHITECTURE.md         # This document
```
