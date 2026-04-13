# Architecture

Technical documentation for StockAgent's internal architecture.

---

## System Overview

```
                      CLI / Cron
                          |
                    Agent (agent.py)
                   /    |    |    \
          Data    Analysis  Execution  Memory
        Sources    Engine    Layer    System
            \      |      |      /
             Broker Abstraction Layer
                    |      |
              Simulated  IBKR
```

The agent orchestrates four subsystems through a central coordinator (`agent.py`). Each subsystem is independently testable and replaceable.

---

## Broker Abstraction

All market interaction goes through the `BaseBroker` interface. The rest of the system never touches a specific broker implementation directly.

```
BaseBroker (ABC)
  |
  +-- SimulatedBroker    local mock, GBM prices, persistent state
  +-- IBKRBroker         real IBKR via ib_insync
```

**Factory**: `create_broker(config)` returns the right implementation based on `trading_mode`.

### Simulated broker

Uses Geometric Brownian Motion for price simulation:

```
dS = mu * S * dt + sigma * S * dW
```

- Configurable drift (default: 5% annual) and per-ticker volatility
- Realistic bid/ask spread simulation (5 bps)
- Commission and slippage modeling
- State persisted to `data/sim_state.json` across sessions

### IBKR broker

Connects to TWS or IB Gateway via `ib_insync`. Supports:
- Portfolio and position queries
- Real-time market data
- Market, limit, and stop orders
- Historical price data

---

## Memory System

Three tiers, each serving a different purpose:

### Tier 1: Long-term memory

**Purpose**: Prevent the user from having to repeat themselves.

**Storage**: SQLite database.

**Injection**: Entire corpus (capped at ~2KB) is injected into every reasoning turn.

**Categories**:
| Category | Example | Priority |
|----------|---------|----------|
| `user_preference` | "User prefers dividend stocks" | 7-8 |
| `correction` | "Don't sell NVDA on dips" | 9 |
| `strategy` | "Always hedge tech with financials" | 7 |
| `lesson` | "Earnings surprises reverse in 3 days" | 6 |
| `environment` | "IBKR port 7497 is paper" | 8 |

### Tier 2: Research corpus

**Purpose**: Deep context available on demand without information overload.

**Storage**: SQLite FTS5 + content files.

**Usage**: Agent queries corpus when analyzing specific tickers. Results are NOT injected into memory — they're provided as supplementary context.

**Document types**: research reports, earnings transcripts, SEC filings, news articles, agent-generated analyses.

### Tier 3: Trade journal

**Purpose**: Institutional memory — what was decided, why, and how it turned out.

**Storage**: SQLite.

**Key design**: Every decision is recorded with a full snapshot of what the agent knew at the time (analysis scores, indicators, portfolio state). This enables post-hoc analysis:

```python
journal.get_decision_accuracy(days=30)
# → {"correct": 18, "total_evaluated": 25, "accuracy": 0.72}
```

---

## Analysis Engine

Four analyzers produce scores from -1 (bearish) to 1 (bullish). The `CompositeAnalyzer` combines them.

### Technical analyzer

| Indicator | What it measures | Signal logic |
|-----------|-----------------|--------------|
| SMA/EMA crossover | Trend direction | Price above/below MAs, golden/death cross |
| RSI | Overbought/oversold | >70 sell, <30 buy |
| MACD | Momentum | Signal line crossover, histogram direction |
| Bollinger Bands | Volatility position | %B within bands |
| Volume | Confirmation | High volume confirms price direction |
| Linear regression | Overall trend | Slope direction over 20 periods |

### Fundamental analyzer

Scores four dimensions:

1. **Valuation** — P/E, PEG, P/B relative to sector benchmarks
2. **Growth** — Revenue and earnings growth rates
3. **Profitability** — Net margin, ROE, operating margin
4. **Financial health** — Debt/equity, current ratio

Sector-specific benchmarks adjust scoring (e.g., tech companies are expected to have higher P/E ratios than utilities).

### Sentiment analyzer

- Rule-based NLP scoring on news headlines (positive/negative word matching)
- Volume-weighted aggregation (recent articles weighted higher)
- VIX integration (market fear gauge)

### Composite signal

```
composite = w1 * technical + w2 * fundamental + w3 * sentiment + w4 * momentum
```

Default weights: technical 0.35, fundamental 0.35, sentiment 0.15, momentum 0.15.

Additional outputs:
- Signal agreement check (do all analyzers agree?)
- Confidence score (0-1)
- Risk level assessment
- Human-readable recommendation

---

## Risk Engine

Pre-trade validation. If any BLOCKER rule fails, the trade does not happen.

| Rule | Default | Type |
|------|---------|------|
| Max position concentration | 20% of portfolio | BLOCKER |
| Max single trade size | 5% of portfolio | BLOCKER |
| Daily loss limit | 3% of portfolio | BLOCKER |
| Max trades per day | 10 | BLOCKER |
| Short selling | Must own shares | BLOCKER |
| Large trade threshold | $5,000 | WARNING |
| Stop-loss | 8% below cost basis | Auto-sell |
| Take-profit | 20% above cost basis | Auto-sell |
| Max drawdown | 15% | WARNING |

The engine also calculates risk-adjusted position sizes — if a requested trade exceeds limits, it returns the maximum safe quantity instead of just blocking.

---

## Data Sources

| Source | Provider | Free | Data |
|--------|----------|------|------|
| Market data | yfinance | Yes | OHLCV, quotes, company info |
| News | Yahoo Finance RSS | Yes | Headlines, basic metadata |
| Fundamentals | yfinance | Yes | Income, balance sheet, cash flow |
| SEC filings | EDGAR API | Yes | 10-K, 10-Q, 8-K, Form 4 |
| Enhanced news | Finnhub | Optional (API key) | Company news with full text |
| Enhanced news | NewsAPI | Optional (API key) | Aggregated news articles |

---

## Trade Execution Flow

```
Signal from Analysis Engine
         |
    Risk Engine check
    /           \
  pass          fail
   |              |
  Order created  Decision recorded (blocked)
   |              |
  Broker.submit  Journal entry
   |
  Filled/rejected
   |
  Journal + Notification
```

Every step is recorded to the trade journal, including blocked trades, so the agent can later analyze what it wanted to do and why it was prevented.

---

## Configuration

Hierarchical: YAML file → environment variables → runtime defaults.

```python
config = AppConfig.from_yaml("config.yaml")
# Env vars with STOCKAGENT_ prefix override YAML values
```

All settings are typed dataclasses with sensible defaults. The system works out of the box with zero configuration.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Market data and fundamentals |
| `pandas` | Data manipulation |
| `pyyaml` | Configuration parsing |
| `requests` | HTTP for news, SEC, webhooks |
| `ta` | Technical analysis indicators |
| `ib_insync` | IBKR integration (optional) |
| `sqlite3` | Memory system (stdlib) |
