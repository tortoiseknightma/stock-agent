"""
Trade Journal
==============
Complete record of every trading decision and its outcome.
Enables post-hoc review, strategy backtesting, and pattern recognition.

Key design (from Hermes session_search):
- Every decision is recorded with full context (what was known at the time)
- Outcomes are tracked with P&L attribution
- Searchable by ticker, strategy, date range, outcome
- Enables "what went wrong" analysis after losses
- Feeds back into long-term memory as lessons learned

The journal is the agent's "institutional memory" — it remembers not just
what happened, but WHY decisions were made and HOW they turned out.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum


class Signal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class AnalysisSnapshot:
    """Snapshot of analysis at decision time — what the agent knew."""
    technical_score: float = 0.0      # -1 to 1
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0
    signal: Signal = Signal.HOLD
    reasoning: str = ""
    indicators: Dict[str, float] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisSnapshot":
        d["signal"] = Signal(d["signal"])
        return cls(**d)


@dataclass
class DecisionRecord:
    """A single trading decision with full context."""
    ticker: str
    signal: Signal
    analysis: AnalysisSnapshot
    timestamp: float = field(default_factory=time.time)
    
    # Decision context
    portfolio_value: float = 0.0
    current_position: int = 0
    current_price: float = 0.0
    
    # What was decided
    intended_action: str = ""          # "buy 50 shares", "sell all", etc.
    confidence: float = 0.0           # 0-1
    risk_assessment: str = ""
    
    # Execution tracking
    executed: bool = False
    execution_reason: str = ""         # Why it was/wasn't executed
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        d["analysis"] = self.analysis.to_dict()
        return d


@dataclass
class TradeRecord:
    """An executed trade with outcome tracking."""
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float                      # Fill price
    timestamp: float = field(default_factory=time.time)
    
    # Context
    portfolio_value_before: float = 0.0
    portfolio_value_after: float = 0.0
    decision_id: Optional[int] = None  # Link to DecisionRecord
    
    # Costs
    commission: float = 0.0
    slippage: float = 0.0
    
    # Outcome tracking (updated later)
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    closed_at: Optional[float] = None
    
    @property
    def total_cost(self) -> float:
        return abs(self.quantity * self.price) + self.commission + abs(self.slippage)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        return d


class TradeJournal:
    """
    Complete trading journal with decision context and outcome tracking.
    
    Usage:
        journal = TradeJournal("data/trade_journal.db")
        
        # Record a decision (even if not executed)
        decision_id = journal.record_decision(DecisionRecord(
            ticker="AAPL",
            signal=Signal.BUY,
            analysis=analysis_snapshot,
            portfolio_value=100000,
            current_position=100,
            current_price=175.50,
            intended_action="buy 50 shares at market",
            confidence=0.75,
            executed=True,
            execution_reason="Within risk limits"
        ))
        
        # Record the actual trade
        journal.record_trade(TradeRecord(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50,
            price=175.65,
            decision_id=decision_id
        ))
        
        # Review outcomes
        review = journal.get_performance_review(days=30)
    """
    
    def __init__(self, db_path: str = "data/trade_journal.db"):
        self._in_memory = (db_path == ":memory:")
        if self._in_memory:
            self.db_path = Path(":memory:")
            # For in-memory mode, hold a persistent connection so the DB isn't lost
            self._mem_conn = sqlite3.connect(":memory:")
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._mem_conn = None
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    portfolio_value REAL,
                    current_position INTEGER,
                    current_price REAL,
                    intended_action TEXT,
                    confidence REAL,
                    risk_assessment TEXT,
                    executed INTEGER DEFAULT 0,
                    execution_reason TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    portfolio_value_before REAL,
                    portfolio_value_after REAL,
                    decision_id INTEGER,
                    commission REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    current_price REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    closed_at REAL,
                    FOREIGN KEY (decision_id) REFERENCES decisions(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_ticker 
                ON decisions(ticker)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_ticker 
                ON trades(ticker)
            """)
            conn.commit()
    
    def _conn(self) -> sqlite3.Connection:
        if self._in_memory:
            return self._mem_conn
        return sqlite3.connect(str(self.db_path))
    
    def record_decision(self, decision: DecisionRecord) -> int:
        """Record a trading decision. Returns decision ID."""
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO decisions 
                   (ticker, signal, analysis_json, timestamp, portfolio_value,
                    current_position, current_price, intended_action, confidence,
                    risk_assessment, executed, execution_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (decision.ticker, decision.signal.value,
                 json.dumps(decision.analysis.to_dict()),
                 decision.timestamp, decision.portfolio_value,
                 decision.current_position, decision.current_price,
                 decision.intended_action, decision.confidence,
                 decision.risk_assessment,
                 1 if decision.executed else 0, decision.execution_reason)
            )
            conn.commit()
            return cursor.lastrowid
    
    def record_trade(self, trade: TradeRecord) -> int:
        """Record an executed trade. Returns trade ID."""
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO trades 
                   (ticker, side, order_type, quantity, price, timestamp,
                    portfolio_value_before, portfolio_value_after, decision_id,
                    commission, slippage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade.ticker, trade.side.value, trade.order_type.value,
                 trade.quantity, trade.price, trade.timestamp,
                 trade.portfolio_value_before, trade.portfolio_value_after,
                 trade.decision_id, trade.commission, trade.slippage)
            )
            conn.commit()
            return cursor.lastrowid
    
    def update_trade_outcome(self, trade_id: int, current_price: float,
                             unrealized_pnl: float = None, realized_pnl: float = None):
        """Update a trade's outcome tracking."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades SET 
                   current_price = ?, unrealized_pnl = ?, realized_pnl = ?
                   WHERE id = ?""",
                (current_price, unrealized_pnl, realized_pnl, trade_id)
            )
            conn.commit()
    
    def get_decisions(self, ticker: str = None, signal: Signal = None,
                      executed: bool = None, days: int = None,
                      limit: int = 50) -> List[Dict]:
        """Query decisions with optional filters."""
        query = "SELECT * FROM decisions WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if signal:
            query += " AND signal = ?"
            params.append(signal.value)
        if executed is not None:
            query += " AND executed = ?"
            params.append(1 if executed else 0)
        if days:
            cutoff = time.time() - (days * 86400)
            query += " AND timestamp > ?"
            params.append(cutoff)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [self._row_to_decision_dict(row) for row in rows]
    
    def get_trades(self, ticker: str = None, side: OrderSide = None,
                   days: int = None, limit: int = 50) -> List[Dict]:
        """Query trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if side:
            query += " AND side = ?"
            params.append(side.value)
        if days:
            cutoff = time.time() - (days * 86400)
            query += " AND timestamp > ?"
            params.append(cutoff)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [self._row_to_trade_dict(row) for row in rows]
    
    def get_performance_review(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate a performance review for the specified period.
        Includes win rate, avg P&L, decision accuracy, etc.
        """
        cutoff = time.time() - (days * 86400)
        
        with self._conn() as conn:
            # Decision stats
            total_decisions = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE timestamp > ?", (cutoff,)
            ).fetchone()[0]
            
            executed_decisions = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE timestamp > ? AND executed = 1",
                (cutoff,)
            ).fetchone()[0]
            
            signal_dist = conn.execute(
                """SELECT signal, COUNT(*) FROM decisions 
                   WHERE timestamp > ? GROUP BY signal""",
                (cutoff,)
            ).fetchall()
            
            # Trade stats
            total_trades = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp > ?", (cutoff,)
            ).fetchone()[0]
            
            buy_trades = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp > ? AND side = 'buy'",
                (cutoff,)
            ).fetchone()[0]
            
            total_commission = conn.execute(
                "SELECT SUM(commission) FROM trades WHERE timestamp > ?",
                (cutoff,)
            ).fetchone()[0] or 0
            
            # Top traded tickers
            top_tickers = conn.execute(
                """SELECT ticker, COUNT(*) as cnt FROM trades 
                   WHERE timestamp > ? GROUP BY ticker ORDER BY cnt DESC LIMIT 5""",
                (cutoff,)
            ).fetchall()
        
        return {
            "period_days": days,
            "decisions": {
                "total": total_decisions,
                "executed": executed_decisions,
                "execution_rate": executed_decisions / max(total_decisions, 1),
                "signal_distribution": {row[0]: row[1] for row in signal_dist},
            },
            "trades": {
                "total": total_trades,
                "buys": buy_trades,
                "sells": total_trades - buy_trades,
                "total_commission": round(total_commission, 2),
            },
            "top_tickers": {row[0]: row[1] for row in top_tickers},
        }
    
    def get_decision_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze decision accuracy by comparing signals to subsequent price moves.
        A 'correct' BUY signal means price went up within N days.
        """
        cutoff = time.time() - (days * 86400)
        
        with self._conn() as conn:
            # Get decisions with associated trades
            rows = conn.execute(
                """SELECT d.signal, d.ticker, d.timestamp, t.price, t.current_price
                   FROM decisions d
                   LEFT JOIN trades t ON d.id = t.decision_id
                   WHERE d.timestamp > ? AND d.executed = 1
                   ORDER BY d.timestamp DESC""",
                (cutoff,)
            ).fetchall()
        
        correct = 0
        total = 0
        for signal, ticker, ts, entry_price, current_price in rows:
            if entry_price and current_price:
                total += 1
                price_change = (current_price - entry_price) / entry_price
                
                if signal in ("buy", "strong_buy") and price_change > 0:
                    correct += 1
                elif signal in ("sell", "strong_sell") and price_change < 0:
                    correct += 1
                elif signal == "hold" and abs(price_change) < 0.02:
                    correct += 1
        
        return {
            "total_evaluated": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
        }
    
    def get_trade_pair(self, sell_trade_id: int) -> Optional[Dict]:
        """
        Find the matching buy trade for a given sell trade.

        Returns a dict with 'sell' and 'buy' keys, or None if no matching
        buy trade is found. Used by the reflection engine to compute hold
        duration and entry price.
        """
        with self._conn() as conn:
            sell_row = conn.execute(
                "SELECT * FROM trades WHERE id = ? AND side = 'sell'",
                (sell_trade_id,)
            ).fetchone()
            if not sell_row:
                return None

            sell = self._row_to_trade_dict(sell_row)

            # Find the most recent buy for the same ticker before this sell
            buy_row = conn.execute(
                """SELECT * FROM trades
                   WHERE ticker = ? AND side = 'buy' AND timestamp < ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (sell["ticker"], sell["timestamp"])
            ).fetchone()

        if not buy_row:
            return None

        return {"sell": sell, "buy": self._row_to_trade_dict(buy_row)}

    def get_recent_sell_trades(self, days: int = 1) -> List[Dict]:
        """Return all sell trades within the given number of days."""
        return self.get_trades(side=OrderSide.SELL, days=days)

    def get_decision_by_id(self, decision_id: int) -> Optional[Dict]:
        """Fetch a single decision record by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        return self._row_to_decision_dict(row) if row else None

    def get_stats(self) -> Dict[str, Any]:
        """Get journal statistics."""
        with self._conn() as conn:
            decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            executed = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE executed = 1"
            ).fetchone()[0]
        
        return {
            "total_decisions": decisions,
            "total_trades": trades,
            "execution_rate": executed / max(decisions, 1),
        }
    
    @staticmethod
    def _row_to_decision_dict(row) -> dict:
        analysis = json.loads(row[3])
        return {
            "id": row[0], "ticker": row[1], "signal": row[2],
            "analysis": analysis, "timestamp": row[4],
            "portfolio_value": row[5], "current_position": row[6],
            "current_price": row[7], "intended_action": row[8],
            "confidence": row[9], "risk_assessment": row[10],
            "executed": bool(row[11]), "execution_reason": row[12],
        }
    
    @staticmethod
    def _row_to_trade_dict(row) -> dict:
        return {
            "id": row[0], "ticker": row[1], "side": row[2],
            "order_type": row[3], "quantity": row[4], "price": row[5],
            "timestamp": row[6], "portfolio_value_before": row[7],
            "portfolio_value_after": row[8], "decision_id": row[9],
            "commission": row[10], "slippage": row[11],
            "current_price": row[12], "unrealized_pnl": row[13],
            "realized_pnl": row[14], "closed_at": row[15],
        }
