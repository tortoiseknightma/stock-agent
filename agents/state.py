"""
Agent State
============
Shared state dataclass for multi-agent communication.
All agents read from and write to AgentState — no direct agent-to-agent calls.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from analysis.technical.technical import TechnicalSignal
from analysis.fundamental.fundamental import FundamentalSignal
from analysis.sentiment.sentiment import SentimentSignal
from analysis.composite.composite import CompositeSignal


@dataclass
class DebateState:
    """Tracks a multi-round debate (used for both research and risk debates)."""
    rounds: List[Dict[str, str]] = field(default_factory=list)
    count: int = 0
    last_speaker: str = ""
    verdict: str = ""


@dataclass
class AgentState:
    """
    Shared state passed through the multi-agent pipeline.

    Flow::

        Analysts (parallel) → Research Debate → Trader → Risk Debate → Final Decision
    """

    # --- Inputs ---
    ticker: str = ""
    trade_date: str = ""

    # --- Raw signals from existing rule-based analyzers ---
    technical_signal: Optional[TechnicalSignal] = None
    fundamental_signal: Optional[FundamentalSignal] = None
    sentiment_signal: Optional[SentimentSignal] = None
    composite_signal: Optional[CompositeSignal] = None

    # --- LLM analyst reports ---
    technical_report: str = ""
    fundamental_report: str = ""
    sentiment_report: str = ""
    macro_report: str = ""

    # --- Research debate ---
    investment_debate_state: DebateState = field(default_factory=DebateState)
    investment_plan: str = ""

    # --- Trade proposal ---
    trade_proposal: str = ""

    # --- Risk debate ---
    risk_debate_state: DebateState = field(default_factory=DebateState)
    final_decision: str = ""
    decision_signal: str = "HOLD"       # BUY / HOLD / SELL
    decision_conviction: float = 0.5

    # --- Memory context ---
    relevant_memories: List[str] = field(default_factory=list)

    # --- Audit trail ---
    agent_log: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, agent_name: str, action: str, content: str):
        """Append an entry to the audit trail."""
        self.agent_log.append({
            "agent": agent_name,
            "action": action,
            "content": content,
            "timestamp": time.time(),
        })
