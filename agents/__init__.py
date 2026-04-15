"""
Multi-Agent Orchestration System
=================================
TradingAgents-style 12-agent system for stock analysis and trading decisions.

4 layers:
1. Analyst Layer (4 agents) — Technical, Fundamental, Sentiment, Macro
2. Research Debate Layer (3 agents) — Bull, Bear, ResearchManager
3. Execution Layer (1 agent) — Trader
4. Risk Debate Layer (4 agents) — Aggressive, Conservative, Neutral, PortfolioManager
"""

from .state import AgentState, DebateState
from .base import BaseAgent
from .graph import TradingGraph

__all__ = ["AgentState", "DebateState", "BaseAgent", "TradingGraph"]
