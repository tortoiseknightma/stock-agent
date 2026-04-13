"""
StockAgent Memory System
=======================
Three-tier memory architecture inspired by Hermes Agent:

1. PERSISTENT MEMORY (long_term.py)
   - Durable facts: user preferences, strategies, lessons learned
   - Injected into every reasoning turn
   - Compact, focused on preventing future corrections

2. RESEARCH CORPUS (research_corpus.py)
   - Indexed collection of research reports, earnings calls, SEC filings
   - Semantic search for relevant context
   - Not stored in memory — accessed on-demand

3. TRADE JOURNAL (trade_journal.py)
   - Complete record of every decision and outcome
   - Enables post-hoc review and strategy improvement
   - Session-based with cross-session search

Design Principles (from Hermes):
- Memory prevents the user from having to repeat themselves
- Research is indexed but NOT injected (avoid information overload)
- Trade journal enables continuous improvement through reflection
- Skills capture reusable patterns (e.g., "how to analyze earnings surprises")
"""

from .long_term import LongTermMemory, MemoryEntry
from .research_corpus import ResearchCorpus, ResearchDoc
from .trade_journal import TradeJournal, TradeRecord, DecisionRecord

__all__ = [
    "LongTermMemory", "MemoryEntry",
    "ResearchCorpus", "ResearchDoc",
    "TradeJournal", "TradeRecord", "DecisionRecord",
]
