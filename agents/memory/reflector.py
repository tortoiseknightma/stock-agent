"""
Agent Reflector
================
Post-trade reflection for agent roles. After an outcome is known, the LLM
extracts a lesson for each decision role that contributed to the trade.
Lessons are stored in role-specific SituationMemory instances.

Usage::

    reflector = AgentReflector(llm, memory_db="data/agent_memory.db")
    reflector.reflect(
        ticker="AAPL",
        state=agent_state,          # AgentState from the pipeline run
        outcome="Gained 4.2% in 5 days",
    )
"""

import json
import re
from typing import Optional, List, Dict

from core.llm.base import BaseLLMClient
from agents.state import AgentState
from agents.memory.situation_memory import SituationMemory


_REFLECT_SYSTEM = (
    "You are a trading post-mortem analyst. You review decisions made by a "
    "specific agent role during a trade and extract a concise, reusable lesson.\n\n"
    "Rules:\n"
    "- Write ONE lesson (1-2 sentences) that would help this role make better "
    "decisions in similar situations.\n"
    "- Be specific: cite what signal or argument was right/wrong.\n"
    "- Output only the lesson text — no JSON, no preamble."
)


class AgentReflector:
    """
    Extracts per-role lessons from trade outcomes and stores them in SituationMemory.

    Args:
        llm_client:  LLM client used for lesson extraction (None → skip reflection).
        memory_db:   Path to the shared SQLite DB for all role memories.
    """

    # Roles to reflect on and the state field holding their key argument
    _ROLES: List[Dict[str, str]] = [
        {"role": "bull",         "source": "investment_debate_state"},
        {"role": "bear",         "source": "investment_debate_state"},
        {"role": "aggressive",   "source": "risk_debate_state"},
        {"role": "conservative", "source": "risk_debate_state"},
        {"role": "neutral",      "source": "risk_debate_state"},
    ]

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        memory_db: str = "data/agent_memory.db",
    ):
        self._llm = llm_client
        self._memory_db = memory_db

    def reflect(self, ticker: str, state: AgentState, outcome: str) -> int:
        """
        Reflect on a completed trade and store lessons for each role.

        Args:
            ticker:  The ticker that was traded.
            state:   The AgentState from the pipeline run (contains debate rounds).
            outcome: Human-readable trade outcome, e.g. "Gained 4.2% in 5 days".

        Returns:
            Number of lessons stored.
        """
        stored = 0
        for role_cfg in self._ROLES:
            role = role_cfg["role"]
            argument = self._extract_argument(state, role, role_cfg["source"])
            if not argument:
                continue

            lesson = self._generate_lesson(ticker, role, argument, outcome)
            if lesson:
                mem = SituationMemory(self._memory_db, role=role)
                mem.store(
                    situation=f"{ticker} — {state.trade_date}",
                    decision=argument[:500],
                    outcome=outcome,
                    lesson=lesson,
                )
                stored += 1

        return stored

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_argument(state: AgentState, role: str, debate_attr: str) -> str:
        """Return the concatenated content of all rounds made by `role`."""
        debate = getattr(state, debate_attr, None)
        if debate is None:
            return ""
        rounds = [r["content"] for r in debate.rounds if r.get("role") == role]
        return " | ".join(rounds)[:800] if rounds else ""

    def _generate_lesson(
        self, ticker: str, role: str, argument: str, outcome: str
    ) -> Optional[str]:
        """Call LLM to distil a lesson; return None if LLM unavailable or fails."""
        if self._llm is None:
            return None

        user = (
            f"Ticker: {ticker}\n"
            f"Your role: {role}\n"
            f"Your argument during the trade: {argument}\n"
            f"Actual outcome: {outcome}\n\n"
            "What single lesson should this role remember for next time?"
        )
        try:
            return self._llm.chat_simple(
                system=_REFLECT_SYSTEM,
                user=user,
                max_tokens=200,
            )
        except Exception:
            return None
