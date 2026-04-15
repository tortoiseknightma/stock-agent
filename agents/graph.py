"""
Trading Graph
==============
Lightweight multi-agent orchestrator. Runs the 4-layer pipeline:

1. Analysts (parallel — sequential for now)
2. Research Debate (Bull ↔ Bear → ResearchManager)
3. Trader (trade proposal)
4. Risk Debate (Aggressive ↔ Conservative ↔ Neutral → PortfolioManager)

All communication flows through a shared AgentState.
"""

import datetime
from typing import Dict, Optional

from core.config import AppConfig
from core.llm.base import BaseLLMClient
from core.broker.base import BaseBroker
from data.sources.market_data import MarketDataProvider
from data.sources.fundamentals import FundamentalsProvider
from data.sources.news import NewsProvider
from analysis.technical.technical import TechnicalAnalyzer
from analysis.fundamental.fundamental import FundamentalAnalyzer
from analysis.sentiment.sentiment import SentimentAnalyzer

from agents.base import BaseAgent
from agents.state import AgentState
from agents.analysts.technical import TechnicalAnalystAgent
from agents.analysts.fundamental import FundamentalAnalystAgent
from agents.analysts.sentiment import SentimentAnalystAgent
from agents.analysts.macro import MacroAnalystAgent
from agents.researchers.bull import BullResearcherAgent
from agents.researchers.bear import BearResearcherAgent
from agents.managers.research_manager import ResearchManagerAgent
from agents.trader.trader import TraderAgent
from agents.risk.aggressive import AggressiveDebatorAgent
from agents.risk.conservative import ConservativeDebatorAgent
from agents.risk.neutral import NeutralDebatorAgent
from agents.managers.portfolio_manager import PortfolioManagerAgent


class TradingGraph:
    """
    Multi-agent orchestrator that runs the full analysis pipeline.

    Usage::

        graph = TradingGraph(config, llm_client=llm, fast_llm_client=fast_llm,
                             broker=broker)
        state = graph.run("AAPL")
    """

    def __init__(
        self,
        config: AppConfig,
        llm_client: Optional[BaseLLMClient] = None,
        fast_llm_client: Optional[BaseLLMClient] = None,
        broker: Optional[BaseBroker] = None,
        market_data: Optional[MarketDataProvider] = None,
        fundamentals: Optional[FundamentalsProvider] = None,
        news: Optional[NewsProvider] = None,
    ):
        self._config = config
        self._llm = llm_client                           # deep model for judges
        self._fast_llm = fast_llm_client or llm_client   # fast model for analysts
        self._broker = broker

        # Data providers (create defaults if not injected)
        self._market_data = market_data or MarketDataProvider()
        self._fundamentals = fundamentals or FundamentalsProvider()
        self._news = news or NewsProvider()

        self._agents: Dict[str, BaseAgent] = self._create_agents()

    def _create_agents(self) -> Dict[str, BaseAgent]:
        agents: Dict[str, BaseAgent] = {}

        # Layer 1: Analysts (use fast model)
        agents["technical"] = TechnicalAnalystAgent(
            self._fast_llm,
            TechnicalAnalyzer(self._config.analysis),
            self._market_data,
        )
        agents["fundamental"] = FundamentalAnalystAgent(
            self._fast_llm,
            FundamentalAnalyzer(),
            self._fundamentals,
        )
        agents["sentiment"] = SentimentAnalystAgent(
            self._fast_llm,
            SentimentAnalyzer(),
            self._news,
        )
        agents["macro"] = MacroAnalystAgent(self._fast_llm)

        # Layer 2: Researchers (fast model) + ResearchManager (deep model)
        agents["bull"] = BullResearcherAgent(self._fast_llm)
        agents["bear"] = BearResearcherAgent(self._fast_llm)
        agents["research_manager"] = ResearchManagerAgent(self._llm)

        # Layer 3: Trader (fast model)
        agents["trader"] = TraderAgent(self._fast_llm, self._broker)

        # Layer 4: Risk debators (fast model) + PortfolioManager (deep model)
        agents["aggressive"] = AggressiveDebatorAgent(self._fast_llm)
        agents["conservative"] = ConservativeDebatorAgent(self._fast_llm)
        agents["neutral"] = NeutralDebatorAgent(self._fast_llm)
        agents["portfolio_manager"] = PortfolioManagerAgent(self._llm)

        return agents

    def run(self, ticker: str, trade_date: str = "") -> AgentState:
        """Execute the full multi-agent pipeline."""
        if not trade_date:
            trade_date = datetime.date.today().isoformat()

        state = AgentState(ticker=ticker, trade_date=trade_date)

        # Layer 1: Analyst reports
        state = self._run_analysts(state)

        # Layer 2: Research debate → investment plan
        state = self._run_research_debate(state)

        # Layer 3: Trader → trade proposal
        state = self._agents["trader"].run(state)

        # Layer 4: Risk debate → final decision
        state = self._run_risk_debate(state)

        return state

    def _run_analysts(self, state: AgentState) -> AgentState:
        """Run all 4 analyst agents sequentially."""
        for name in ["technical", "fundamental", "sentiment", "macro"]:
            state = self._agents[name].run(state)
        return state

    def _run_research_debate(self, state: AgentState) -> AgentState:
        """Run Bull ↔ Bear debate then ResearchManager judgment."""
        max_rounds = self._config.agents.max_debate_rounds
        for _ in range(max_rounds):
            state = self._agents["bull"].run(state)
            state = self._agents["bear"].run(state)
        state = self._agents["research_manager"].run(state)
        return state

    def _run_risk_debate(self, state: AgentState) -> AgentState:
        """Run Aggressive ↔ Conservative ↔ Neutral debate then PortfolioManager."""
        max_rounds = self._config.agents.max_risk_discuss_rounds
        for _ in range(max_rounds):
            state = self._agents["aggressive"].run(state)
            state = self._agents["conservative"].run(state)
            state = self._agents["neutral"].run(state)
        state = self._agents["portfolio_manager"].run(state)
        return state
