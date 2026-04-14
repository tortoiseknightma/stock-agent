"""
StockAgent Main Agent
=====================
The central orchestrator that ties everything together.

This is the main entry point for running the investment agent.
It coordinates:
- Data collection (market data, news, fundamentals)
- Analysis (technical, fundamental, sentiment → composite)
- Risk management (pre-trade checks, stop-losses)
- Trade execution (buy/sell with journal recording)
- Notifications (alerts, daily reports)
- Memory management (long-term memory, research corpus, trade journal)

Architecture:
    ┌──────────────┐
    │   Scheduler   │ ← cron / manual trigger
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Agent Core  │ ← this file
    └──┬───┬───┬───┘
       │   │   │
  ┌────▼┐ ┌▼───▼────┐ ┌──────▼──────┐
  │Data │ │Analysis │ │  Execution  │
  │Sources│ │Engine  │ │  (Risk+E)   │
  └─────┘ └────────┘ └─────────────┘
       │   │   │
  ┌────▼───▼───▼────┐
  │ Memory System   │
  │ (LTM+Corpus+TJ) │
  └─────────────────┘
"""

import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from core.config import AppConfig, TradingMode, load_config
from core.broker import create_broker, BaseBroker
from core.memory.long_term import LongTermMemory, MemoryCategory
from core.memory.research_corpus import ResearchCorpus, DocType, ResearchDoc
from core.memory.trade_journal import TradeJournal
from data.sources.market_data import MarketDataProvider
from data.sources.news import NewsProvider
from data.sources.fundamentals import FundamentalsProvider
from analysis.technical.technical import TechnicalAnalyzer
from analysis.fundamental.fundamental import FundamentalAnalyzer
from analysis.sentiment.sentiment import SentimentAnalyzer
from analysis.composite.composite import CompositeAnalyzer, CompositeSignal
from analysis.llm.news_analyzer import LLMNewsAnalyzer
from analysis.llm.thesis_generator import ThesisGenerator
from analysis.llm.earnings_analyzer import EarningsAnalyzer
from analysis.llm.risk_assessor import LLMRiskAssessor
from execution.risk_engine import RiskEngine
from execution.executor import TradeExecutor
from push.notifier import Notifier, Notification
from core.llm import create_llm_client, BaseLLMClient


class StockAgentAgent:
    """
    The main StockAgent investment agent.
    
    Usage:
        agent = StockAgentAgent.from_config("config.yaml")
        agent.connect()
        
        # Run a full analysis cycle
        results = agent.run_analysis_cycle()
        
        # Run pre-market routine
        agent.run_premarket()
        
        # Run post-market routine  
        agent.run_postmarket()
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Core components
        self.broker: BaseBroker = create_broker(config)
        self.memory = LongTermMemory(config.memory.memory_db_path)
        self.corpus = ResearchCorpus(config.memory.research_index_path)
        self.journal = TradeJournal("data/trade_journal.db")
        
        # Data providers
        self.market_data = MarketDataProvider()
        self.news = NewsProvider()
        self.fundamentals = FundamentalsProvider()
        
        # LLM client (lazy-init, None if provider not available)
        self.llm: Optional[BaseLLMClient] = None
        self._init_llm()

        # LLM-powered analysis components (None when LLM unavailable)
        self.llm_news: Optional[LLMNewsAnalyzer] = None
        self.llm_thesis: Optional[ThesisGenerator] = None
        self.llm_earnings: Optional[EarningsAnalyzer] = None
        self.llm_risk: Optional[LLMRiskAssessor] = None
        self._init_llm_analyzers()

        # Analysis (wire LLM components in)
        self.technical = TechnicalAnalyzer(config.analysis)
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment = SentimentAnalyzer(llm_analyzer=self.llm_news)
        self.composite = CompositeAnalyzer(config.analysis,
                                           thesis_generator=self.llm_thesis)

        # Execution
        self.risk = RiskEngine(config.risk, self.broker)
        self.executor = TradeExecutor(config, self.broker, self.risk, self.journal)
        
        # Notifications
        self.notifier = Notifier(config.push)
        
        # State
        self._tickers: List[str] = []
        self._signals: Dict[str, CompositeSignal] = {}
    
    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "StockAgentAgent":
        """Create agent from configuration file."""
        config = load_config(config_path)
        return cls(config)

    def _init_llm(self):
        """Initialize LLM client if API key is available."""
        try:
            self.llm = create_llm_client(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                base_url=self.config.llm.base_url,
            )
        except Exception:
            self.llm = None  # Graceful fallback: agent works without LLM

    def _init_llm_analyzers(self):
        """Wire LLM client into analysis components (no-op if LLM unavailable)."""
        if self.llm is None:
            return
        self.llm_news = LLMNewsAnalyzer(self.llm)
        self.llm_thesis = ThesisGenerator(self.llm)
        self.llm_earnings = EarningsAnalyzer(self.llm)
        self.llm_risk = LLMRiskAssessor(self.llm)

    def get_llm(self) -> Optional[BaseLLMClient]:
        """Get LLM client, or None if not available."""
        return self.llm

    def connect(self) -> bool:
        """Connect to broker."""
        success = self.broker.connect()
        if success:
            self.risk.reset_daily()
            self._load_portfolio_tickers()
        return success
    
    def disconnect(self):
        """Disconnect from broker."""
        self.broker.disconnect()
    
    # === Main Routines ===
    
    def run_premarket(self) -> Dict[str, Any]:
        """
        Pre-market routine (run at ~09:00 ET).
        
        1. Connect and sync portfolio
        2. Run full analysis on all holdings
        3. Collect overnight news
        4. Generate pre-market briefing
        """
        print("\n" + "="*50)
        print("  STOCKAGENT PRE-MARKET BRIEFING")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*50 + "\n")
        
        if not self.connect():
            return {"error": "Failed to connect to broker"}
        
        self.risk.reset_daily()
        self._load_portfolio_tickers()
        
        # Run analysis
        results = self.run_analysis_cycle()
        
        # Check risk status
        risk_warnings = self.risk.check_portfolio_health()
        
        # Send report
        portfolio = self.broker.get_portfolio_summary()
        self.notifier.send_daily_report(portfolio, self._signals, risk_warnings)
        
        return results
    
    def run_intraday(self) -> Dict[str, Any]:
        """
        Intraday check (run every 30 min during market hours).
        
        1. Quick price update
        2. Check stop-losses and take-profits
        3. Re-run analysis if significant price moves
        4. Execute urgent trades
        """
        if not self.broker.is_connected():
            self.connect()
        
        results = {
            "timestamp": time.time(),
            "stop_losses": [],
            "take_profits": [],
            "trades": [],
        }
        
        # Check stop-losses
        sl_results = self.executor.check_stop_losses()
        for r in sl_results:
            if r.success:
                results["stop_losses"].append(r.message)
                self.notifier.send_trade_alert(
                    r.order_result.order.ticker, "SELL",
                    r.order_result.order.quantity,
                    r.order_result.order.filled_price,
                    "stop-loss triggered"
                )
        
        # Check take-profits
        tp_results = self.executor.check_take_profits()
        for r in tp_results:
            if r.success:
                results["take_profits"].append(r.message)
                self.notifier.send_trade_alert(
                    r.order_result.order.ticker, "SELL",
                    r.order_result.order.quantity,
                    r.order_result.order.filled_price,
                    "take-profit triggered"
                )
        
        # Portfolio health check
        warnings = self.risk.check_portfolio_health()
        for w in warnings:
            self.notifier.send_risk_alert(str(w))
        
        return results
    
    def run_postmarket(self) -> Dict[str, Any]:
        """
        Post-market routine (run at ~16:30 ET).
        
        1. Final analysis cycle
        2. Performance review
        3. Trade journal review
        4. Update long-term memory with lessons
        5. Generate daily report
        """
        print("\n" + "="*50)
        print("  STOCKAGENT POST-MARKET REVIEW")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*50 + "\n")
        
        if not self.broker.is_connected():
            self.connect()
        
        # Final analysis
        results = self.run_analysis_cycle()
        
        # Performance review
        review = self.journal.get_performance_review(days=1)
        accuracy = self.journal.get_decision_accuracy(days=7)
        
        # Portfolio summary
        portfolio = self.broker.get_portfolio_summary()
        
        # Send final report
        self.notifier.send_daily_report(portfolio, self._signals)
        
        # Store daily analysis in research corpus
        self._store_daily_analysis(results)
        
        # Print summary
        print(f"\nDaily Performance:")
        print(f"  Trades today: {review['trades']['total']}")
        print(f"  Commission paid: ${review['trades']['total_commission']:.2f}")
        print(f"  7-day accuracy: {accuracy['accuracy']:.1%}")
        
        return results
    
    def run_analysis_cycle(self) -> Dict[str, Any]:
        """
        Core analysis cycle: analyze all holdings and generate signals.
        """
        if not self._tickers:
            self._load_portfolio_tickers()
        
        results = {}
        self._signals = {}
        
        for ticker in self._tickers:
            try:
                signal = self._analyze_ticker(ticker)
                self._signals[ticker] = signal
                results[ticker] = signal.to_dict()
                
                # Print signal
                emoji = {"strong_buy": "🟢🟢", "buy": "🟢", "hold": "⚪",
                         "sell": "🔴", "strong_sell": "🔴🔴"}.get(signal.signal, "⚪")
                print(f"  {emoji} {ticker:<8} {signal.signal:<12} "
                      f"score={signal.composite_score:+.2f} "
                      f"conf={signal.confidence:.0%}")
                
            except Exception as e:
                print(f"  ⚠ {ticker}: Analysis error — {e}")
                results[ticker] = {"error": str(e)}
        
        return results
    
    def analyze_single(self, ticker: str) -> CompositeSignal:
        """Analyze a single ticker (for CLI use)."""
        return self._analyze_ticker(ticker)
    
    # === Trading Actions ===
    
    def execute_buy(self, ticker: str, target_pct: float = None):
        """Execute a buy based on current analysis."""
        signal = self._signals.get(ticker) or self._analyze_ticker(ticker)
        account = self.broker.get_account_info()
        result = self.executor.execute_buy(ticker, signal, account.net_liquidation, target_pct)
        self.notifier.send(Notification(
            title=f"{'✅' if result.success else '❌'} {result.message}",
            body=result.risk_check.reasoning,
            level="trade" if result.success else "warning"
        ))
        return result
    
    def execute_sell(self, ticker: str, quantity: int = None):
        """Execute a sell."""
        signal = self._signals.get(ticker)
        result = self.executor.execute_sell(ticker, signal, quantity)
        self.notifier.send(Notification(
            title=f"{'✅' if result.success else '❌'} {result.message}",
            body=result.risk_check.reasoning,
            level="trade" if result.success else "warning"
        ))
        return result
    
    # === Memory Management ===
    
    def add_memory(self, content: str, category: str = "lesson", importance: int = 5, tags: List[str] = None):
        """Add an entry to long-term memory."""
        cat = MemoryCategory(category)
        return self.memory.add(content, cat, importance, tags)
    
    def search_research(self, query: str, tickers: List[str] = None) -> List:
        """Search the research corpus."""
        return self.corpus.search(query, tickers=tickers)
    
    def get_trade_review(self, days: int = 7) -> Dict:
        """Get trade journal review."""
        return {
            "performance": self.journal.get_performance_review(days),
            "accuracy": self.journal.get_decision_accuracy(days),
            "recent_decisions": self.journal.get_decisions(days=days),
            "recent_trades": self.journal.get_trades(days=days),
        }
    
    # === Internal ===
    
    def _analyze_ticker(self, ticker: str) -> CompositeSignal:
        """Run full analysis on a single ticker."""
        # 1. Get price history
        history = self.market_data.get_history(ticker, period="6mo")
        
        # 2. Technical analysis
        if history:
            tech_signal = self.technical.analyze(history)
            tech_signal.ticker = ticker
        else:
            tech_signal = None
        
        # 3. Fundamental analysis
        ratios = self.fundamentals.get_ratios(ticker)
        info = self.market_data.get_company_info(ticker)
        sector = info.get("sector", "")
        fund_signal = self.fundamental_analyzer.analyze(ratios, sector)
        fund_signal.ticker = ticker
        
        # 4. Sentiment analysis
        news_items = self.news.get_ticker_news(ticker, limit=10)
        sent_signal = self.sentiment.analyze(news_items)
        sent_signal.ticker = ticker
        
        # 5. Research context
        research_context = self.corpus.get_context_for_ticker(ticker, max_chars=500)
        
        # 6. Long-term memory context
        memory_text = self.memory.get_injection_text()
        
        # 7. Composite signal
        signal = self.composite.analyze(ticker, tech_signal, fund_signal, sent_signal)
        
        return signal
    
    def _load_portfolio_tickers(self):
        """Load tickers from current portfolio."""
        if self.broker.is_connected():
            positions = self.broker.get_positions()
            self._tickers = [p.ticker for p in positions]
        else:
            # Use default portfolio from config
            self._tickers = list(self.config.simulated.default_portfolio.keys())
    
    def _store_daily_analysis(self, results: Dict):
        """Store daily analysis results in research corpus."""
        for ticker, data in results.items():
            if "error" not in data:
                doc = ResearchDoc(
                    title=f"Daily Analysis: {ticker} — {datetime.now().strftime('%Y-%m-%d')}",
                    content=json.dumps(data, indent=2, default=str),
                    doc_type=DocType.AGENT_ANALYSIS,
                    tickers=[ticker],
                    summary=f"Signal: {data.get('signal', 'unknown')}, Score: {data.get('composite_score', 0):.2f}",
                )
                self.corpus.add(doc)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status summary."""
        return {
            "trading_mode": self.config.trading_mode.value,
            "connected": self.broker.is_connected(),
            "tickers": self._tickers,
            "memory": self.memory.get_stats(),
            "corpus": self.corpus.get_stats(),
            "journal": self.journal.get_stats(),
        }
