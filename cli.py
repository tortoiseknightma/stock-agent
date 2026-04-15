"""
StockAgent CLI
==============
Command-line interface for the StockAgent investment agent.

Commands:
    status      Show agent and portfolio status
    analyze     Run analysis on portfolio or specific ticker
    buy         Execute a buy order
    sell        Execute a sell order
    report      Generate portfolio report
    review      Review trade journal
    memory      Manage long-term memory
    research    Search research corpus
    premarket   Run pre-market routine
    intraday    Run intraday check
    postmarket  Run post-market routine
    daemon      Run as scheduled daemon
    backtest    Run strategy backtest on historical data
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import load_config, TradingMode
from agent import StockAgentAgent


def main():
    parser = argparse.ArgumentParser(
        prog="stockagent",
        description="StockAgent — AI-Powered Investment Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stockagent status                     Show portfolio and agent status
  stockagent analyze                   Analyze all holdings
  stockagent analyze AAPL              Analyze specific ticker
  stockagent buy AAPL                  Buy AAPL based on analysis
  stockagent sell AAPL --qty 50        Sell 50 shares of AAPL
  stockagent report                    Generate daily report
  stockagent review --days 7           Review last 7 days of trades
  stockagent memory list               List all memories
  stockagent memory add "lesson text"  Add a memory entry
  stockagent research "AI semiconductors"  Search research corpus
  stockagent premarket                 Run pre-market routine
  stockagent daemon                    Run scheduled daemon
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # status
    subparsers.add_parser("status", help="Show agent and portfolio status")
    
    # analyze
    p = subparsers.add_parser("analyze", help="Run analysis")
    p.add_argument("ticker", nargs="?", help="Ticker to analyze (default: all)")
    p.add_argument("--execute", action="store_true", help="Execute top signals")
    p.add_argument("--agents", action="store_true",
                   help="Use multi-agent pipeline (12-agent system)")
    
    # buy
    p = subparsers.add_parser("buy", help="Execute buy order")
    p.add_argument("ticker", help="Ticker to buy")
    p.add_argument("--pct", type=float, help="Target position as %% of portfolio")
    p.add_argument("--qty", type=int, help="Number of shares")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    
    # sell
    p = subparsers.add_parser("sell", help="Execute sell order")
    p.add_argument("ticker", help="Ticker to sell")
    p.add_argument("--qty", type=int, help="Shares to sell (default: all)")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    
    # report
    subparsers.add_parser("report", help="Generate portfolio report")
    
    # review
    p = subparsers.add_parser("review", help="Review trade journal")
    p.add_argument("--days", type=int, default=7, help="Days to review")
    
    # memory
    p_mem = subparsers.add_parser("memory", help="Manage long-term memory")
    mem_sub = p_mem.add_subparsers(dest="memory_action")
    mem_sub.add_parser("list", help="List all memories")
    p_add = mem_sub.add_parser("add", help="Add memory entry")
    p_add.add_argument("content", help="Memory content")
    p_add.add_argument("--category", default="lesson",
                       choices=["preference", "correction", "strategy", "lesson", "skill"],
                       help="Memory category")
    p_add.add_argument("--importance", type=int, default=5, help="Importance 1-10")
    p_add.add_argument("--tags", nargs="*", help="Tags")
    mem_sub.add_parser("stats", help="Show memory statistics")
    
    # research
    p_res = subparsers.add_parser("research", help="Search research corpus")
    p_res.add_argument("query", help="Search query")
    p_res.add_argument("--tickers", nargs="*", help="Filter by tickers")
    p_res.add_argument("--limit", type=int, default=5, help="Max results")
    
    # routines
    subparsers.add_parser("premarket", help="Run pre-market routine")
    subparsers.add_parser("intraday", help="Run intraday check")
    subparsers.add_parser("postmarket", help="Run post-market routine")
    
    # daemon
    p_daemon = subparsers.add_parser("daemon", help="Run as scheduled daemon")
    p_daemon.add_argument("--once", action="store_true", help="Run once then exit")

    # backtest
    p_bt = subparsers.add_parser(
        "backtest", help="Run strategy backtest on historical data"
    )
    p_bt.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    p_bt.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p_bt.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p_bt.add_argument("--capital", type=float, default=100_000.0,
                      help="Initial capital (default: 100000)")
    p_bt.add_argument("--slippage", type=float, default=0.001,
                      help="Slippage fraction (default: 0.001)")
    p_bt.add_argument("--buy-threshold", type=float, default=0.4,
                      help="Composite score to trigger buy (default: 0.4)")
    p_bt.add_argument("--sell-threshold", type=float, default=-0.4,
                      help="Composite score to trigger sell (default: -0.4)")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # backtest runs without the full agent stack (no LLM keys required)
    if args.command == "backtest":
        cmd_backtest(args)
        return

    # Load agent
    try:
        agent = StockAgentAgent.from_config("config.yaml")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Make sure config.yaml exists in the current directory.")
        return

    # Execute command
    if args.command == "status":
        cmd_status(agent)
    elif args.command == "analyze":
        cmd_analyze(agent, args)
    elif args.command == "buy":
        cmd_buy(agent, args)
    elif args.command == "sell":
        cmd_sell(agent, args)
    elif args.command == "report":
        cmd_report(agent)
    elif args.command == "review":
        cmd_review(agent, args)
    elif args.command == "memory":
        cmd_memory(agent, args)
    elif args.command == "research":
        cmd_research(agent, args)
    elif args.command == "premarket":
        agent.run_premarket()
    elif args.command == "intraday":
        agent.run_intraday()
    elif args.command == "postmarket":
        agent.run_postmarket()
    elif args.command == "daemon":
        cmd_daemon(agent, args)


def cmd_status(agent: StockAgentAgent):
    """Show agent and portfolio status."""
    agent.connect()
    status = agent.get_status()
    portfolio = agent.broker.get_portfolio_summary()
    
    print("\n" + "="*55)
    print("  STOCKAGENT STATUS")
    print("="*55)
    
    print(f"\n  Trading Mode:   {status['trading_mode'].upper()}")
    print(f"  Connected:      {'✅' if status['connected'] else '❌'}")
    print(f"  Holdings:       {len(status['tickers'])} tickers")
    
    print(f"\n  Portfolio:")
    print(f"    Net Value:    ${portfolio['net_liquidation']:,.2f}")
    print(f"    Cash:         ${portfolio['total_cash']:,.2f}")
    print(f"    Day P&L:      ${portfolio['daily_pnl']:+,.2f}")
    print(f"    Unrealized:   ${portfolio['unrealized_pnl']:+,.2f}")
    
    if portfolio['positions']:
        print(f"\n  Positions:")
        print(f"    {'Ticker':<8} {'Shares':>8} {'Price':>10} {'Value':>12} {'P&L':>10}")
        for p in sorted(portfolio['positions'], key=lambda x: x['weight_pct'], reverse=True):
            pnl_str = f"${p['unrealized_pnl']:+,.0f}"
            print(f"    {p['ticker']:<8} {p['quantity']:>8} "
                  f"${p['current_price']:>9,.2f} ${p['market_value']:>11,.2f} {pnl_str:>10}")
    
    print(f"\n  Memory:   {status['memory']['total_active']} entries")
    print(f"  Corpus:   {status['corpus']['total_documents']} documents")
    print(f"  Journal:  {status['journal']['total_decisions']} decisions, "
          f"{status['journal']['total_trades']} trades")
    print()


def cmd_analyze(agent: StockAgentAgent, args):
    """Run analysis."""
    agent.connect()

    if getattr(args, "agents", False) and args.ticker:
        # Multi-agent pipeline
        print(f"\n[Multi-Agent] Analyzing {args.ticker}...")
        state = agent.run_multi_agent_analysis(args.ticker)
        signal_emoji = {"BUY": "🟢", "HOLD": "⚪", "SELL": "🔴"}.get(
            state.decision_signal, "⚪"
        )
        print(f"\n  {signal_emoji} {args.ticker}: {state.decision_signal}")
        print(f"  Conviction: {state.decision_conviction:.0%}")
        if state.trade_proposal:
            print(f"  Trade Proposal: {state.trade_proposal}")
        if state.final_decision:
            print(f"\n  Final Decision:\n  {state.final_decision[:500]}")
        print(f"\n  Pipeline log: {len(state.agent_log)} agent actions")
        return

    if args.ticker:
        # Single ticker
        print(f"\nAnalyzing {args.ticker}...")
        signal = agent.analyze_single(args.ticker)
        
        emoji = {"strong_buy": "🟢🟢", "buy": "🟢", "hold": "⚪",
                 "sell": "🔴", "strong_sell": "🔴🔴"}.get(signal.signal, "⚪")
        
        print(f"\n  {emoji} {args.ticker}: {signal.signal.upper()}")
        print(f"  Score: {signal.composite_score:+.3f}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Risk: {signal.risk_level}")
        print(f"\n  {signal.recommendation}")
        
        if signal.technical:
            print(f"\n  Technical: {signal.technical.score:+.2f} "
                  f"(MA:{signal.technical.ma_score:+.2f} RSI:{signal.technical.rsi_score:+.2f} "
                  f"MACD:{signal.technical.macd_score:+.2f})")
            print(f"    RSI: {signal.technical.indicators.get('rsi', 'N/A')}")
        
        if signal.fundamental:
            print(f"  Fundamental: {signal.fundamental.score:+.2f} "
                  f"(Val:{signal.fundamental.valuation_score:+.2f} "
                  f"Growth:{signal.fundamental.growth_score:+.2f})")
        
        if signal.sentiment:
            print(f"  Sentiment: {signal.sentiment.score:+.2f} "
                  f"({signal.sentiment.news_volume} articles)")
    else:
        # Full portfolio
        print("\nRunning full portfolio analysis...\n")
        results = agent.run_analysis_cycle()
        
        print(f"\nAnalyzed {len(results)} tickers.")


def cmd_buy(agent: StockAgentAgent, args):
    """Execute buy order."""
    agent.connect()
    
    if not args.yes:
        confirm = input(f"Buy {args.ticker}? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    signal = agent.analyze_single(args.ticker)
    result = agent.executor.execute_buy(args.ticker, signal, target_pct=args.pct / 100 if args.pct else None)
    print(f"\n{'✅' if result.success else '❌'} {result.message}")


def cmd_sell(agent: StockAgentAgent, args):
    """Execute sell order."""
    agent.connect()
    
    if not args.yes:
        confirm = input(f"Sell {args.ticker}? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    result = agent.execute_sell(args.ticker, args.qty)
    print(f"\n{'✅' if result.success else '❌'} {result.message}")


def cmd_report(agent: StockAgentAgent):
    """Generate portfolio report."""
    agent.connect()
    agent.run_analysis_cycle()
    portfolio = agent.broker.get_portfolio_summary()
    agent.notifier.send_daily_report(portfolio, agent._signals)


def cmd_review(agent: StockAgentAgent, args):
    """Review trade journal."""
    review = agent.get_trade_review(args.days)
    
    print(f"\n{'='*55}")
    print(f"  TRADE REVIEW — Last {args.days} days")
    print(f"{'='*55}")
    
    perf = review['performance']
    print(f"\n  Decisions: {perf['decisions']['total']} "
          f"({perf['decisions']['executed']} executed, "
          f"{perf['decisions']['execution_rate']:.0%} rate)")
    print(f"  Trades: {perf['trades']['total']} "
          f"({perf['trades']['buys']} buys, {perf['trades']['sells']} sells)")
    print(f"  Commission: ${perf['trades']['total_commission']:.2f}")
    
    acc = review['accuracy']
    print(f"  Accuracy: {acc['accuracy']:.0%} ({acc['correct']}/{acc['total_evaluated']})")
    
    if perf['decisions']['signal_distribution']:
        print(f"\n  Signal Distribution:")
        for sig, count in perf['decisions']['signal_distribution'].items():
            print(f"    {sig}: {count}")
    
    if review['recent_trades']:
        print(f"\n  Recent Trades:")
        for t in review['recent_trades'][:10]:
            ts = datetime.fromtimestamp(t['timestamp']).strftime('%m-%d %H:%M')
            print(f"    {ts} {t['side']:>4} {t['quantity']:>5} {t['ticker']:<6} @ ${t['price']:.2f}")
    print()


def cmd_memory(agent: StockAgentAgent, args):
    """Manage long-term memory."""
    if args.memory_action == "list":
        entries = agent.memory.get_all()
        print(f"\n{'='*55}")
        print(f"  LONG-TERM MEMORY ({len(entries)} entries)")
        print(f"{'='*55}")
        for e in entries:
            cat = e.category.value.upper().replace("_", " ")
            print(f"  [{cat}] (imp:{e.importance}) {e.content}")
            if e.tags:
                print(f"           Tags: {', '.join(e.tags)}")
        print()
    
    elif args.memory_action == "add":
        from core.memory.long_term import MemoryCategory
        cat_map = {
            "preference": MemoryCategory.USER_PREFERENCE,
            "correction": MemoryCategory.CORRECTION,
            "strategy": MemoryCategory.STRATEGY,
            "lesson": MemoryCategory.LESSON,
            "skill": MemoryCategory.SKILL,
        }
        cat = cat_map.get(args.category, MemoryCategory.LESSON)
        agent.memory.add(args.content, cat, args.importance, args.tags or [])
        print(f"✅ Memory added: [{args.category}] {args.content}")
    
    elif args.memory_action == "stats":
        stats = agent.memory.get_stats()
        print(f"\nMemory Statistics:")
        print(f"  Total active: {stats['total_active']}")
        print(f"  Avg importance: {stats['avg_importance']}")
        print(f"  By category: {json.dumps(stats['by_category'], indent=4)}")


def cmd_research(agent: StockAgentAgent, args):
    """Search research corpus."""
    docs = agent.search_research(args.query, args.tickers)
    
    print(f"\n{'='*55}")
    print(f"  RESEARCH CORPUS — {len(docs)} results for '{args.query}'")
    print(f"{'='*55}")
    
    for doc in docs[:args.limit]:
        print(f"\n  [{doc.doc_type.value}] {doc.title}")
        if doc.tickers:
            print(f"    Tickers: {', '.join(doc.tickers)}")
        if doc.summary:
            print(f"    Summary: {doc.summary[:150]}...")
        print(f"    Source: {doc.source}")
    print()


def cmd_daemon(agent: StockAgentAgent, args):
    """Run as scheduled daemon."""
    import time
    
    print("StockAgent Daemon starting...")
    print("Press Ctrl+C to stop.\n")
    
    agent.connect()
    
    try:
        while True:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            weekday = now.weekday()  # 0=Monday
            
            # Skip weekends
            if weekday >= 5:
                print(f"[{now.strftime('%H:%M')}] Weekend — sleeping 1 hour")
                time.sleep(3600)
                continue
            
            # Pre-market at 09:00
            if hour == 9 and minute == 0:
                agent.run_premarket()
                time.sleep(60)  # Skip the rest of this minute
            
            # Intraday every 30 min (09:30-16:00)
            elif 9 <= hour < 16 and minute in (0, 30):
                if hour >= 9 and (hour > 9 or minute >= 30):
                    agent.run_intraday()
                    time.sleep(60)
            
            # Post-market at 16:30
            elif hour == 16 and minute == 30:
                agent.run_postmarket()
                time.sleep(60)
            
            else:
                time.sleep(30)  # Check every 30 seconds
            
            if args.once:
                break
                
    except KeyboardInterrupt:
        print("\nDaemon stopped.")
        agent.disconnect()


def cmd_backtest(args):
    """Run strategy backtest on historical data. Does not require LLM API keys."""
    from backtest.engine import BacktestEngine, BacktestConfig
    from backtest.metrics import compute_metrics, format_report

    cfg = BacktestConfig(
        ticker=args.ticker.upper(),
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        slippage_pct=args.slippage,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    engine = BacktestEngine(cfg)
    result = engine.run()

    if not result.daily_values:
        print("No data returned — check the ticker and date range.")
        return

    metrics = compute_metrics(result)
    print(format_report(metrics, result))


if __name__ == "__main__":
    main()
