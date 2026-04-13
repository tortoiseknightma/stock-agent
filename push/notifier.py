"""
StockAgent Notification System
===============================
Multi-channel alert and report delivery.

Supported channels:
- CLI (terminal output)
- Feishu/Lark webhook
- Telegram bot
- Email (SMTP)
- File (local markdown reports)

Generates:
- Daily portfolio reports
- Trade alerts
- Risk warnings
- Weekly performance reviews
"""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from core.config import PushConfig


@dataclass
class Notification:
    """A notification to be delivered."""
    title: str
    body: str
    level: str = "info"        # info, warning, alert, trade
    channels: List[str] = None
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = ["cli"]


class Notifier:
    """
    Multi-channel notification delivery system.
    
    Usage:
        notifier = Notifier(config)
        
        # Send a trade alert
        notifier.send(Notification(
            title="BUY: AAPL",
            body="Bought 50 shares @ $178.50",
            level="trade"
        ))
        
        # Generate and send daily report
        notifier.send_daily_report(portfolio_summary, signals)
    """
    
    def __init__(self, config: PushConfig):
        self.config = config
        self.channels = config.channels or ["cli"]
        self._log_dir = Path("reports")
        self._log_dir.mkdir(parents=True, exist_ok=True)
    
    def send(self, notification: Notification):
        """Send notification through configured channels."""
        channels = notification.channels or self.channels
        
        for channel in channels:
            try:
                if channel == "cli":
                    self._send_cli(notification)
                elif channel == "feishu":
                    self._send_feishu(notification)
                elif channel == "telegram":
                    self._send_telegram(notification)
                elif channel == "file":
                    self._send_file(notification)
            except Exception as e:
                print(f"Notification error ({channel}): {e}")
    
    def send_daily_report(self, portfolio: Dict, signals: Dict,
                          risk_warnings: List = None):
        """Generate and send a formatted daily portfolio report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        lines = [
            f"{'='*60}",
            f"  STOCKAGENT DAILY REPORT — {now}",
            f"{'='*60}",
            "",
            "PORTFOLIO SUMMARY",
            f"  Net Value:  ${portfolio.get('net_liquidation', 0):,.2f}",
            f"  Cash:       ${portfolio.get('total_cash', 0):,.2f}",
            f"  Day P&L:    ${portfolio.get('daily_pnl', 0):+,.2f}",
            f"  Unrealized: ${portfolio.get('unrealized_pnl', 0):+,.2f}",
            f"  Positions:  {portfolio.get('num_positions', 0)}",
            "",
        ]
        
        # Positions table
        positions = portfolio.get("positions", [])
        if positions:
            lines.append("HOLDINGS")
            lines.append(f"  {'Ticker':<8} {'Shares':>8} {'Price':>10} {'Value':>12} {'P&L':>10} {'Weight':>8}")
            lines.append(f"  {'-'*58}")
            for p in sorted(positions, key=lambda x: x.get("weight_pct", 0), reverse=True):
                pnl = p.get("unrealized_pnl", 0)
                pnl_str = f"${pnl:+,.0f}"
                lines.append(
                    f"  {p['ticker']:<8} {p['quantity']:>8} "
                    f"${p['current_price']:>9,.2f} ${p['market_value']:>11,.2f} "
                    f"{pnl_str:>10} {p['weight_pct']:>7.1f}%"
                )
            lines.append("")
        
        # Signals
        if signals:
            lines.append("ANALYSIS SIGNALS")
            for ticker, sig in signals.items():
                if hasattr(sig, 'signal'):
                    emoji = {"strong_buy": "🟢🟢", "buy": "🟢", "hold": "⚪",
                             "sell": "🔴", "strong_sell": "🔴🔴"}.get(sig.signal, "⚪")
                    lines.append(f"  {emoji} {ticker:<8} {sig.signal:<12} (score: {sig.composite_score:+.2f})")
            lines.append("")
        
        # Risk warnings
        if risk_warnings:
            lines.append("RISK WARNINGS")
            for w in risk_warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")
        
        lines.append(f"{'='*60}")
        
        report = "\n".join(lines)
        
        # Send to channels
        self.send(Notification(
            title="Daily Portfolio Report",
            body=report,
            level="info",
            channels=["cli", "file"],
        ))
        
        # Save to file
        filename = f"daily_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        (self._log_dir / "daily").mkdir(exist_ok=True)
        (self._log_dir / "daily" / filename).write_text(report)
    
    def send_trade_alert(self, ticker: str, side: str, quantity: int,
                         price: float, reason: str = ""):
        """Send a trade execution alert."""
        emoji = "🟢" if side == "BUY" else "🔴"
        body = f"{emoji} {side} {quantity} {ticker} @ ${price:.2f}"
        if reason:
            body += f"\n  Reason: {reason}"
        
        self.send(Notification(
            title=f"Trade: {side} {ticker}",
            body=body,
            level="trade",
        ))
    
    def send_risk_alert(self, message: str):
        """Send a risk warning alert."""
        self.send(Notification(
            title="⚠ RISK ALERT",
            body=message,
            level="warning",
        ))
    
    # --- Channel Implementations ---
    
    def _send_cli(self, notification: Notification):
        """Print to terminal."""
        print(f"\n[{notification.level.upper()}] {notification.title}")
        print(notification.body)
        print()
    
    def _send_feishu(self, notification: Notification):
        """Send to Feishu/Lark webhook."""
        webhook_url = self.config.feishu_webhook
        if not webhook_url:
            return
        
        import requests
        
        # Feishu webhook format
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": notification.title},
                    "template": "blue" if notification.level == "info" else "red",
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": notification.body}
                    }
                ],
            },
        }
        
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if not resp.ok:
                print(f"Feishu webhook failed: {resp.status_code}")
        except Exception as e:
            print(f"Feishu error: {e}")
    
    def _send_telegram(self, notification: Notification):
        """Send to Telegram bot."""
        token = self.config.telegram_bot_token
        chat_id = self.config.telegram_chat_id
        if not token or not chat_id:
            return
        
        import requests
        
        text = f"*{notification.title}*\n\n{notification.body}"
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if not resp.ok:
                print(f"Telegram failed: {resp.status_code}")
        except Exception as e:
            print(f"Telegram error: {e}")
    
    def _send_file(self, notification: Notification):
        """Append to notification log file."""
        log_file = self._log_dir / "notifications.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, "a") as f:
            f.write(f"\n[{timestamp}] [{notification.level}] {notification.title}\n")
            f.write(f"{notification.body}\n")
            f.write("-" * 40 + "\n")
