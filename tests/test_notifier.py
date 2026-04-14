"""Tests for Notifier — channel routing, CLI output, file output."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from push.notifier import Notifier, Notification
from core.config import PushConfig


def make_notifier(tmp_path, channels=None):
    cfg = PushConfig(channels=channels or ["cli"])
    n = Notifier(cfg)
    n._log_dir = tmp_path / "reports"
    n._log_dir.mkdir(parents=True, exist_ok=True)
    return n


def make_notification(title="Test", body="Test body",
                      level="info", channels=None):
    return Notification(title=title, body=body, level=level,
                        channels=channels or ["cli"])


# ---------------------------------------------------------------------------
# Notification dataclass
# ---------------------------------------------------------------------------

class TestNotificationDataclass:
    def test_default_channels_cli(self):
        n = Notification(title="T", body="B")
        assert n.channels == ["cli"]

    def test_level_default_info(self):
        n = Notification(title="T", body="B")
        assert n.level == "info"

    def test_custom_channels(self):
        n = Notification(title="T", body="B", channels=["feishu", "file"])
        assert "feishu" in n.channels


# ---------------------------------------------------------------------------
# CLI channel (just checks it doesn't crash — stdout is side-effecting)
# ---------------------------------------------------------------------------

class TestCLIChannel:
    def test_send_cli_does_not_raise(self, tmp_path, capsys):
        n = make_notifier(tmp_path, channels=["cli"])
        n.send(make_notification())
        # If we get here, it didn't raise
        out = capsys.readouterr().out
        # CLI output should contain the title
        assert "Test" in out

    def test_send_trade_level_logged(self, tmp_path, capsys):
        n = make_notifier(tmp_path, channels=["cli"])
        n.send(Notification(title="BUY AAPL", body="Bought 10 shares",
                            level="trade", channels=["cli"]))
        out = capsys.readouterr().out
        assert "AAPL" in out or "BUY" in out or "Bought" in out


# ---------------------------------------------------------------------------
# File channel
# ---------------------------------------------------------------------------

class TestFileChannel:
    def test_send_file_creates_log(self, tmp_path):
        n = make_notifier(tmp_path, channels=["file"])
        notif = make_notification(channels=["file"])
        n.send(notif)
        log = tmp_path / "reports" / "notifications.log"
        assert log.exists()

    def test_file_log_contains_title(self, tmp_path):
        n = make_notifier(tmp_path, channels=["file"])
        notif = make_notification(title="UniqueTitle999", channels=["file"])
        n.send(notif)
        log = tmp_path / "reports" / "notifications.log"
        content = log.read_text(encoding="utf-8")
        assert "UniqueTitle999" in content


# ---------------------------------------------------------------------------
# Trade alert helper
# ---------------------------------------------------------------------------

class TestTradeAlert:
    def test_send_trade_alert_does_not_raise(self, tmp_path, capsys):
        n = make_notifier(tmp_path, channels=["cli"])
        n.send_trade_alert(
            ticker="AAPL", side="BUY", quantity=10,
            price=182.50, reason="Executed successfully"
        )
        out = capsys.readouterr().out
        assert "AAPL" in out or "BUY" in out or "182" in out

    def test_send_risk_alert_does_not_raise(self, tmp_path, capsys):
        n = make_notifier(tmp_path, channels=["cli"])
        n.send_risk_alert("Position limit exceeded for TSLA")
        out = capsys.readouterr().out
        assert out  # something printed


# ---------------------------------------------------------------------------
# Daily report
# ---------------------------------------------------------------------------

class TestDailyReport:
    def test_daily_report_does_not_raise(self, tmp_path, capsys):
        n = make_notifier(tmp_path, channels=["cli"])
        portfolio = {
            "net_liquidation": 105_000.0,
            "total_cash": 50_000.0,
            "daily_pnl": 1_500.0,
            "unrealized_pnl": 2_000.0,
            "positions": [
                {"ticker": "AAPL", "quantity": 50, "current_price": 182.5,
                 "market_value": 9125.0, "unrealized_pnl": 250.0,
                 "weight_pct": 8.7},
            ],
        }
        signals = {}
        n.send_daily_report(portfolio, signals)
        out = capsys.readouterr().out
        assert out  # something printed


# ---------------------------------------------------------------------------
# Channel fallback
# ---------------------------------------------------------------------------

class TestChannelFallback:
    def test_unknown_channel_does_not_crash(self, tmp_path, capsys):
        cfg = PushConfig(channels=["cli"])
        n = Notifier(cfg)
        # Manually send to a nonsense channel — should just skip
        notif = Notification(title="T", body="B", channels=["nonexistent_channel"])
        n.send(notif)  # must not raise

    def test_feishu_without_webhook_skips_gracefully(self, tmp_path):
        cfg = PushConfig(channels=["feishu"], feishu_webhook=None)
        n = Notifier(cfg)
        # send_feishu should skip silently when webhook is None
        n.send(Notification(title="T", body="B", channels=["feishu"]))
