"""Tests for TradeJournal — record, retrieve, performance review."""

import pytest
import time
from core.memory.trade_journal import (
    TradeJournal, DecisionRecord, TradeRecord,
    AnalysisSnapshot, Signal, OrderSide, OrderType,
)


def make_journal(tmp_path):
    return TradeJournal(str(tmp_path / "test_journal.db"))


def make_decision(ticker="AAPL", signal=Signal.BUY, executed=True,
                  composite_score=0.6, price=180.0) -> DecisionRecord:
    snap = AnalysisSnapshot(
        technical_score=0.5, fundamental_score=0.4,
        sentiment_score=0.3, composite_score=composite_score,
        signal=signal, reasoning="Test reasoning",
    )
    return DecisionRecord(
        ticker=ticker, signal=signal, analysis=snap,
        portfolio_value=100_000.0, current_position=0,
        current_price=price, intended_action="buy 10 shares",
        confidence=0.7, risk_assessment="acceptable",
        executed=executed, execution_reason="risk approved",
    )


def make_trade(ticker="AAPL", side=OrderSide.BUY, qty=10,
               price=180.0, decision_id=1) -> TradeRecord:
    return TradeRecord(
        ticker=ticker, side=side, order_type=OrderType.MARKET,
        quantity=qty, price=price,
        portfolio_value_before=100_000.0, decision_id=decision_id,
        commission=1.0,
    )


# ---------------------------------------------------------------------------
# Decision recording
# ---------------------------------------------------------------------------

class TestDecisionRecording:
    def test_record_decision_returns_id(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        assert isinstance(dec_id, int)
        assert dec_id > 0

    def test_multiple_decisions_get_unique_ids(self, tmp_path):
        j = make_journal(tmp_path)
        id1 = j.record_decision(make_decision())
        id2 = j.record_decision(make_decision())
        assert id1 != id2

    def test_get_decisions_all(self, tmp_path):
        j = make_journal(tmp_path)
        j.record_decision(make_decision("AAPL"))
        j.record_decision(make_decision("MSFT"))
        decisions = j.get_decisions()
        assert len(decisions) >= 2

    def test_get_decisions_by_ticker(self, tmp_path):
        j = make_journal(tmp_path)
        j.record_decision(make_decision("AAPL"))
        j.record_decision(make_decision("MSFT"))
        decisions = j.get_decisions(ticker="AAPL")
        assert all(d["ticker"] == "AAPL" for d in decisions)

    def test_get_decisions_by_signal(self, tmp_path):
        j = make_journal(tmp_path)
        j.record_decision(make_decision(signal=Signal.BUY))
        j.record_decision(make_decision(signal=Signal.SELL))
        decisions = j.get_decisions(signal=Signal.BUY)
        assert all(d["signal"] == "buy" for d in decisions)

    def test_decision_executed_flag(self, tmp_path):
        j = make_journal(tmp_path)
        j.record_decision(make_decision(executed=True))
        j.record_decision(make_decision(executed=False))
        all_dec = j.get_decisions()
        executed = [d for d in all_dec if d["executed"]]
        blocked = [d for d in all_dec if not d["executed"]]
        assert len(executed) >= 1
        assert len(blocked) >= 1


# ---------------------------------------------------------------------------
# Trade recording
# ---------------------------------------------------------------------------

class TestTradeRecording:
    def test_record_trade_returns_id(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        trade_id = j.record_trade(make_trade(decision_id=dec_id))
        assert isinstance(trade_id, int)
        assert trade_id > 0

    def test_get_trades_all(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        j.record_trade(make_trade(side=OrderSide.BUY, decision_id=dec_id))
        j.record_trade(make_trade(side=OrderSide.SELL, decision_id=dec_id))
        trades = j.get_trades()
        assert len(trades) >= 2

    def test_get_trades_by_ticker(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        j.record_trade(make_trade("AAPL", decision_id=dec_id))
        j.record_trade(make_trade("MSFT", decision_id=dec_id))
        trades = j.get_trades(ticker="AAPL")
        assert all(t["ticker"] == "AAPL" for t in trades)

    def test_get_trades_by_side(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        j.record_trade(make_trade(side=OrderSide.BUY, decision_id=dec_id))
        j.record_trade(make_trade(side=OrderSide.SELL, decision_id=dec_id))
        buys = j.get_trades(side=OrderSide.BUY)
        assert all(t["side"] == "buy" for t in buys)

    def test_trade_data_stored_correctly(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        j.record_trade(make_trade(ticker="AAPL", qty=25, price=182.50,
                                  decision_id=dec_id))
        trades = j.get_trades(ticker="AAPL")
        t = trades[0]
        assert t["ticker"] == "AAPL"
        assert t["quantity"] == 25
        assert t["price"] == pytest.approx(182.50)


# ---------------------------------------------------------------------------
# Outcome updates
# ---------------------------------------------------------------------------

class TestOutcomeUpdate:
    def test_update_trade_outcome(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        trade_id = j.record_trade(make_trade(decision_id=dec_id))
        # Update with a higher price → profitable
        j.update_trade_outcome(trade_id, current_price=200.0)
        # No exception = success

    def test_update_nonexistent_trade_no_crash(self, tmp_path):
        j = make_journal(tmp_path)
        j.update_trade_outcome(9999, current_price=200.0)  # must not raise


# ---------------------------------------------------------------------------
# Performance review
# ---------------------------------------------------------------------------

class TestPerformanceReview:
    def test_performance_review_structure(self, tmp_path):
        j = make_journal(tmp_path)
        dec_id = j.record_decision(make_decision())
        j.record_trade(make_trade(decision_id=dec_id))
        review = j.get_performance_review(days=30)
        assert "decisions" in review
        assert "trades" in review

    def test_performance_counts(self, tmp_path):
        j = make_journal(tmp_path)
        d1 = j.record_decision(make_decision(executed=True))
        j.record_decision(make_decision(executed=False))
        j.record_trade(make_trade(side=OrderSide.BUY, decision_id=d1))
        j.record_trade(make_trade(side=OrderSide.SELL, decision_id=d1))
        review = j.get_performance_review(days=30)
        assert review["decisions"]["total"] >= 2
        assert review["trades"]["buys"] >= 1
        assert review["trades"]["sells"] >= 1

    def test_decision_accuracy_structure(self, tmp_path):
        j = make_journal(tmp_path)
        acc = j.get_decision_accuracy(days=30)
        assert "accuracy" in acc
        assert "total_evaluated" in acc


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_structure(self, tmp_path):
        j = make_journal(tmp_path)
        stats = j.get_stats()
        assert "total_decisions" in stats
        assert "total_trades" in stats

    def test_stats_count_increases(self, tmp_path):
        j = make_journal(tmp_path)
        before = j.get_stats()["total_decisions"]
        j.record_decision(make_decision())
        after = j.get_stats()["total_decisions"]
        assert after == before + 1
