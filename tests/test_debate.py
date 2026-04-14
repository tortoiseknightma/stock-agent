"""Tests for DebateEngine — Bull/Bear adversarial debate mechanism."""

import json
import pytest
from unittest.mock import MagicMock, patch
from analysis.llm.debate_engine import (
    DebateEngine, DebateResult, DebateRound,
    _BULL_SYSTEM, _BEAR_SYSTEM, _JUDGE_SYSTEM,
)
from core.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_client(responses: list) -> MagicMock:
    """Return a mock BaseLLMClient whose chat_simple yields successive strings."""
    client = MagicMock()
    client.chat_simple.side_effect = responses
    return client


def make_judge_json(**overrides) -> str:
    """Build a minimal valid judge JSON string."""
    data = {
        "signal": "buy",
        "conviction": 0.72,
        "bull_score": 0.7,
        "bear_score": 0.55,
        "deciding_factors": ["Strong technicals", "Positive momentum"],
        "debate_summary": "Bull case was more convincing.",
    }
    data.update(overrides)
    return json.dumps(data)


def make_engine_with_canned_responses(**verdict_overrides) -> tuple:
    """
    Return (engine, mock_client) pre-loaded with 5 canned responses:
    bull_opening, bear_counter, bull_rebuttal, bear_rebuttal, judge_verdict.
    """
    judge_json = make_judge_json(**verdict_overrides)
    responses = [
        "Bull opening: strong RSI, positive momentum.",       # round 1
        "Bear counter: overvalued, high PE.",                  # round 2
        "Bull rebuttal: growth justifies PE.",                 # round 3
        "Bear rebuttal: growth slowing.",                      # round 4
        judge_json,                                            # round 5 (judge)
    ]
    client = make_llm_client(responses)
    return DebateEngine(client), client


# ---------------------------------------------------------------------------
# DebateRound dataclass
# ---------------------------------------------------------------------------

class TestDebateRound:
    def test_fields(self):
        r = DebateRound(role="bull", round_name="opening", content="Hello")
        assert r.role == "bull"
        assert r.round_name == "opening"
        assert r.content == "Hello"
        assert r.timestamp > 0


# ---------------------------------------------------------------------------
# DebateResult dataclass
# ---------------------------------------------------------------------------

class TestDebateResult:
    def test_defaults(self):
        dr = DebateResult(ticker="AAPL")
        assert dr.final_signal == "hold"
        assert dr.final_conviction == 0.5
        assert dr.rounds == []

    def test_get_round_found(self):
        dr = DebateResult(ticker="AAPL")
        r = DebateRound("bull", "opening", "bull text")
        dr.rounds.append(r)
        assert dr.get_round("bull", "opening") is r

    def test_get_round_not_found(self):
        dr = DebateResult(ticker="AAPL")
        assert dr.get_round("bull", "opening") is None

    def test_to_dict_keys(self):
        engine, _ = make_engine_with_canned_responses()
        result = engine.debate("AAPL")
        d = result.to_dict()
        assert "ticker" in d
        assert "final_signal" in d
        assert "final_conviction" in d
        assert "rounds" in d
        assert "debate_summary" in d
        assert "deciding_factors" in d

    def test_to_dict_rounds_list(self):
        engine, _ = make_engine_with_canned_responses()
        result = engine.debate("AAPL")
        d = result.to_dict()
        for round_dict in d["rounds"]:
            assert "role" in round_dict
            assert "round_name" in round_dict
            assert "content" in round_dict


# ---------------------------------------------------------------------------
# DebateEngine.debate — happy path
# ---------------------------------------------------------------------------

class TestDebateHappyPath:
    def test_returns_debate_result(self):
        engine, _ = make_engine_with_canned_responses()
        result = engine.debate("AAPL")
        assert isinstance(result, DebateResult)

    def test_ticker_set(self):
        engine, _ = make_engine_with_canned_responses()
        result = engine.debate("TSLA")
        assert result.ticker == "TSLA"

    def test_five_rounds_generated(self):
        engine, _ = make_engine_with_canned_responses()
        result = engine.debate("AAPL")
        assert len(result.rounds) == 5

    def test_round_roles_in_order(self):
        engine, _ = make_engine_with_canned_responses()
        result = engine.debate("AAPL")
        expected = [
            ("bull", "opening"),
            ("bear", "counter"),
            ("bull", "rebuttal"),
            ("bear", "rebuttal"),
            ("judge", "verdict"),
        ]
        for r, (role, name) in zip(result.rounds, expected):
            assert r.role == role
            assert r.round_name == name

    def test_llm_called_five_times(self):
        engine, client = make_engine_with_canned_responses()
        engine.debate("AAPL")
        assert client.chat_simple.call_count == 5

    def test_final_signal_from_judge(self):
        engine, _ = make_engine_with_canned_responses(signal="strong_buy")
        result = engine.debate("AAPL")
        assert result.final_signal == "strong_buy"

    def test_final_conviction_from_judge(self):
        engine, _ = make_engine_with_canned_responses(conviction=0.85)
        result = engine.debate("AAPL")
        assert result.final_conviction == pytest.approx(0.85)

    def test_bull_bear_scores(self):
        engine, _ = make_engine_with_canned_responses(bull_score=0.8, bear_score=0.4)
        result = engine.debate("AAPL")
        assert result.bull_score == pytest.approx(0.8)
        assert result.bear_score == pytest.approx(0.4)

    def test_deciding_factors_populated(self):
        engine, _ = make_engine_with_canned_responses(
            deciding_factors=["RSI oversold", "PE below sector"]
        )
        result = engine.debate("AAPL")
        assert "RSI oversold" in result.deciding_factors

    def test_debate_summary_set(self):
        engine, _ = make_engine_with_canned_responses(
            debate_summary="Bull won on fundamentals."
        )
        result = engine.debate("AAPL")
        assert "Bull" in result.debate_summary

    def test_judge_verdict_stored_in_round(self):
        engine, _ = make_engine_with_canned_responses()
        result = engine.debate("AAPL")
        verdict_round = result.get_round("judge", "verdict")
        assert verdict_round is not None
        assert len(verdict_round.content) > 0


# ---------------------------------------------------------------------------
# Signal data passed through to data summary
# ---------------------------------------------------------------------------

class TestDataSummary:
    def _make_technical(self, score=0.5, signal="buy"):
        t = MagicMock()
        t.score = score
        t.signal = signal
        t.ma_score = 0.3
        t.rsi_score = 0.4
        t.macd_score = 0.2
        t.indicators = {"rsi": 55, "macd": 0.5, "bb_pct_b": 0.6, "volume_ratio": 1.2}
        return t

    def _make_fundamental(self, score=0.3, signal="buy"):
        f = MagicMock()
        f.score = score
        f.signal = signal
        f.reasoning = "Solid balance sheet."
        f.valuation_score = 0.2
        f.growth_score = 0.4
        f.profitability_score = 0.5
        f.health_score = 0.3
        return f

    def _make_sentiment(self, score=0.2, signal="buy"):
        s = MagicMock()
        s.score = score
        s.signal = signal
        s.reasoning = "Positive news."
        s.news_volume = 5
        return s

    def test_data_summary_contains_ticker(self):
        summary = DebateEngine._build_data_summary("NVDA")
        assert "NVDA" in summary

    def test_data_summary_with_price(self):
        summary = DebateEngine._build_data_summary("AAPL", price=182.50)
        assert "182.50" in summary

    def test_data_summary_includes_technical(self):
        t = self._make_technical()
        summary = DebateEngine._build_data_summary("AAPL", technical_signal=t)
        assert "Technical" in summary
        assert "RSI" in summary

    def test_data_summary_includes_fundamental(self):
        f = self._make_fundamental()
        summary = DebateEngine._build_data_summary("AAPL", fundamental_signal=f)
        assert "Fundamental" in summary
        assert "Solid balance sheet" in summary

    def test_data_summary_includes_sentiment(self):
        s = self._make_sentiment()
        summary = DebateEngine._build_data_summary("AAPL", sentiment_signal=s)
        assert "Sentiment" in summary
        assert "Positive news" in summary

    def test_debate_with_all_signals(self):
        engine, client = make_engine_with_canned_responses()
        result = engine.debate(
            "AAPL",
            technical_signal=self._make_technical(),
            fundamental_signal=self._make_fundamental(),
            sentiment_signal=self._make_sentiment(),
            price=182.50,
        )
        assert result is not None
        # Verify signals were passed to the first call's user prompt
        first_call_prompt = client.chat_simple.call_args_list[0][1]["user"]
        assert "AAPL" in first_call_prompt
        assert "182.50" in first_call_prompt


# ---------------------------------------------------------------------------
# Prompt routing — correct system prompts used per round
# ---------------------------------------------------------------------------

class TestSystemPrompts:
    def test_bull_system_used_for_openings(self):
        engine, client = make_engine_with_canned_responses()
        engine.debate("AAPL")
        calls = client.chat_simple.call_args_list
        # Rounds 0 and 2 are bull
        assert calls[0][1]["system"] == _BULL_SYSTEM
        assert calls[2][1]["system"] == _BULL_SYSTEM

    def test_bear_system_used_for_counters(self):
        engine, client = make_engine_with_canned_responses()
        engine.debate("AAPL")
        calls = client.chat_simple.call_args_list
        # Rounds 1 and 3 are bear
        assert calls[1][1]["system"] == _BEAR_SYSTEM
        assert calls[3][1]["system"] == _BEAR_SYSTEM

    def test_judge_system_used_for_verdict(self):
        engine, client = make_engine_with_canned_responses()
        engine.debate("AAPL")
        calls = client.chat_simple.call_args_list
        assert calls[4][1]["system"] == _JUDGE_SYSTEM


# ---------------------------------------------------------------------------
# JSON parsing — judge verdict
# ---------------------------------------------------------------------------

class TestJudgeJsonParsing:
    def test_verdict_in_code_fence_parsed(self):
        """Judge verdict wrapped in ```json ... ``` should still parse."""
        json_data = make_judge_json(signal="sell", conviction=0.75)
        fenced = f"```json\n{json_data}\n```"
        responses = [
            "Bull opening",
            "Bear counter",
            "Bull rebuttal",
            "Bear rebuttal",
            fenced,
        ]
        client = make_llm_client(responses)
        engine = DebateEngine(client)
        result = engine.debate("AAPL")
        assert result.final_signal == "sell"
        assert result.final_conviction == pytest.approx(0.75)

    def test_malformed_json_falls_back_gracefully(self):
        """Malformed judge JSON should not raise — fallback values used."""
        responses = [
            "Bull opening",
            "Bear counter",
            "Bull rebuttal",
            "Bear rebuttal",
            "This is not JSON at all.",
        ]
        client = make_llm_client(responses)
        engine = DebateEngine(client)
        result = engine.debate("AAPL")
        assert result is not None
        assert result.final_signal == "hold"  # fallback default
        assert result.final_conviction == pytest.approx(0.5)  # fallback default
        # The non-JSON text should appear somewhere (summary or verdict)
        assert "not JSON" in result.debate_summary or len(result.debate_summary) > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_llm_exception_returns_none(self):
        """If any LLM call raises, debate() returns None without crashing."""
        client = MagicMock()
        client.chat_simple.side_effect = RuntimeError("API unavailable")
        engine = DebateEngine(client)
        result = engine.debate("AAPL")
        assert result is None

    def test_exception_on_round_3_returns_none(self):
        """Exception mid-debate returns None."""
        client = MagicMock()
        client.chat_simple.side_effect = [
            "Bull opening",
            "Bear counter",
            RuntimeError("timeout"),
        ]
        engine = DebateEngine(client)
        result = engine.debate("AAPL")
        assert result is None


# ---------------------------------------------------------------------------
# Integration with CompositeAnalyzer
# ---------------------------------------------------------------------------

class TestCompositeIntegration:
    def test_composite_accepts_debate_engine(self):
        from analysis.composite.composite import CompositeAnalyzer
        engine, _ = make_engine_with_canned_responses(
            signal="buy", conviction=0.7
        )
        composite = CompositeAnalyzer(debate_engine=engine)
        assert composite._debate_engine is engine

    def _make_tech(self, score=0.5, signal="buy"):
        """Minimal technical mock with numeric attributes for f-string formatting."""
        t = MagicMock()
        t.score = score
        t.signal = signal
        t.ma_score = 0.3
        t.rsi_score = 0.4
        t.macd_score = 0.2
        t.indicators = {"rsi": 55, "macd": 0.5, "bb_pct_b": 0.6,
                        "volume_ratio": 1.2, "current_price": 180.0}
        return t

    def test_debate_result_stored_on_composite_signal(self):
        engine, _ = make_engine_with_canned_responses(
            signal="buy", conviction=0.7
        )
        from analysis.composite.composite import CompositeAnalyzer
        composite = CompositeAnalyzer(debate_engine=engine)
        result = composite.analyze("AAPL", technical=self._make_tech())
        assert result.debate is not None
        assert result.debate.ticker == "AAPL"

    def test_high_conviction_debate_adjusts_signal(self):
        """When debate conviction >= 0.65, the composite signal is overridden."""
        engine, _ = make_engine_with_canned_responses(
            signal="strong_buy", conviction=0.9
        )
        from analysis.composite.composite import CompositeAnalyzer
        composite = CompositeAnalyzer(debate_engine=engine)
        result = composite.analyze("AAPL", technical=self._make_tech(score=0.45))
        assert result.signal == "strong_buy"

    def test_low_conviction_debate_does_not_override_signal(self):
        """When debate conviction < 0.65, composite signal is left as-is."""
        engine, _ = make_engine_with_canned_responses(
            signal="strong_sell", conviction=0.4
        )
        from analysis.composite.composite import CompositeAnalyzer
        composite = CompositeAnalyzer(debate_engine=engine)
        result = composite.analyze("AAPL", technical=self._make_tech(score=0.6))
        # Rule-based result should be "buy" or "strong_buy", not "strong_sell"
        assert result.signal in ("buy", "strong_buy")

    def test_no_debate_engine_leaves_debate_none(self):
        from analysis.composite.composite import CompositeAnalyzer
        composite = CompositeAnalyzer()
        tech = MagicMock()
        tech.score = 0.5
        tech.signal = "buy"
        tech.indicators = {"rsi": 55}
        result = composite.analyze("AAPL", technical=tech)
        assert result.debate is None

    def test_debate_failure_does_not_crash_composite(self):
        """If debate() returns None, composite still returns a valid signal."""
        from analysis.composite.composite import CompositeAnalyzer
        client = MagicMock()
        client.chat_simple.side_effect = RuntimeError("LLM down")
        broken_engine = DebateEngine(client)
        composite = CompositeAnalyzer(debate_engine=broken_engine)
        tech = MagicMock()
        tech.score = 0.5
        tech.signal = "buy"
        tech.indicators = {"rsi": 55}
        result = composite.analyze("AAPL", technical=tech)
        assert result is not None
        assert result.debate is None
