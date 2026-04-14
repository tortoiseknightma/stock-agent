"""Tests for CompositeAnalyzer — signal synthesis, agreement detection, risk."""

import pytest
from analysis.composite.composite import CompositeAnalyzer, CompositeSignal
from analysis.technical.technical import TechnicalSignal
from analysis.fundamental.fundamental import FundamentalSignal
from analysis.sentiment.sentiment import SentimentSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tech(score: float, signal: str = None) -> TechnicalSignal:
    if signal is None:
        signal = _score_sig(score)
    return TechnicalSignal(
        ticker="TEST", score=score, signal=signal,
        ma_score=score, rsi_score=score * 0.5,
        macd_score=score * 0.5, volume_score=0.0,
        indicators={"rsi": 50 + score * 30, "volume_ratio": 1.0, "bb_pct_b": 0.5},
    )


def make_fund(score: float, signal: str = None) -> FundamentalSignal:
    if signal is None:
        signal = _score_sig(score)
    return FundamentalSignal(
        ticker="TEST", score=score, signal=signal,
        valuation_score=score, growth_score=score,
        profitability_score=score, health_score=score,
        reasoning=f"Fund score {score:+.2f}",
    )


def make_sent(score: float, signal: str = None) -> SentimentSignal:
    if signal is None:
        signal = _score_sig(score)
    return SentimentSignal(
        ticker="TEST", score=score, signal=signal,
        news_sentiment=score, news_volume=5,
    )


def _score_sig(score: float) -> str:
    if score >= 0.7:
        return "strong_buy"
    elif score >= 0.4:
        return "buy"
    elif score <= -0.7:
        return "strong_sell"
    elif score <= -0.4:
        return "sell"
    return "hold"


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_weights(self):
        ca = CompositeAnalyzer()
        assert ca.w_technical == pytest.approx(0.35)
        assert ca.w_fundamental == pytest.approx(0.35)
        assert ca.w_sentiment == pytest.approx(0.15)

    def test_config_weights(self):
        class Cfg:
            weight_technical = 0.4
            weight_fundamental = 0.4
            weight_sentiment = 0.1
            weight_momentum = 0.1
            strong_buy_threshold = 0.65
            buy_threshold = 0.35
            sell_threshold = -0.35
            strong_sell_threshold = -0.65
        ca = CompositeAnalyzer(Cfg())
        assert ca.w_technical == pytest.approx(0.4)
        assert ca.strong_buy == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# No data
# ---------------------------------------------------------------------------

class TestNoData:
    def test_returns_hold_when_no_signals(self):
        ca = CompositeAnalyzer()
        result = ca.analyze("AAPL")
        assert result.signal == "hold"
        assert result.composite_score == 0
        assert result.confidence == 0


# ---------------------------------------------------------------------------
# Score range & direction
# ---------------------------------------------------------------------------

class TestScoreProperties:
    def test_score_in_bounds_bullish(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(0.8), make_fund(0.7), make_sent(0.6))
        assert -1.0 <= r.composite_score <= 1.0

    def test_score_in_bounds_bearish(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(-0.8), make_fund(-0.7), make_sent(-0.6))
        assert -1.0 <= r.composite_score <= 1.0

    def test_bullish_inputs_positive_score(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(0.8), make_fund(0.7), make_sent(0.6))
        assert r.composite_score > 0

    def test_bearish_inputs_negative_score(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(-0.8), make_fund(-0.7), make_sent(-0.6))
        assert r.composite_score < 0

    def test_bullish_outscores_bearish(self):
        ca = CompositeAnalyzer()
        bull = ca.analyze("A", make_tech(0.8), make_fund(0.7))
        bear = ca.analyze("A", make_tech(-0.8), make_fund(-0.7))
        assert bull.composite_score > bear.composite_score

    def test_single_signal_still_works(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", technical=make_tech(0.6))
        assert r.composite_score > 0
        assert r.signal != "hold"


# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------

class TestSignalThresholds:
    @pytest.mark.parametrize("score,expected", [
        (0.8, "strong_buy"),
        (0.5, "buy"),
        (0.0, "hold"),
        (-0.5, "sell"),
        (-0.8, "strong_sell"),
    ])
    def test_score_to_signal(self, score, expected):
        assert CompositeAnalyzer._score_to_signal(score) == expected


# ---------------------------------------------------------------------------
# Agreement detection
# ---------------------------------------------------------------------------

class TestAgreement:
    def test_all_bullish_signals_agree(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(0.8), make_fund(0.7), make_sent(0.6))
        assert r.signals_agree is True
        assert r.disagreeing_signals == []

    def test_all_bearish_signals_agree(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(-0.8), make_fund(-0.7), make_sent(-0.6))
        assert r.signals_agree is True

    def test_mixed_signals_disagree(self):
        ca = CompositeAnalyzer()
        # Strong tech buy, strong fund sell → disagreement
        r = ca.analyze("A", make_tech(0.9), make_fund(-0.8))
        assert r.signals_agree is False
        assert len(r.disagreeing_signals) > 0

    def test_agreement_raises_confidence(self):
        ca = CompositeAnalyzer()
        agree = ca.analyze("A", make_tech(0.7), make_fund(0.7))
        disagree = ca.analyze("A", make_tech(0.7), make_fund(-0.7))
        assert agree.confidence > disagree.confidence


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_in_bounds(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(0.5), make_fund(0.5), make_sent(0.5))
        assert 0.0 <= r.confidence <= 1.0

    def test_stronger_signal_higher_confidence(self):
        ca = CompositeAnalyzer()
        weak = ca.analyze("A", make_tech(0.2), make_fund(0.2))
        strong = ca.analyze("A", make_tech(0.8), make_fund(0.8))
        assert strong.confidence > weak.confidence


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    def test_risk_level_low_for_clean_signals(self):
        ca = CompositeAnalyzer()
        t = make_tech(0.4)
        t.indicators["rsi"] = 55
        t.indicators["volume_ratio"] = 1.2
        r = ca.analyze("A", t, make_fund(0.3))
        assert r.risk_level in ("low", "medium")

    def test_risk_level_elevated_overbought(self):
        ca = CompositeAnalyzer()
        t = make_tech(0.5)
        t.indicators["rsi"] = 82   # overbought
        r = ca.analyze("A", t)
        assert r.risk_level in ("medium", "high")

    def test_risk_level_high_when_many_factors(self):
        ca = CompositeAnalyzer()
        t = make_tech(0.5)
        t.indicators["rsi"] = 85
        t.indicators["volume_ratio"] = 0.3
        f = make_fund(-0.5)
        f.health_score = -0.5
        f.growth_score = -0.4
        s = make_sent(0.9)
        r = ca.analyze("A", t, f, s)
        assert r.risk_level == "high"


# ---------------------------------------------------------------------------
# Recommendation text
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_recommendation_contains_direction(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(0.8), make_fund(0.7))
        assert any(w in r.recommendation for w in ("BUY", "SELL", "HOLD"))

    def test_recommendation_disagree_contains_warning(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(0.9, "strong_buy"), make_fund(-0.8, "strong_sell"))
        if not r.signals_agree:
            assert "WARNING" in r.recommendation or r.disagreeing_signals


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_to_dict_has_required_keys(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("AAPL", make_tech(0.5), make_fund(0.4))
        d = r.to_dict()
        for key in ("ticker", "composite_score", "signal", "confidence",
                    "signals_agree", "recommendation", "risk_level"):
            assert key in d

    def test_thesis_is_none_without_generator(self):
        ca = CompositeAnalyzer()
        r = ca.analyze("A", make_tech(0.5))
        assert r.thesis is None

    def test_component_signals_stored(self):
        ca = CompositeAnalyzer()
        t = make_tech(0.6)
        f = make_fund(0.5)
        r = ca.analyze("A", t, f)
        assert r.technical is t
        assert r.fundamental is f
