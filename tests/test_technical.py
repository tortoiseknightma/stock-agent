"""Tests for TechnicalAnalyzer — indicators, scoring, edge cases."""

import pytest
from analysis.technical.technical import TechnicalAnalyzer, TechnicalSignal


def make_bars(closes, volumes=None):
    """Build minimal OHLCV bar list from a close price series."""
    volumes = volumes or [1_000_000] * len(closes)
    return [
        {
            "date": i * 86400,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": v,
        }
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


def trending_up(n=60, start=100.0, step=1.0):
    """Monotonically rising prices — strong bullish."""
    return [start + i * step for i in range(n)]


def trending_down(n=60, start=160.0, step=1.0):
    """Monotonically falling prices — strong bearish."""
    return [start - i * step for i in range(n)]


def flat(n=60, price=100.0):
    return [price] * n


class TestAnalyzerInit:
    def test_default_config(self):
        analyzer = TechnicalAnalyzer()
        assert analyzer.rsi_period == 14
        assert analyzer.ma_short == 20
        assert analyzer.ma_long == 50

    def test_config_overrides(self):
        class Cfg:
            ma_short_period = 10
            ma_long_period = 30
            rsi_period = 7
            rsi_overbought = 80
            rsi_oversold = 20
            macd_fast = 8
            macd_slow = 17
            macd_signal = 9
            bollinger_period = 20
            bollinger_std = 2.0

        analyzer = TechnicalAnalyzer(Cfg())
        assert analyzer.ma_short == 10
        assert analyzer.rsi_period == 7


class TestInsufficientData:
    def test_returns_hold_when_too_few_bars(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars([100.0] * 10)  # Less than ma_long (50)
        signal = analyzer.analyze(bars)
        assert signal.signal == "hold"
        assert signal.score == 0


class TestScoreRange:
    def test_score_within_bounds_bullish(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars(trending_up(60))
        signal = analyzer.analyze(bars)
        assert -1.0 <= signal.score <= 1.0

    def test_score_within_bounds_bearish(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars(trending_down(60))
        signal = analyzer.analyze(bars)
        assert -1.0 <= signal.score <= 1.0


class TestSignalDirection:
    def test_uptrend_produces_positive_score(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars(trending_up(60))
        signal = analyzer.analyze(bars)
        assert signal.score > 0

    def test_downtrend_produces_negative_score(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars(trending_down(60))
        signal = analyzer.analyze(bars)
        assert signal.score < 0

    def test_uptrend_score_higher_than_downtrend(self):
        """An uptrend must score higher than a downtrend — signal direction is consistent."""
        analyzer = TechnicalAnalyzer()
        up_signal = analyzer.analyze(make_bars(trending_up(60)))
        down_signal = analyzer.analyze(make_bars(trending_down(60)))
        assert up_signal.score > down_signal.score


class TestIndicatorValues:
    def test_rsi_overbought_gives_negative_rsi_score(self):
        """Flat high price (no losses) → RSI → 100 → sell signal."""
        analyzer = TechnicalAnalyzer()
        # First 50 bars rise slowly to satisfy ma_long, last 10 shoot up
        closes = list(range(100, 150)) + [200.0] * 10
        bars = make_bars(closes)
        signal = analyzer.analyze(bars)
        assert signal.rsi_score < 0

    def test_indicators_dict_populated(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars(trending_up(60))
        signal = analyzer.analyze(bars)
        assert "rsi" in signal.indicators
        assert "sma_short" in signal.indicators
        assert "sma_long" in signal.indicators

    def test_rsi_value_in_valid_range(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars(trending_up(60))
        signal = analyzer.analyze(bars)
        rsi = signal.indicators.get("rsi", 50)
        assert 0 <= rsi <= 100

    def test_bollinger_pct_b_present(self):
        analyzer = TechnicalAnalyzer()
        bars = make_bars(trending_up(60))
        signal = analyzer.analyze(bars)
        assert "bb_pct_b" in signal.indicators


class TestScoreToSignal:
    @pytest.mark.parametrize("score,expected", [
        (0.8, "strong_buy"),
        (0.5, "buy"),
        (0.0, "hold"),
        (-0.5, "sell"),
        (-0.8, "strong_sell"),
    ])
    def test_score_to_signal_thresholds(self, score, expected):
        result = TechnicalAnalyzer._score_to_signal(score)
        assert result == expected


class TestMovingAverages:
    def test_sma_length(self):
        analyzer = TechnicalAnalyzer()
        data = list(range(1, 21))
        sma = analyzer._sma(data, 5)
        assert len(sma) == 16  # n - period + 1

    def test_sma_value(self):
        analyzer = TechnicalAnalyzer()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        sma = analyzer._sma(data, 3)
        assert sma[0] == pytest.approx(2.0)
        assert sma[-1] == pytest.approx(4.0)

    def test_ema_last_value_close_to_recent_prices(self):
        """EMA should be more responsive to recent prices than SMA."""
        analyzer = TechnicalAnalyzer()
        # Steady price then jump
        data = [100.0] * 20 + [200.0] * 5
        ema = analyzer._ema(data, 10)
        sma = analyzer._sma(data, 10)
        # EMA last value should be higher than SMA last value after the jump
        assert ema[-1] > sma[-1]

    def test_sma_insufficient_data_returns_empty(self):
        analyzer = TechnicalAnalyzer()
        assert analyzer._sma([1.0, 2.0], 5) == []

    def test_ema_insufficient_data_returns_empty(self):
        analyzer = TechnicalAnalyzer()
        assert analyzer._ema([1.0, 2.0], 5) == []
