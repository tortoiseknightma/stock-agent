"""Tests for FundamentalAnalyzer — scoring dimensions and sector benchmarks."""

import pytest
from analysis.fundamental.fundamental import FundamentalAnalyzer, FundamentalSignal
from data.sources.fundamentals import FinancialRatios


def ratios(**kwargs):
    return FinancialRatios(ticker="TEST", **kwargs)


class TestScoreRange:
    def test_composite_score_within_bounds(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=15, revenue_growth=0.20, net_margin=0.25, roe=0.30,
                   debt_to_equity=50, current_ratio=2.5)
        signal = analyzer.analyze(r, "Technology")
        assert -1.0 <= signal.score <= 1.0

    def test_all_dimension_scores_within_bounds(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=15, peg_ratio=0.8, pb_ratio=2.0,
                   revenue_growth=0.20, earnings_growth=0.15,
                   net_margin=0.25, roe=0.30, operating_margin=0.20,
                   debt_to_equity=40, current_ratio=2.0)
        signal = analyzer.analyze(r)
        for dim in (signal.valuation_score, signal.growth_score,
                    signal.profitability_score, signal.health_score):
            assert -1.0 <= dim <= 1.0


class TestValuationScoring:
    def test_low_pe_gives_positive_valuation(self):
        """PE well below benchmark → attractive valuation."""
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=8.0)   # Default benchmark 18; 8/18 < 0.7 → 0.8
        signal = analyzer.analyze(r, "")
        assert signal.valuation_score > 0

    def test_high_pe_gives_negative_valuation(self):
        """PE 2x+ benchmark → expensive."""
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=50.0)  # Default benchmark 18; 50/18 > 1.5
        signal = analyzer.analyze(r, "")
        assert signal.valuation_score < 0

    def test_low_peg_gives_high_valuation_score(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(peg_ratio=0.4)
        signal = analyzer.analyze(r, "")
        assert signal.valuation_score > 0.5

    def test_high_pb_gives_negative_valuation(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(pb_ratio=10.0)
        signal = analyzer.analyze(r, "")
        assert signal.valuation_score < 0


class TestGrowthScoring:
    def test_high_revenue_growth_gives_positive_score(self):
        analyzer = FundamentalAnalyzer()
        # 2x+ benchmark (0.08 default) → 0.9
        r = ratios(revenue_growth=0.30, earnings_growth=0.25)
        signal = analyzer.analyze(r, "")
        assert signal.growth_score > 0.5

    def test_negative_revenue_growth_gives_negative_score(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(revenue_growth=-0.10)
        signal = analyzer.analyze(r, "")
        assert signal.growth_score < 0

    def test_no_growth_data_returns_zero(self):
        analyzer = FundamentalAnalyzer()
        r = ratios()  # All None
        signal = analyzer.analyze(r, "")
        assert signal.growth_score == 0.0


class TestProfitabilityScoring:
    def test_high_margin_gives_positive_score(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(net_margin=0.35, roe=0.40, operating_margin=0.30)
        signal = analyzer.analyze(r, "Technology")
        assert signal.profitability_score > 0

    def test_negative_margin_gives_negative_score(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(net_margin=-0.10)
        signal = analyzer.analyze(r, "")
        assert signal.profitability_score < 0


class TestHealthScoring:
    def test_low_debt_gives_positive_health(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(debt_to_equity=10, current_ratio=3.0)
        signal = analyzer.analyze(r, "")
        assert signal.health_score > 0

    def test_very_high_debt_gives_negative_health(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(debt_to_equity=200)
        signal = analyzer.analyze(r, "")
        assert signal.health_score < 0

    def test_low_current_ratio_gives_negative_health(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(current_ratio=0.5)
        signal = analyzer.analyze(r, "")
        assert signal.health_score < 0


class TestSectorBenchmarks:
    def test_tech_sector_uses_higher_pe_benchmark(self):
        """Same PE should score better in Tech (higher benchmark) than Energy."""
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=15.0)
        tech_signal = analyzer.analyze(r, "Technology")     # benchmark=25
        energy_signal = analyzer.analyze(r, "Energy")       # benchmark=10
        # In Tech, 15/25 = 0.6 → cheap; in Energy, 15/10 = 1.5 → expensive
        assert tech_signal.valuation_score > energy_signal.valuation_score

    def test_unknown_sector_uses_default_benchmarks(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=18.0)
        signal = analyzer.analyze(r, "UnknownSector")
        # Should not raise, should use default
        assert isinstance(signal, FundamentalSignal)


class TestSignalString:
    @pytest.mark.parametrize("score,expected", [
        (0.75, "strong_buy"),
        (0.5, "buy"),
        (0.0, "hold"),
        (-0.5, "sell"),
        (-0.75, "strong_sell"),
    ])
    def test_score_to_signal(self, score, expected):
        result = FundamentalAnalyzer._score_to_signal(score)
        assert result == expected


class TestReasoningText:
    def test_reasoning_populated(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=8, revenue_growth=0.30, net_margin=0.30, roe=0.35)
        signal = analyzer.analyze(r, "Technology")
        assert len(signal.reasoning) > 0
        assert "Fundamentals" in signal.reasoning

    def test_ratios_dict_attached(self):
        analyzer = FundamentalAnalyzer()
        r = ratios(pe_forward=20)
        signal = analyzer.analyze(r, "")
        assert isinstance(signal.ratios, dict)
        assert "pe_forward" in signal.ratios
