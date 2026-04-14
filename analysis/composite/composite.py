"""
Composite Analysis Engine
==========================
Combines technical, fundamental, and sentiment signals into
a single actionable recommendation.

This is the main analysis interface that the rest of the system uses.
"""

import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field

from ..technical.technical import TechnicalSignal
from ..fundamental.fundamental import FundamentalSignal
from ..sentiment.sentiment import SentimentSignal

if TYPE_CHECKING:
    from analysis.llm.thesis_generator import ThesisGenerator, InvestmentThesis
    from analysis.llm.debate_engine import DebateEngine, DebateResult


@dataclass
class CompositeSignal:
    """Final composite analysis signal."""
    ticker: str
    composite_score: float        # -1 to 1
    signal: str                   # "strong_buy", "buy", "hold", "sell", "strong_sell"
    confidence: float             # 0 to 1
    
    # Component signals
    technical: Optional[TechnicalSignal] = None
    fundamental: Optional[FundamentalSignal] = None
    sentiment: Optional[SentimentSignal] = None
    
    # Agreement analysis
    signals_agree: bool = False
    disagreeing_signals: List[str] = field(default_factory=list)
    
    # Recommendation
    recommendation: str = ""
    risk_level: str = ""          # "low", "medium", "high"
    timestamp: float = field(default_factory=time.time)

    # Optional LLM-generated thesis (None if LLM not configured)
    thesis: Optional["InvestmentThesis"] = None

    # Optional Bull/Bear debate result (None if debate engine not configured)
    debate: Optional["DebateResult"] = None
    
    def to_dict(self) -> dict:
        d = {
            "ticker": self.ticker,
            "composite_score": self.composite_score,
            "signal": self.signal,
            "confidence": self.confidence,
            "signals_agree": self.signals_agree,
            "disagreeing_signals": self.disagreeing_signals,
            "recommendation": self.recommendation,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp,
        }
        if self.technical:
            d["technical"] = self.technical.to_dict()
        if self.fundamental:
            d["fundamental"] = self.fundamental.to_dict()
        if self.sentiment:
            d["sentiment"] = self.sentiment.to_dict()
        if self.thesis:
            d["thesis"] = self.thesis.to_dict()
        if self.debate:
            d["debate"] = self.debate.to_dict()
        return d


class CompositeAnalyzer:
    """
    Combines multi-dimensional signals into a unified recommendation.
    
    Usage:
        composite = CompositeAnalyzer(config)
        
        signal = composite.analyze(
            ticker="AAPL",
            technical=technical_signal,
            fundamental=fundamental_signal,
            sentiment=sentiment_signal,
        )
        
        print(signal.signal)           # "buy"
        print(signal.recommendation)   # "Moderate BUY — Strong technicals..."
    """
    
    def __init__(self, config=None,
                 thesis_generator: "Optional[ThesisGenerator]" = None,
                 debate_engine: "Optional[DebateEngine]" = None):
        if config:
            self.w_technical = config.weight_technical
            self.w_fundamental = config.weight_fundamental
            self.w_sentiment = config.weight_sentiment
            self.w_momentum = config.weight_momentum
            self.strong_buy = config.strong_buy_threshold
            self.buy = config.buy_threshold
            self.sell = config.sell_threshold
            self.strong_sell = config.strong_sell_threshold
        else:
            self.w_technical = 0.35
            self.w_fundamental = 0.35
            self.w_sentiment = 0.15
            self.w_momentum = 0.15
            self.strong_buy = 0.7
            self.buy = 0.4
            self.sell = -0.4
            self.strong_sell = -0.7

        self._thesis_gen = thesis_generator
        self._debate_engine = debate_engine
    
    def analyze(self, ticker: str,
                technical: TechnicalSignal = None,
                fundamental: FundamentalSignal = None,
                sentiment: SentimentSignal = None) -> CompositeSignal:
        """
        Combine signals into a composite recommendation.
        
        The composite weights can be adjusted per the config.
        Technical and fundamental analysis carry more weight than sentiment.
        """
        # Collect available scores with weights
        scores = []
        weights = []
        signal_names = []
        
        if technical:
            scores.append(technical.score)
            weights.append(self.w_technical)
            signal_names.append(("technical", technical.signal))
        
        if fundamental:
            scores.append(fundamental.score)
            weights.append(self.w_fundamental)
            signal_names.append(("fundamental", fundamental.signal))
        
        if sentiment:
            scores.append(sentiment.score)
            weights.append(self.w_sentiment)
            signal_names.append(("sentiment", sentiment.signal))
        
        if not scores:
            return CompositeSignal(
                ticker=ticker, composite_score=0, signal="hold",
                confidence=0, recommendation="No data available"
            )
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        composite = sum(s * w for s, w in zip(scores, normalized_weights))
        composite = max(-1, min(1, composite))
        
        # Determine signal
        signal = self._score_to_signal(composite)
        
        # Check signal agreement
        buy_sell_signals = [s for _, s in signal_names if s in ("strong_buy", "buy", "sell", "strong_sell")]
        if buy_sell_signals:
            all_bullish = all(s in ("strong_buy", "buy") for s in buy_sell_signals)
            all_bearish = all(s in ("strong_sell", "sell") for s in buy_sell_signals)
            signals_agree = all_bullish or all_bearish
        else:
            signals_agree = True
        
        disagreeing = []
        if not signals_agree:
            for name, sig in signal_names:
                if sig in ("strong_buy", "buy") and composite < 0:
                    disagreeing.append(name)
                elif sig in ("strong_sell", "sell") and composite > 0:
                    disagreeing.append(name)
        
        # Confidence: higher when signals agree and are strong
        base_confidence = abs(composite)
        agreement_bonus = 0.2 if signals_agree else -0.1
        confidence = max(0, min(1, base_confidence + agreement_bonus))
        
        # Risk assessment
        risk_level = self._assess_risk(technical, fundamental, sentiment)
        
        # Generate recommendation text
        recommendation = self._generate_recommendation(
            ticker, composite, signal, signals_agree, disagreeing,
            technical, fundamental, sentiment
        )
        
        result = CompositeSignal(
            ticker=ticker,
            composite_score=round(composite, 3),
            signal=signal,
            confidence=round(confidence, 3),
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            signals_agree=signals_agree,
            disagreeing_signals=disagreeing,
            recommendation=recommendation,
            risk_level=risk_level,
        )

        price = None
        if technical and technical.indicators:
            price = technical.indicators.get("current_price")

        # Optional: generate LLM investment thesis
        if self._thesis_gen is not None:
            try:
                result.thesis = self._thesis_gen.generate(
                    ticker=ticker,
                    technical_signal=technical,
                    fundamental_signal=fundamental,
                    sentiment_signal=sentiment,
                    price=price,
                )
            except Exception as e:
                print(f"[CompositeAnalyzer] Thesis generation failed for {ticker}: {e}")

        # Optional: run Bull/Bear debate (overrides thesis signal when conviction is higher)
        if self._debate_engine is not None:
            try:
                debate = self._debate_engine.debate(
                    ticker=ticker,
                    technical_signal=technical,
                    fundamental_signal=fundamental,
                    sentiment_signal=sentiment,
                    price=price,
                )
                if debate is not None:
                    result.debate = debate
                    # When the debate conviction is high, let it adjust the composite signal
                    if debate.final_conviction >= 0.65:
                        result.signal = debate.final_signal
                        result.confidence = min(1.0, result.confidence * 0.5
                                                + debate.final_conviction * 0.5)
            except Exception as e:
                print(f"[CompositeAnalyzer] Debate failed for {ticker}: {e}")

        return result
    
    def _assess_risk(self, technical, fundamental, sentiment) -> str:
        """Assess overall risk level of the position."""
        risk_factors = 0
        
        if technical:
            # High RSI = higher risk
            rsi = technical.indicators.get("rsi", 50)
            if rsi > 70 or rsi < 30:
                risk_factors += 1
            
            # Low volume = higher risk
            vol_ratio = technical.indicators.get("volume_ratio", 1)
            if vol_ratio < 0.5:
                risk_factors += 1
        
        if fundamental:
            # High debt = higher risk
            if fundamental.health_score < -0.3:
                risk_factors += 1
            
            # Negative earnings = higher risk
            if fundamental.growth_score < -0.3:
                risk_factors += 1
        
        if sentiment:
            # Extreme sentiment = higher risk
            if abs(sentiment.score) > 0.7:
                risk_factors += 1
        
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 1:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendation(self, ticker, composite, signal, agree,
                                   disagreeing, technical, fundamental, sentiment) -> str:
        """Generate human-readable recommendation."""
        strength = "Strong" if abs(composite) > 0.6 else "Moderate" if abs(composite) > 0.3 else "Weak"
        direction = "BUY" if composite > 0 else "SELL" if composite < 0 else "HOLD"
        
        parts = [f"{strength} {direction}"]
        
        # Add supporting reasoning
        if technical and abs(technical.score) > 0.3:
            parts.append(f"Technicals: {technical.signal}")
        
        if fundamental:
            parts.append(fundamental.reasoning)
        
        if sentiment:
            parts.append(sentiment.reasoning)
        
        # Add disagreement warning
        if not agree and disagreeing:
            parts.append(f"WARNING: {', '.join(disagreeing)} disagree")
        
        return " — ".join(parts)
    
    @staticmethod
    def _score_to_signal(score: float) -> str:
        if score >= 0.7:
            return "strong_buy"
        elif score >= 0.4:
            return "buy"
        elif score <= -0.7:
            return "strong_sell"
        elif score <= -0.4:
            return "sell"
        else:
            return "hold"
