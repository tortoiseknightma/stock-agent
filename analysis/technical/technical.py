"""
Technical Analysis Module
==========================
Calculates standard technical indicators and generates trading signals.

Indicators implemented:
- Moving Averages (SMA, EMA crossover)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume analysis
- Support/Resistance levels

Output: TechnicalSignal with score -1 to 1 and detailed indicator values.
"""

import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TechnicalSignal:
    """Technical analysis result."""
    ticker: str
    score: float              # -1 (bearish) to 1 (bullish)
    signal: str               # "strong_buy", "buy", "hold", "sell", "strong_sell"
    
    # Individual indicator scores
    ma_score: float = 0.0     # Moving average crossover
    rsi_score: float = 0.0    # RSI overbought/oversold
    macd_score: float = 0.0   # MACD signal
    bb_score: float = 0.0     # Bollinger band position
    volume_score: float = 0.0 # Volume confirmation
    trend_score: float = 0.0  # Overall trend direction
    
    # Raw indicator values
    indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return self.__dict__


class TechnicalAnalyzer:
    """
    Technical analysis engine.
    
    Usage:
        analyzer = TechnicalAnalyzer(config)
        signal = analyzer.analyze(price_history)
    """
    
    def __init__(self, config=None):
        """Initialize with analysis configuration."""
        if config:
            self.ma_short = config.ma_short_period
            self.ma_long = config.ma_long_period
            self.rsi_period = config.rsi_period
            self.rsi_overbought = config.rsi_overbought
            self.rsi_oversold = config.rsi_oversold
            self.macd_fast = config.macd_fast
            self.macd_slow = config.macd_slow
            self.macd_signal = config.macd_signal
            self.bb_period = config.bollinger_period
            self.bb_std = config.bollinger_std
        else:
            self.ma_short = 20
            self.ma_long = 50
            self.rsi_period = 14
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            self.bb_period = 20
            self.bb_std = 2.0
    
    def analyze(self, price_history: List[Dict[str, Any]]) -> TechnicalSignal:
        """
        Run full technical analysis on price history.
        
        Args:
            price_history: List of OHLCV dicts with keys: date, open, high, low, close, volume
        
        Returns:
            TechnicalSignal with composite score and individual indicators
        """
        if len(price_history) < self.ma_long:
            return TechnicalSignal(ticker="", score=0, signal="hold")
        
        closes = [bar["close"] for bar in price_history]
        highs = [bar["high"] for bar in price_history]
        lows = [bar["low"] for bar in price_history]
        volumes = [bar["volume"] for bar in price_history]
        
        # Calculate all indicators
        ma_score, ma_indicators = self._analyze_moving_averages(closes)
        rsi_score, rsi_value = self._analyze_rsi(closes)
        macd_score, macd_indicators = self._analyze_macd(closes)
        bb_score, bb_indicators = self._analyze_bollinger(closes)
        volume_score, vol_indicators = self._analyze_volume(closes, volumes)
        trend_score = self._analyze_trend(closes, highs, lows)
        
        # Composite score (equal weighting by default)
        weights = {"ma": 0.25, "rsi": 0.15, "macd": 0.20, "bb": 0.15, "vol": 0.10, "trend": 0.15}
        composite = (
            weights["ma"] * ma_score +
            weights["rsi"] * rsi_score +
            weights["macd"] * macd_score +
            weights["bb"] * bb_score +
            weights["vol"] * volume_score +
            weights["trend"] * trend_score
        )
        
        # Collect all indicators
        all_indicators = {}
        all_indicators.update(ma_indicators)
        all_indicators["rsi"] = rsi_value
        all_indicators.update(macd_indicators)
        all_indicators.update(bb_indicators)
        all_indicators.update(vol_indicators)
        all_indicators["trend_score"] = trend_score
        
        return TechnicalSignal(
            ticker="",
            score=round(composite, 3),
            signal=self._score_to_signal(composite),
            ma_score=round(ma_score, 3),
            rsi_score=round(rsi_score, 3),
            macd_score=round(macd_score, 3),
            bb_score=round(bb_score, 3),
            volume_score=round(volume_score, 3),
            trend_score=round(trend_score, 3),
            indicators=all_indicators,
        )
    
    # --- Individual Indicator Calculations ---
    
    def _sma(self, data: List[float], period: int) -> List[float]:
        """Simple Moving Average."""
        if len(data) < period:
            return []
        return [sum(data[i-period+1:i+1]) / period for i in range(period-1, len(data))]
    
    def _ema(self, data: List[float], period: int) -> List[float]:
        """Exponential Moving Average."""
        if len(data) < period:
            return []
        
        k = 2 / (period + 1)
        ema = [sum(data[:period]) / period]
        
        for i in range(period, len(data)):
            ema.append(data[i] * k + ema[-1] * (1 - k))
        
        return ema
    
    def _analyze_moving_averages(self, closes: List[float]) -> tuple:
        """Analyze MA crossovers and price position relative to MAs."""
        sma_short = self._sma(closes, self.ma_short)
        sma_long = self._sma(closes, self.ma_long)
        ema_short = self._ema(closes, self.ma_short)
        ema_long = self._ema(closes, self.ma_long)
        
        if not sma_short or not sma_long:
            return 0, {}
        
        current_price = closes[-1]
        current_sma_short = sma_short[-1]
        current_sma_long = sma_long[-1]
        
        indicators = {
            "sma_short": round(current_sma_short, 2),
            "sma_long": round(current_sma_long, 2),
            "price_vs_sma_short": round((current_price / current_sma_short - 1) * 100, 2),
            "price_vs_sma_long": round((current_price / current_sma_long - 1) * 100, 2),
        }
        
        score = 0
        
        # Price above/below MAs
        if current_price > current_sma_short:
            score += 0.3
        else:
            score -= 0.3
        
        if current_price > current_sma_long:
            score += 0.3
        else:
            score -= 0.3
        
        # Golden/Death cross (short MA vs long MA)
        if current_sma_short > current_sma_long:
            score += 0.4  # Golden cross
        else:
            score -= 0.4  # Death cross
        
        return max(-1, min(1, score)), indicators
    
    def _analyze_rsi(self, closes: List[float]) -> tuple:
        """Calculate RSI and generate signal."""
        if len(closes) < self.rsi_period + 1:
            return 0, 50
        
        # Calculate price changes
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # Separate gains and losses
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # Calculate average gain/loss over RSI period
        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Generate score
        if rsi >= self.rsi_overbought:
            score = -0.7  # Overbought → sell signal
        elif rsi >= 60:
            score = -0.2  # Approaching overbought
        elif rsi <= self.rsi_oversold:
            score = 0.7   # Oversold → buy signal
        elif rsi <= 40:
            score = 0.2   # Approaching oversold
        else:
            score = 0     # Neutral
        
        return score, round(rsi, 2)
    
    def _analyze_macd(self, closes: List[float]) -> tuple:
        """Calculate MACD and generate signal."""
        if len(closes) < self.macd_slow + self.macd_signal:
            return 0, {}
        
        ema_fast = self._ema(closes, self.macd_fast)
        ema_slow = self._ema(closes, self.macd_slow)
        
        if not ema_fast or not ema_slow:
            return 0, {}
        
        # Align lengths
        offset = self.macd_slow - self.macd_fast
        macd_line = [ema_fast[i + offset] - ema_slow[i] for i in range(len(ema_slow))]
        
        if len(macd_line) < self.macd_signal:
            return 0, {}
        
        signal_line = self._ema(macd_line, self.macd_signal)
        
        if not signal_line:
            return 0, {}
        
        # Align and calculate histogram
        hist_offset = len(macd_line) - len(signal_line)
        histogram = [macd_line[i + hist_offset] - signal_line[i] for i in range(len(signal_line))]
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0
        
        indicators = {
            "macd_line": round(current_macd, 4),
            "macd_signal": round(current_signal, 4),
            "macd_histogram": round(current_hist, 4),
        }
        
        score = 0
        
        # MACD above/below signal line
        if current_macd > current_signal:
            score += 0.4
        else:
            score -= 0.4
        
        # Histogram increasing/decreasing
        if current_hist > prev_hist:
            score += 0.3  # Momentum increasing
        else:
            score -= 0.3  # Momentum decreasing
        
        # Zero line crossover
        if current_macd > 0:
            score += 0.3
        else:
            score -= 0.3
        
        return max(-1, min(1, score)), indicators
    
    def _analyze_bollinger(self, closes: List[float]) -> tuple:
        """Analyze Bollinger Band position."""
        if len(closes) < self.bb_period:
            return 0, {}
        
        sma = self._sma(closes, self.bb_period)
        if not sma:
            return 0, {}
        
        # Calculate standard deviation
        recent_closes = closes[-self.bb_period:]
        mean = sum(recent_closes) / len(recent_closes)
        variance = sum((x - mean) ** 2 for x in recent_closes) / len(recent_closes)
        std = math.sqrt(variance)
        
        upper = sma[-1] + self.bb_std * std
        lower = sma[-1] - self.bb_std * std
        middle = sma[-1]
        current = closes[-1]
        
        # Calculate %B (position within bands)
        band_width = upper - lower
        if band_width == 0:
            pct_b = 0.5
        else:
            pct_b = (current - lower) / band_width
        
        indicators = {
            "bb_upper": round(upper, 2),
            "bb_middle": round(middle, 2),
            "bb_lower": round(lower, 2),
            "bb_pct_b": round(pct_b, 3),
            "bb_width": round(band_width / middle * 100, 2),  # Width as % of price
        }
        
        # Score based on position within bands
        if pct_b < 0:
            score = 0.6    # Below lower band → oversold
        elif pct_b < 0.2:
            score = 0.3    # Near lower band
        elif pct_b > 1:
            score = -0.6   # Above upper band → overbought
        elif pct_b > 0.8:
            score = -0.3   # Near upper band
        else:
            score = 0      # Within normal range
        
        return score, indicators
    
    def _analyze_volume(self, closes: List[float], volumes: List[int]) -> tuple:
        """Analyze volume trends for confirmation."""
        if len(volumes) < 20:
            return 0, {}
        
        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Price change direction
        price_change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] else 0
        
        indicators = {
            "current_volume": current_volume,
            "avg_volume_20d": int(avg_volume),
            "volume_ratio": round(volume_ratio, 2),
        }
        
        score = 0
        
        # High volume confirms direction
        if volume_ratio > 1.5:
            if price_change > 0:
                score = 0.5   # High volume + price up = bullish
            else:
                score = -0.5  # High volume + price down = bearish
        elif volume_ratio < 0.5:
            score = 0  # Low volume = inconclusive
        
        return score, indicators
    
    def _analyze_trend(self, closes: List[float], highs: List[float],
                        lows: List[float]) -> float:
        """Analyze overall trend direction using linear regression slope."""
        if len(closes) < 20:
            return 0
        
        # Use last 20 periods
        n = 20
        y = closes[-n:]
        x = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        slope = numerator / denominator
        
        # Normalize slope to -1 to 1 range
        slope_pct = slope / y_mean * 100  # Slope as % of price
        
        return max(-1, min(1, slope_pct * 2))  # Scale factor
    
    @staticmethod
    def _score_to_signal(score: float) -> str:
        """Convert numeric score to signal string."""
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
