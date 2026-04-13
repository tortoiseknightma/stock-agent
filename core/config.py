"""
StockAgent Configuration Module
==============================
Centralized configuration with environment variable overrides.
Supports multiple profiles: simulated, paper, live.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum


class TradingMode(Enum):
    SIMULATED = "simulated"   # Local mock with random walks
    PAPER = "paper"           # IBKR paper trading account
    ADVISORY = "advisory"     # Analysis only, no execution
    LIVE = "live"             # Real trading (requires extra confirmation)


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497          # 7497=paper, 7496=live
    client_id: int = 1
    timeout: int = 30
    readonly: bool = True     # Safety: read-only by default


@dataclass
class SimulatedConfig:
    initial_capital: float = 100_000.0
    default_portfolio: Dict[str, int] = field(default_factory=lambda: {
        "AAPL": 100, "MSFT": 50, "GOOGL": 30, "NVDA": 40,
        "AMZN": 25, "JPM": 40, "V": 35, "TSLA": 20
    })
    slippage_pct: float = 0.001    # 0.1% simulated slippage
    commission_per_trade: float = 1.0


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20    # Max 20% in single position
    max_trade_pct: float = 0.05       # Max 5% per trade
    daily_loss_limit_pct: float = 0.03  # 3% daily loss limit
    stop_loss_pct: float = 0.08       # 8% stop-loss
    take_profit_pct: float = 0.20     # 20% take-profit
    max_trades_per_day: int = 10
    large_trade_threshold: float = 5000.0  # Trades above need confirmation
    var_confidence: float = 0.95      # VaR confidence level
    max_drawdown_pct: float = 0.15    # 15% max portfolio drawdown


@dataclass
class AnalysisConfig:
    # Technical analysis
    ma_short_period: int = 20
    ma_long_period: int = 50
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Signal weights
    weight_technical: float = 0.35
    weight_fundamental: float = 0.35
    weight_sentiment: float = 0.15
    weight_momentum: float = 0.15
    
    # Decision thresholds
    strong_buy_threshold: float = 0.7
    buy_threshold: float = 0.4
    sell_threshold: float = -0.4
    strong_sell_threshold: float = -0.7


@dataclass
class SchedulerConfig:
    premarket_time: str = "09:00"
    intraday_interval_min: int = 30
    postmarket_time: str = "16:30"
    weekend_enabled: bool = False
    timezone: str = "US/Eastern"


@dataclass
class MemoryConfig:
    max_session_transcripts: int = 100
    max_research_docs: int = 500
    max_trade_journal_entries: int = 1000
    memory_db_path: str = "data/memory.db"
    research_index_path: str = "data/research_corpus"


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai"         # openai, anthropic, google, deepseek, qwen, glm, xai, ollama, openrouter
    model: str = "gpt-4o"            # Model for analysis tasks
    reasoning_model: str = "gpt-4o"  # Model for complex reasoning (debate, reflection)
    base_url: Optional[str] = None   # Custom API endpoint
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class PushConfig:
    channels: list = field(default_factory=lambda: ["cli"])
    feishu_webhook: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_recipient: Optional[str] = None


@dataclass
class AppConfig:
    trading_mode: TradingMode = TradingMode.SIMULATED
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    simulated: SimulatedConfig = field(default_factory=SimulatedConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    push: PushConfig = field(default_factory=PushConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from YAML file with env var overrides."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}
        
        # Apply env var overrides
        env_overrides = cls._load_env_overrides()
        raw = cls._deep_merge(raw, env_overrides)
        
        return cls._from_dict(raw)
    
    @classmethod
    def _load_env_overrides(cls) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        mode = os.getenv("STOCKAGENT_TRADING_MODE")
        if mode:
            overrides["trading_mode"] = mode
        
        ibkr_host = os.getenv("STOCKAGENT_IBKR_HOST")
        ibkr_port = os.getenv("STOCKAGENT_IBKR_PORT")
        if ibkr_host or ibkr_port:
            overrides.setdefault("ibkr", {})
            if ibkr_host:
                overrides["ibkr"]["host"] = ibkr_host
            if ibkr_port:
                overrides["ibkr"]["port"] = int(ibkr_port)
        
        feishu = os.getenv("STOCKAGENT_FEISHU_WEBHOOK")
        tg_token = os.getenv("STOCKAGENT_TELEGRAM_TOKEN")
        if feishu or tg_token:
            overrides.setdefault("push", {})
            if feishu:
                overrides["push"]["feishu_webhook"] = feishu
            if tg_token:
                overrides["push"]["telegram_bot_token"] = tg_token
                overrides["push"]["telegram_chat_id"] = os.getenv("STOCKAGENT_TELEGRAM_CHAT_ID")

        # LLM overrides
        llm_provider = os.getenv("STOCKAGENT_LLM_PROVIDER")
        llm_model = os.getenv("STOCKAGENT_LLM_MODEL")
        if llm_provider or llm_model:
            overrides.setdefault("llm", {})
            if llm_provider:
                overrides["llm"]["provider"] = llm_provider
            if llm_model:
                overrides["llm"]["model"] = llm_model

        return overrides
    
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = AppConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "AppConfig":
        """Construct AppConfig from a dictionary."""
        mode_str = raw.get("trading_mode", "simulated")
        try:
            mode = TradingMode(mode_str)
        except ValueError:
            mode = TradingMode.SIMULATED
        
        def _dc_from_dict(dc_class, data):
            if not data:
                return dc_class()
            fields = {k: v for k, v in data.items() if k in dc_class.__dataclass_fields__}
            return dc_class(**fields)
        
        return cls(
            trading_mode=mode,
            ibkr=_dc_from_dict(IBKRConfig, raw.get("ibkr")),
            simulated=_dc_from_dict(SimulatedConfig, raw.get("simulated")),
            risk=_dc_from_dict(RiskConfig, raw.get("risk")),
            analysis=_dc_from_dict(AnalysisConfig, raw.get("analysis")),
            scheduler=_dc_from_dict(SchedulerConfig, raw.get("scheduler")),
            memory=_dc_from_dict(MemoryConfig, raw.get("memory")),
            llm=_dc_from_dict(LLMConfig, raw.get("llm")),
            push=_dc_from_dict(PushConfig, raw.get("push")),
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        data["trading_mode"] = self.trading_mode.value
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load application configuration."""
    return AppConfig.from_yaml(path)
