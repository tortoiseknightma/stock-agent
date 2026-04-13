"""Tests for configuration loading and validation."""

import os
import pytest
import tempfile
import yaml

from core.config import (
    AppConfig, TradingMode, RiskConfig, AnalysisConfig,
    IBKRConfig, SimulatedConfig, load_config,
)


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestAppConfigDefaults:
    def test_default_trading_mode_is_simulated(self):
        config = AppConfig()
        assert config.trading_mode == TradingMode.SIMULATED

    def test_default_risk_limits(self):
        risk = RiskConfig()
        assert risk.max_position_pct == 0.20
        assert risk.max_trade_pct == 0.05
        assert risk.daily_loss_limit_pct == 0.03
        assert risk.stop_loss_pct == 0.08
        assert risk.take_profit_pct == 0.20
        assert risk.max_trades_per_day == 10

    def test_default_analysis_weights_sum_to_one(self):
        cfg = AnalysisConfig()
        total = (cfg.weight_technical + cfg.weight_fundamental +
                 cfg.weight_sentiment + cfg.weight_momentum)
        assert abs(total - 1.0) < 1e-9

    def test_ibkr_readonly_by_default(self):
        assert IBKRConfig().readonly is True


class TestYamlLoading:
    def test_load_trading_mode_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        _write_yaml(cfg_file, {"trading_mode": "paper"})
        config = AppConfig.from_yaml(str(cfg_file))
        assert config.trading_mode == TradingMode.PAPER

    def test_load_risk_overrides(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        _write_yaml(cfg_file, {"risk": {"max_position_pct": 0.10, "max_trades_per_day": 5}})
        config = AppConfig.from_yaml(str(cfg_file))
        assert config.risk.max_position_pct == 0.10
        assert config.risk.max_trades_per_day == 5
        # Unspecified fields keep defaults
        assert config.risk.stop_loss_pct == 0.08

    def test_missing_yaml_uses_defaults(self, tmp_path):
        config = AppConfig.from_yaml(str(tmp_path / "nonexistent.yaml"))
        assert config.trading_mode == TradingMode.SIMULATED

    def test_invalid_trading_mode_falls_back_to_simulated(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        _write_yaml(cfg_file, {"trading_mode": "invalid_mode"})
        config = AppConfig.from_yaml(str(cfg_file))
        assert config.trading_mode == TradingMode.SIMULATED


class TestEnvVarOverrides:
    def test_trading_mode_env_override(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.yaml"
        _write_yaml(cfg_file, {"trading_mode": "simulated"})
        monkeypatch.setenv("STOCKAGENT_TRADING_MODE", "advisory")
        config = AppConfig.from_yaml(str(cfg_file))
        assert config.trading_mode == TradingMode.ADVISORY

    def test_ibkr_host_env_override(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.yaml"
        _write_yaml(cfg_file, {})
        monkeypatch.setenv("STOCKAGENT_IBKR_HOST", "192.168.1.1")
        monkeypatch.setenv("STOCKAGENT_IBKR_PORT", "7496")
        config = AppConfig.from_yaml(str(cfg_file))
        assert config.ibkr.host == "192.168.1.1"
        assert config.ibkr.port == 7496

    def test_feishu_webhook_env_override(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.yaml"
        _write_yaml(cfg_file, {})
        monkeypatch.setenv("STOCKAGENT_FEISHU_WEBHOOK", "https://example.com/hook")
        config = AppConfig.from_yaml(str(cfg_file))
        assert config.push.feishu_webhook == "https://example.com/hook"


class TestLoadConfigHelper:
    def test_load_config_returns_app_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        _write_yaml(cfg_file, {})
        config = load_config(str(cfg_file))
        assert isinstance(config, AppConfig)
