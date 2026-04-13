# StockAgent 仓库结构

## 双仓库架构

```
tortoiseknightma/stock-agent       ← Public (对外发布)
  └── main                         用户看到的：README + LICENSE + 代码

tortoiseknightma/stock-agent-dev   ← Private (内部开发)
  ├── main                         同步的发布版本
  └── dev                          构思笔记、设计过程、实验
```

## 本地 Git Remotes

| Remote | 指向 | 用途 |
|--------|------|------|
| `origin` | `stock-agent` (public) | `git push origin main` 发布 |
| `dev-origin` | `stock-agent-dev` (private) | `git push dev-origin dev` 备份开发 |

## 分支策略

| 分支 | 位置 | 内容 |
|------|------|------|
| `main` | 两个仓库都有 | 面向使用者的代码、README、LICENSE |
| `dev` | 仅 private 仓库 | 开发笔记、设计过程、实验性代码 |

## 日常工作流

```bash
# 开发 → 推到私有仓库（包含所有分支）
git push dev-origin main
git push dev-origin dev

# 发布 → 只推 main 到公开仓库
git push origin main
```

## 目录结构

```
stock-agent/
├── agent.py                       # 主编排器
├── cli.py                         # CLI 入口 (16 个子命令)
├── config.yaml                    # 默认配置
├── README.md                      # 公开：用户文档
├── LICENSE                        # 公开：MIT
├── CONTRIBUTING.md                # 公开：贡献指南
├── requirements.txt               # 依赖
├── core/
│   ├── config.py                  # 配置数据类 + YAML + 环境变量
│   ├── broker/
│   │   ├── base.py                # 抽象经纪商接口
│   │   ├── simulated.py           # 模拟经纪商 (GBM)
│   │   ├── ibkr_broker.py         # IBKR 实现
│   │   └── factory.py             # 工厂模式
│   └── memory/
│       ├── long_term.py           # 持久记忆 (SQLite)
│       ├── research_corpus.py     # 研报语料库 (FTS5)
│       └── trade_journal.py       # 交易决策日志
├── data/sources/
│   ├── market_data.py             # Yahoo Finance
│   ├── news.py                    # 新闻聚合 + 情绪
│   ├── fundamentals.py            # 财务比率
│   └── sec_filings.py             # SEC EDGAR
├── analysis/
│   ├── technical/technical.py     # 技术分析
│   ├── fundamental/fundamental.py # 基本面分析
│   ├── sentiment/sentiment.py     # 情绪分析
│   └── composite/composite.py     # 综合信号
├── execution/
│   ├── risk_engine.py             # 风控引擎
│   └── executor.py                # 交易执行器
├── push/
│   └── notifier.py                # 多渠道通知
├── docs/
│   └── ARCHITECTURE.md            # 公开：技术架构文档
├── tests/
└── .gitignore
```

## GitHub 仓库链接

- 公开发布: https://github.com/tortoiseknightma/stock-agent
- 私有开发: https://github.com/tortoiseknightma/stock-agent-dev
