# StockAgent — 项目路线图与开发记录

---

## 当前阶段评估

### 已完成

| 模块 | 状态 | 测试 | 说明 |
|------|------|------|------|
| 项目结构 | ✅ | — | 40+ 文件，清晰的模块划分 |
| 配置系统 | ✅ | ✅ | YAML + 环境变量，4 种交易模式 + LLM 配置 |
| 经纪商抽象层 | ✅ | ✅ | SimulatedBroker + IBKRBroker + 工厂 |
| 模拟经纪商 | ✅ | ✅ | GBM 价格模型，持久化状态 |
| IBKR 经纪商 | ✅ | ❌ | ib_insync 集成，需实盘验证 |
| 长期记忆 | ✅ | ✅ | SQLite，分类/搜索/注入/归档 |
| 研报语料库 | ✅ | ✅ | FTS5 全文检索 |
| 交易日志 | ✅ | ✅ | 决策+交易+准确率追踪 |
| 市场数据 | ✅ | — | yfinance 集成 |
| 新闻/情绪 | ✅ | ✅ | Yahoo RSS + 规则 NLP |
| 基本面数据 | ✅ | — | 财务比率计算 |
| SEC 文件 | ✅ | — | EDGAR API 集成 |
| 技术分析 | ✅ | ✅ | 5 类指标，趋势/RSI/MACD/布林/成交量 |
| 基本面分析 | ✅ | ✅ | 4 维评分，行业基准对比 |
| 综合信号 | ✅ | ✅ | 加权合成 + 一致性检测 |
| 风控引擎 | ✅ | ✅ | 8+ 条规则，mock broker 测试 |
| 交易执行 | ✅ | ✅ | 下单 + 日志 + 通知 |
| 通知系统 | ✅ | ✅ | CLI + 飞书 + Telegram + 文件 |
| LLM 客户端 | ✅ | ✅ | 9 厂商支持 |
| **LLM 辩论引擎** | ✅ | ✅ 36 tests | Bull/Bear 5 轮辩论 + Judge 裁决 |
| **LLM 财报解读** | ✅ | ✅ | EPS/营收/指引三维分析 |
| **LLM 新闻分析** | ✅ | ✅ | 规则+LLM 60/40 混合评分 |
| **LLM 论点生成** | ✅ | ✅ | bull/bear case + 催化剂 |
| **LLM 风险评估** | ✅ | ✅ | 组合级风险打分 |
| **LLM 反思引擎** | ✅ | ✅ 23 tests | 交易后自动提炼教训 |
| CLI | ✅ | ✅ | 完整子命令集（含 backtest 子命令）|
| CI/CD | ✅ | ✅ | GitHub Actions 自动化 |
| 打包 | ✅ | ✅ | pyproject.toml |
| 文档 | ✅ | — | README + ARCHITECTURE + CONTRIBUTING |
| **回测引擎** | ✅ | ✅ 76 tests | HistoricalDataManager + BacktestBroker + BacktestEngine |
| **绩效指标** | ✅ | ✅ | Sharpe/Sortino/最大回撤/胜率/Calmar |
| **参数优化** | ✅ | ✅ | GridOptimizer，Cartesian 网格搜索 |
| **多 Agent 编排架构** | ✅ | ✅ 102 tests | 12-Agent TradingAgents 式四层架构 |
| **Agent 情景记忆** | ✅ | ✅ | 每角色 FTS5 记忆 + 反思引擎 |
| **Docker 支持** | ✅ | — | 多阶段构建 + docker-compose + CI 集成 |

**测试覆盖**: 493 测试函数，31 个测试文件，CI 自动化

### 待完成

| 缺失项 | 优先级 | 影响 |
|--------|--------|------|
| IBKR 实盘验证 | ⭐⭐ | 需要真实 TWS 环境测试 |
| 可视化/Web | 低 | 图表 + Web Dashboard |

---

## 开发路线图

### ✅ Phase 1：测试补全 + CI — 已完成

315+ 测试覆盖所有核心模块，CI 流水线已建立。

### ✅ Phase 2：LLM Agent 辩论 — 已完成

核心差异化特性：
- Bull/Bear 辩论引擎（5 轮结构化辩论，390 行）
- 6 个 LLM 分析模块（财报/新闻/论点/风险/反思）
- 反思记忆引擎（交易后自动学习）

### ✅ Phase 3：回测框架 — 已完成

**目标**：用历史数据验证策略。

- `HistoricalDataManager`：yfinance 下载 + CSV 缓存，无前视偏差
- `BacktestBroker`：实现 `BaseBroker`，按日重放历史价格，支持滑点 / 手续费
- `BacktestEngine`：技术分析驱动的每日循环，`TradeJournal(":memory:")` 内存 SQLite
- `PerformanceMetrics`：Sharpe / Sortino / 最大回撤 / 胜率 / Calmar 等 14 项指标
- `GridOptimizer`：Cartesian 网格搜索，按任意指标排序
- CLI：`python cli.py backtest --ticker AAPL --start 2023-01-01 --end 2024-01-01`
- 76 个新测试，全部无需真实 yfinance 调用

```
backtest/
├── engine.py              # 回测引擎
├── metrics.py             # 绩效指标
├── optimizer.py           # 参数优化
└── data_manager.py        # 历史数据管理
```

### ✅ Phase 4：多 Agent 编排架构 — 已完成

**目标**：从单一辩论引擎升级为 TradingAgents 式多 Agent 组织架构。

实现了 12 个独立 Agent 的四层协作系统：

```
Layer 1 (Analysts):   TechnicalAnalyst · FundamentalAnalyst · SentimentAnalyst · MacroAnalyst
Layer 2 (Research):   BullResearcher ↔ BearResearcher → ResearchManager (judge, deep model)
Layer 3 (Execution):  TraderAgent (concrete trade proposal + broker context)
Layer 4 (Risk):       AggressiveDebator ↔ ConservativeDebator ↔ NeutralDebator → PortfolioManager (judge)
```

新增特性：
- `AgentState` 共享状态贯穿全管道（无 Agent 间直接调用）
- 两路 LLM：fast_model（分析师/辩手）vs deep model（ResearchManager/PortfolioManager）
- `SituationMemory`：每角色独立的 FTS5 SQLite 情景记忆
- `AgentReflector`：交易后 LLM 提炼教训，写入对应角色记忆
- CLI：`python cli.py analyze AAPL --agents`（单 agent 模式不变）
- 102 个新测试，493 总测试数

```
agents/
├── state.py              # AgentState + DebateState
├── base.py               # BaseAgent ABC
├── graph.py              # TradingGraph 编排器
├── analysts/             # technical · fundamental · sentiment · macro
├── researchers/          # bull · bear
├── managers/             # research_manager · portfolio_manager
├── trader/               # trader
├── risk/                 # aggressive · conservative · neutral
└── memory/               # SituationMemory · AgentReflector
```

详见 `docs/MULTI_AGENT_ARCHITECTURE.md`

### ✅ Phase 5：Docker + 部署 — 已完成

- `Dockerfile`：两阶段构建（builder + runtime），非 root 用户，支持 `--build-arg EXTRAS="viz"`
- `docker-compose.yml`：`stockagent`（CLI 交互）+ `daemon`（长期运行调度）两个 service
- `.dockerignore`：排除测试、文档、缓存
- CI：新增 `docker-build` job，仅在 push to main 时触发，smoke test 验证镜像启动

```bash
docker build -t stockagent .
docker run --rm stockagent status
docker run --rm -v $(pwd)/data:/app/data stockagent analyze AAPL
docker compose up -d daemon
```

### Phase 6：Web Dashboard（3-4 周）可选

- FastAPI 后端 + WebSocket 实时推送
- React 前端：持仓概览、分析结果、交易日志

---

## 优先级矩阵

```
                高价值
                  │
    Phase 4       │      Phase 6
    多Agent架构    │      Web Dashboard
                  │
  ────────────────┼────────────────
                  │
    Phase 3       │      Phase 5
    回测框架       │      Docker+部署
                  │
                低价值

    ◄── 难度低          难度高 ──►
```

**当前进度**：
- Phase 1: ✅ 完成
- Phase 2: ✅ 完成（辩论引擎 + 6 个 LLM 模块）
- Phase 3: ✅ 完成（回测框架，76 tests，493 总测试数）
- Phase 4: ✅ 完成（12-Agent 多 Agent 架构，102 个新测试）
- Phase 5: ✅ 完成（Docker 多阶段构建 + docker-compose + CI）
- Phase 6: 未开始（可选）

**推荐顺序**：Phase 6（Web Dashboard，可选）

---

## 相关文档

- [求职简历项目介绍](RESUME.md)
- [多 Agent 架构设计](MULTI_AGENT_ARCHITECTURE.md)
- [项目 README](../README.md)