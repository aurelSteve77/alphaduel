# AlphaDuel — Specification

> A modular research platform to benchmark **quantitative RL agents**, **off-the-shelf LLM agents**, and **fine-tuned LLM agents (SFT + RL)** on a shared, realistic trading environment.
>
> **Central research question:** *Does the textual representation and reasoning of an LLM produce better trading decisions than a purely quantitative neural network, when both are given the same information?*

---

## 1. Motivation & positioning

This is a portfolio project aimed at quant-research / ML roles (hedge funds, GAFAM ML, bulge-bracket banks). The *differentiator* is **methodological rigor**, not feature count. The project is designed to demonstrate that the author understands **why naive backtests lie** and how to run a fair, leakage-free, statistically honest comparison between decision-making paradigms.

Non-goals: a production trading system, a profitable strategy, or HFT/microstructure modeling.

---

## 2. Core design principles

1. **Shared environment, swappable agents.** The `AlphaDuelGym` env defines the rules of the game (observations, actions, costs, reward). Every agent — baseline, RL, LLM — implements the same `Agent` interface and is judged identically.
2. **Fair comparison via factorial design.** The research question is only answerable if agents receive the *same information*. We evaluate each **agent × feature-set** combination (quant-only / text-only / both) so we isolate *reasoning modality* from *information access*.
3. **Point-in-time (PIT) correctness first.** No datum reaches the agent before it was knowable. Enforced by design and by an automated leakage test-suite.
4. **Everything is a config.** Experiments are declared in YAML, validated by Pydantic, and reproducible from a single config hash + seed.
5. **Honest evaluation.** Mandatory naive baselines, walk-forward/purged CV, multiple regimes, risk-adjusted metrics, and confidence intervals. If nothing beats buy-and-hold net of costs, we report that.
6. **Modularity for scale.** Single-asset MVP → multi-asset. Free data → paid data. Local GPU → cloud. Each is a config/adapter swap, not a rewrite.

---

## 3. Scope & phasing

| Phase | Deliverable | Status |
|------|-------------|--------|
| **P0 (MVP)** | Data (prices+macro) → feature store → single-asset `AlphaDuelGym` (weight actions, realistic costs) → buy-and-hold/random baselines → evaluation harness w/ metrics | **Target of this scaffold** |
| **P1** | Quant RL agent (PPO/A2C), reward shaping, walk-forward eval, feature ablations | Planned |
| **P2** | Off-the-shelf LLM agent (with leakage controls), news/text features + sentiment embeddings for the quant side | Planned |
| **P3** | Trained LLM: SFT on synthetic traces → GRPO/DPO | Planned |
| **P4** | Live dashboard + multi-asset portfolio | Planned |

**MVP decisions (locked):** single asset · weight-based actions · quant-only features + baselines · single local GPU (16–24 GB) · free data only (yfinance / FRED / GDELT) · YAML + Pydantic config · MLflow tracking.

---

## 4. Methodology (the part reviewers judge)

### 4.1 Look-ahead & leakage controls
- **Execution lag:** decisions made on bar `t` observations execute at bar `t+1` open. The agent never trades on a price it has already observed.
- **PIT feature store:** every feature carries an "available-at" timestamp; the store returns only rows knowable at the decision time.
- **News timestamps:** stored in UTC with publication time, not article date; only news with `published_at <= decision_time` is exposed.
- **LLM memorization risk (P2/P3):** off-the-shelf LLMs may have *memorized* the future. Mitigations: (a) evaluate primarily on data **after the model's training cutoff**; (b) offer an **entity-masking** mode (anonymize tickers/company names/dates) to test reasoning vs recall; (c) always report the train-cutoff gap.
- **Leakage test-suite:** unit tests assert that perturbing any future value cannot change the current observation.

### 4.2 Data splitting
- **Walk-forward** anchored windows (train → validation → out-of-sample), rolled forward.
- **Purged & embargoed** boundaries (López de Prado) to prevent train/test contamination from overlapping feature windows.
- **Regime tagging:** episodes tagged by regime (bull / bear / high-vol / crash-2020 / drawdown-2022) and reported per-regime.

### 4.3 Statistical rigor
- **Block bootstrap** confidence intervals on all headline metrics.
- **Deflated Sharpe Ratio** / multiple-testing correction across the config sweep.
- Report **distributions across seeds and episodes**, never a single path.

### 4.4 Mandatory baselines
`buy_and_hold` · `equal_weight` (multi-asset) · `random` · `momentum` · `mean_reversion`. An agent's result is only meaningful relative to these, net of costs.

---

## 5. Environment: `AlphaDuelGym`

Gymnasium-compatible (`reset`/`step`/`render`), vectorizable.

### 5.1 Observation
A flat `Box` (P0) assembled by the **FeatureStore** from enabled feature groups plus **portfolio state** (current weights, cash fraction, unrealized PnL, steps remaining). Feature groups are toggled by config:
- `technical` — returns, RSI, realized vol, MACD, Bollinger, ATR, momentum.
- `macro` — FRED series (rates, CPI, VIX, term spread), forward-filled PIT.
- `text` — (P2) sentiment scores / embeddings from news.

### 5.2 Action
- **P0 (recommended):** continuous **target portfolio weights** — a `Box` over `[0,1]^(n_assets+1)` (assets + cash), normalized (softmax/Dirichlet). At execution these map to **integer shares** given available cash and next-bar open price; residual goes to cash.
- **Later:** discrete integer buy/sell/hold per asset (harder for PPO; deferred).

### 5.3 Transaction costs (realistic)
`TransactionCostModel` composes:
- **Commission:** flat + per-share + bps of notional.
- **Bid-ask spread:** half-spread paid on every fill.
- **Slippage / market impact:** linear (and optional square-root) in traded fraction of ADV.
Cost-sensitivity sweeps are first-class experiments.

### 5.4 Reward
Pluggable `RewardFunction`:
- `log_return` — dense log portfolio return per step.
- `differential_sharpe` — online risk-adjusted (Moody & Saffell); default for RL.
- optional **drawdown / turnover penalties**.
Terminal-only PnL is available but flagged as sparse/hard.

### 5.5 Episode
Configurable window (e.g. N trading days), fixed initial cash, one or more tradable symbols, seeded start-date sampling for stochastic episodes.

---

## 6. Agents

Common interface (`agents/base.py`):
```python
class Agent(Protocol):
    def act(self, observation, info) -> Action: ...
    def reset(self) -> None: ...
```
Agent families:
- **Baselines** — buy&hold, random, momentum, mean-reversion (no training).
- **RL** (`agents/rl`) — PPO/A2C via Stable-Baselines3; policy over the weight action space.
- **LLM** (`agents/llm`) — off-the-shelf (P2) and fine-tuned (P3). The env observation is rendered to a **textual market brief**; the LLM returns a structured action (JSON) + rationale ("thoughts") that the dashboard can display. Heavy response caching by (obs-hash, model, prompt-version).

---

## 7. Data layer

`data/` with a common `DataSource` interface, a Parquet cache, and PIT metadata:
- **Prices:** `yfinance` (documented limitations: split-adjusted history has look-ahead; unreliable — cache aggressively, plan a Polygon/Tiingo adapter later).
- **Macro:** `FRED` (`fredapi`) — reliable, free; forward-filled to trading calendar with release-date PIT.
- **News:** `GDELT` (P2) — free, noisy; timestamped headlines.
All downloads are idempotent, cached, and versioned by content hash.

---

## 8. Feature store

`features/store.py`: registry of named features, each declaring inputs, warmup length, and PIT semantics. The store:
- assembles observations from a **config-declared enable/disable set** (the ablation mechanism),
- guarantees **no NaN leakage** and correct warmup handling,
- exposes a stable column order + a manifest hash used in experiment tracking.

---

## 9. Evaluation & tracking

- **Metrics** (`evaluation/metrics.py`, pure functions): total/annualized return, volatility, **Sharpe**, **Sortino**, **max drawdown**, Calmar, turnover, #transactions, hit rate, **deflated Sharpe**, PnL curve, per-episode reward.
- **Backtest runner** (`evaluation/backtest.py`): runs an agent over a split, collects a `Trajectory` (obs, actions, fills, costs, portfolio values, rewards, LLM thoughts), computes metrics, logs to MLflow.
- **Splits** (`evaluation/splits.py`): walk-forward + purged/embargoed CV.
- **Tracking** (`tracking/`): MLflow — params (full resolved config), metrics, artifacts (equity curves, trajectory parquet), and config/data/feature manifest hashes for reproducibility.

---

## 10. Configuration

Plain **YAML** composed by domain, validated by **Pydantic** models (`config/schema.py`); secrets via `.env` (`pydantic-settings`). A run resolves one `ExperimentConfig`; its JSON dump + hash is logged. Example groups: `data/`, `env/`, `features/`, `agent/`, `experiment/`.

---

## 11. Repository layout

```
alphaduel/
├── pyproject.toml            # uv-managed, optional-deps: rl, llm, dashboard, dev
├── SPEC.md
├── README.md
├── .env.example
├── configs/                  # composable YAML experiment configs
├── src/alphaduel/
│   ├── config/               # pydantic schema + loader
│   ├── data/                 # sources (prices/macro/news) + cache
│   ├── features/             # feature registry + PIT store
│   ├── envs/                 # AlphaDuelGym, portfolio, costs, rewards
│   ├── agents/               # base + baselines + rl/ + llm/
│   ├── evaluation/           # metrics, backtest, splits, report
│   ├── tracking/             # mlflow utils
│   ├── leakage/              # PIT leakage checks
│   └── cli.py                # `alphaduel` entry point
├── scripts/download_data.py
├── dashboard/app.py          # Streamlit (P4)
└── tests/                    # leakage / costs / env unit tests
```

---

## 12. CLI (entry points)

```
alphaduel download   --config configs/experiment/p0_mvp.yaml   # fetch + cache data
alphaduel run        --config configs/experiment/p0_mvp.yaml   # run agent(s) over splits
alphaduel evaluate   --config ...                              # metrics + MLflow logging
alphaduel dashboard                                            # launch Streamlit (P4)
```

---

## 13. Success criteria

The project is "portfolio-grade" when it can, from a single config + seed, reproducibly: download PIT data, build a feature set with ablation toggles, run several agents through walk-forward evaluation with realistic costs, and produce **honest, CI-bounded, regime-segmented** comparisons against naive baselines — including passing the leakage test-suite. The headline artifact is a defensible answer (with caveats) to the text-vs-quant question.

---

## 14. Known limitations & risks (stated up front)

- Free data (yfinance split-adjustment look-ahead; GDELT noise) caps realism — adapters allow upgrade.
- LLM memorization is a fundamental confound; mitigated but not eliminated.
- Multi-asset integer-action RL and LLM-RL (GRPO/DPO) are compute-heavy and deferred.
- Results are historical-path dependent; conclusions are statistical, not guarantees.
