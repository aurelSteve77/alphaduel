# AlphaDuel

Benchmark **quantitative**, **generative**, and **LLM** trading agents on one shared, realistic Gymnasium environment.

> Central question: *Does the textual reasoning of an LLM beat a purely quantitative policy when both see the same information?*

This is a research / portfolio project focused on **methodological rigor** (shared env, costs, leakage controls, risk-adjusted metrics) — not a production trading system. Full methodology and roadmap live in [SPEC.md](SPEC.md).

---

## What’s implemented

| Area | Status |
|------|--------|
| Data (yfinance prices, FRED macro, Parquet cache) | Done |
| Feature store (technical + macro) | Done |
| Single-asset `AlphaDuelGym` + multi-asset `MultiAssetGym` | Done |
| Baselines (single + multi) | Done |
| PPO/SB3 wrapper (`--extra rl`) | Scaffold (train then act) |
| Vanilla LLM agent + LangGraph + multi-provider factory | Done |
| Ticker anonymization (`ASS1`, `ASS2`, …) for LLMs | Done |
| GenPortfolio agent (equal-weight prior; model optional) | Done |
| Evaluation metrics, CLI runs, MLflow tracking | Done |
| Streamlit dashboard (configure, live, evaluate, analyze) | Done |
| News/GDELT text features, LLM SFT/GRPO | Planned (SPEC P2/P3) |

---

## Quickstart

```bash
# 1. Install (uv recommended)
uv sync                              # core
uv sync --extra llm --extra dashboard --extra dev

# 2. Secrets
cp .env.example .env                 # set FRED_API_KEY; optional OPENAI_API_KEY / ANTHROPIC_API_KEY / REPLICATE_API_TOKEN

# 3. Download + run the P0 baselines experiment
uv run alphaduel download -c configs/experiment/p0_mvp.yaml
uv run alphaduel run      -c configs/experiment/p0_mvp.yaml

# 4. Dashboard
uv run alphaduel dashboard

# 5. LLM locally (optional)
ollama pull qwen3.5:2b               # default Ollama model
uv run pytest
```

Multi-asset / GenPortfolio experiment:

```bash
uv run alphaduel download -c configs/experiment/p5_genportfolio.yaml
uv run alphaduel run      -c configs/experiment/p5_genportfolio.yaml
```

---

## Architecture

```
yfinance / FRED  →  Parquet cache  →  FeatureStore  →  MarketPanel
                                                         │
                                              AlphaDuelGym / MultiAssetGym
                                                         │
                    Agents (baselines · RL · LLM · GenPortfolio)  ← same Agent API
                                                         │
                              Metrics / walk-forward / MLflow / Streamlit
```

**Fairness rules**

- Shared env, costs, and reward for every agent.
- Decisions on bar `t` execute at the open of `t + execution_lag` (default lag **1**).
- Integer share fills; residual cash; transaction costs (commission, spread, impact).
- LLM tickers anonymized by default (`ASS1`…) so the model cannot lean on pretrained company knowledge; the dashboard still shows real symbols.

---

## Repository layout

```
alphaduel/
├── configs/                 # Composable YAML (data, env, features, agents, experiments)
├── resources/prompts/       # Jinja prompt templates for LLM agents
├── notebooks/sandbox.ipynb  # Interactive walkthrough
├── scripts/download_data.py
├── static/                  # Dashboard fonts
├── data_cache/              # Local Parquet cache (ALPHADUEL_DATA_DIR)
├── .alphaduel/presets.json  # Dashboard experiment presets
├── .streamlit/config.toml
├── tests/
├── SPEC.md
└── src/alphaduel/
    ├── cli.py               # Typer CLI: download | run | evaluate | dashboard
    ├── pipeline.py          # download → panel → evaluate
    ├── config/              # Pydantic schema + YAML loader (includes + hash)
    ├── data/                # prices (yfinance), macro (FRED), storage
    ├── features/            # technical / macro / text (stub)
    ├── envs/                # gyms, costs, rewards, portfolio
    ├── agents/              # registry, baselines, rl/, llm/, generative/
    ├── generative/          # GenPortfolio tokenizer / model / constraints
    ├── evaluation/          # metrics, backtest, splits, report
    ├── tracking/            # MLflow
    ├── leakage/             # leakage helpers + tests
    ├── prompts/             # PromptManager → resources/prompts
    └── dashboard/           # Streamlit multipage app
```

---

## Installation

Requires **Python ≥ 3.11**. Prefer [uv](https://github.com/astral-sh/uv).

| Extra | Installs | Needed for |
|-------|----------|------------|
| *(core)* | numpy, pandas, gymnasium, yfinance, fredapi, mlflow, typer, jinja2, … | CLI, envs, baselines |
| `rl` | stable-baselines3, torch, tensorboard | PPO agent |
| `llm` | langchain-ollama, langchain-openai, langchain-anthropic, langgraph, langchain-replicate | Vanilla LLM agent |
| `dashboard` | streamlit, plotly | UI |
| `dev` | pytest, ruff, mypy, jupyter | tests & notebooks |
| `llm-train` / `generative` | mostly stubs in `pyproject.toml` | future fine-tuning |

```bash
uv sync --extra rl --extra llm --extra dashboard --extra dev
# or
uv sync --all-extras
```

`langchain-replicate` is pulled from GitHub via `[tool.uv.sources]` (not on PyPI yet).

---

## Configuration & secrets

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `FRED_API_KEY` | Macro series (required when macro features are enabled) |
| `OPENAI_API_KEY` | OpenAI chat models |
| `ANTHROPIC_API_KEY` | Anthropic Claude chat models |
| `REPLICATE_API_TOKEN` | Replicate chat models |
| `HF_TOKEN` | Hugging Face (future fine-tuning) |
| `POLYGON_API_KEY` / `TIINGO_API_KEY` | Optional paid price providers (later) |
| `MLFLOW_TRACKING_URI` | Default `file:./mlruns` |
| `MLFLOW_EXPERIMENT_NAME` | Default `alphaduel` |
| `ALPHADUEL_DATA_DIR` | Parquet cache root (default `./data_cache`) |

Experiments are YAML compositions validated by Pydantic (`ExperimentConfig`). Fragments live under `configs/{data,env,features,agent}/`; experiments under `configs/experiment/`. Each run gets a short config hash for reproducibility.

---

## CLI

```bash
uv run alphaduel download  -c configs/experiment/p0_mvp.yaml
uv run alphaduel run       -c configs/experiment/p0_mvp.yaml
uv run alphaduel evaluate  -c configs/experiment/p0_mvp.yaml   # alias of run
uv run alphaduel dashboard                                       # Streamlit UI
```

`run` / `evaluate` build the market panel, roll every configured agent for `evaluation.n_episodes`, print a Rich metrics table (with bootstrap CIs), and log to MLflow when tracking is enabled.

---

## Environments

| Env | Module | Action space |
|-----|--------|--------------|
| Single-asset | `envs/alphaduel_gym.py` | Target weight in `[0, 1]` (cash = remainder) |
| Multi-asset | `envs/multi_asset_gym.py` | Target weights `(n_assets,)`; cash = `1 − sum` |

Both:

- Expose a flat observation (features + portfolio state).
- Convert target weights → **integer share** orders.
- Apply costs: commission (flat / per-share / bps), half-spread, optional impact vs ADV (`envs/costs.py`).
- Support rewards: `log_return` (default), `differential_sharpe`, `terminal_pnl` (`envs/rewards.py`).

Defaults: see `configs/env/single_asset.yaml` and `configs/env/multi_asset.yaml`.

---

## Agents

All agents implement the shared `Agent` interface and are built via `agents/registry.py` (`kind`: `baseline` | `rl` | `llm` | `generative`).

### Baselines

| Name | Mode |
|------|------|
| `buy_and_hold` | Single |
| `momentum`, `mean_reversion`, `volatility_target` | Single |
| `random` | Single |
| `equal_weight`, `inverse_volatility`, `random_weights` | Multi |
| `genportfolio` | Multi (generative prior; see below) |

Config fragment: `configs/agent/baselines.yaml`.

### RL (PPO)

- `SB3Agent` in `agents/rl/` — needs `uv sync --extra rl`.
- Config: `configs/agent/ppo.yaml`. Train a policy before expecting meaningful `act` behavior.

### LLM Vanilla

Off-the-shelf chat model → structured market brief → free-form rationale + fenced JSON **share deltas** → target weights.

```bash
uv sync --extra llm
# Ollama (default): ollama serve && ollama pull qwen3.5:2b
```

**Providers** (`agents/llm/factory.py` — `LLMHandler.create` / `resolve_llm`):

| Provider | Default model | Auth |
|----------|---------------|------|
| `ollama` | `qwen3.5:2b` | Local daemon |
| `openai` | `gpt-5.4-mini` | `OPENAI_API_KEY` (default `reasoning_effort=low`) |
| `anthropic` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| `replicate` | `meta/meta-llama-3-8b-instruct` | `REPLICATE_API_TOKEN` |

Config: `configs/agent/llm_vanilla.yaml`. Prompts: `resources/prompts/base_agent_*.prompt` (Jinja), loaded by `prompts/manager.py`.

**Ticker masking (default on)** — `agents/llm/masking.py`:

- LLM sees `ASS1`, `ASS2`, … in the brief and must emit those keys.
- Actions are unmasked to real tickers before env execution.
- Dashboard / `last_actions` show **real** symbols; `last_masked_actions` keeps ASS* keys.
- Real ticker keys in the model JSON are ignored under masking.
- Toggle: `mask_symbols` / dashboard “Anonymize tickers for the LLM”.

### GenPortfolio

`agents/generative/genportfolio.py` — autoregressive book-construction scaffold. With `use_model: false` (default) it acts as a constrained equal-weight prior (no torch). Config: `configs/agent/genportfolio.yaml`. Experiment: `configs/experiment/p5_genportfolio.yaml`.

---

## Data & features

| Source | Module | Notes |
|--------|--------|-------|
| yfinance OHLCV | `data/prices.py` | Daily bars; split-adjusted history can embed look-ahead — documented caveat |
| FRED macro | `data/macro.py` | Needs `FRED_API_KEY` (VIX, rates, CPI, …) |
| News / GDELT | stub | Provider `none` by default |
| Cache | `data/storage.py` | Parquet under `ALPHADUEL_DATA_DIR` |

Feature groups (`features/`): technical (returns, momentum, vol, …), macro (PIT forward-fill), text (planned). Toggle via `configs/features/quant_only.yaml` or `quant_macro.yaml`.

Universes:

- Single: `configs/data/single_asset.yaml` (e.g. `AAPL`)
- Multi: `configs/data/multi_asset.yaml` (liquid US names)

---

## Evaluation

Metrics (`evaluation/metrics.py`), annualized with 252 trading days:

- Total / annualized return, volatility  
- Sharpe, Sortino, Calmar  
- Max drawdown  
- Deflated Sharpe (multiple-testing haircut)  
- Transaction count, total reward  

CLI runs bootstrap confidence intervals (`evaluation/report.py`) and log to MLflow. Walk-forward / purge / embargo parameters are declared in experiment YAML (`evaluation.splits`).

---

## Streamlit dashboard

```bash
uv sync --extra dashboard --extra llm   # llm optional but needed for LLM Vanilla
uv run alphaduel dashboard
```

Runs `src/alphaduel/dashboard/app.py` from the project root (so `.streamlit/config.toml` applies). Market data is **real** (yfinance, cached) unless tests force mock GBM panels.

### Pages

| Section | Page | What it does |
|---------|------|----------------|
| — | **Home** | Overview + shortcuts |
| Setup | **Configure** | Universe (single/multi), agents, window, episodes, costs, reward, LLM settings, presets |
| Setup | **Data** | Download price/macro universes; list / clear Parquet cache |
| Setup | **Live run** | Step one agent through an episode; equity, allocation, LLM rationale & actions |
| Analyze | **Evaluate** | Roster many agents (incl. several LLM Vanilla configs); parallel workers (max 3); per-agent episode/step progress; leaderboard & Sharpe distributions |
| Analyze | **Overview** | Equity curves + metrics + drawdown for the Configure agent set |
| Analyze | **Agent detail** | Exposure, fills, LLM thoughts |
| Analyze | **Market** | Normalized price paths for the current panel |

Presets (built-in + saved): `.alphaduel/presets.json` — e.g. balanced multi-asset, single-asset trend, high-cost stress, short LLM vanilla.

**Evaluate tips**

- Reuses Configure market/cost settings; build a roster with distinct LLM provider/model rows.
- Start with few episodes / short length while iterating on LLMs (each step calls the model).
- Progress table shows `queued` / `running` (`episode i/n · step j/L`) / `done` / `error`.
- **SFT datasets (default on):** every LLM contestant writes under `datasets/llm_sft/<run_id>/`:
  - `manifest.json` — run metadata + roster
  - `<agent_slug>/meta.json` — provider/model/params
  - `<agent_slug>/episode_XXX.json` — full per-step traces (messages, market state, actions, rewards, weights)
  - `<agent_slug>/sft.jsonl` — one chat example per parse-ok step (ready for supervised fine-tuning)

---

## Notebooks

```bash
uv sync --extra dev
# open notebooks/sandbox.ipynb in Jupyter / VS Code / Cursor
```

The sandbox walks through panel build, env stepping, and the LLM agent path.

---

## Testing & lint

```bash
uv sync --extra dev
uv run pytest
uv run pytest tests/test_leakage.py tests/test_llm_agent.py
uv run ruff check src tests
uv run mypy src/alphaduel
```

Test modules cover env/costs, multi-asset, metrics, cache, leakage, LLM agent (mocked LLM — no network), dashboard engine (mock panels), and GenPortfolio tokenizer.

---

## Programmatic LLM agent

```python
from alphaduel.agents.llm import LLMHandler, VanillaLLMAgent

llm = LLMHandler.create("qwen3.5:2b", provider="ollama", temperature=0.0)
# llm = LLMHandler.create("gpt-5.4-mini", provider="openai", temperature=0.0)
# llm = LLMHandler.create("claude-haiku-4-5-20251001", provider="anthropic", temperature=0.0)
agent = VanillaLLMAgent(llm=llm, mask_symbols=True, use_memory=False)

weights = agent.act(observation, info)   # np.ndarray target weights
print(agent.last_thoughts)               # free-form rationale
print(agent.last_actions)                # real tickers, e.g. {"AAPL": 2}
print(agent.last_masked_actions)         # {"ASS1": 2}
```

Expected model output shape:

````text
…reasoning…

```json
{"actions": {"ASS1": 2, "ASS2": -1}}
```
````

---

## Caveats

1. **yfinance adjustments** can embed future splits/dividends into historical bars — prefer paid PIT providers later for strict research.
2. **FRED** availability is approximated by observation date in P0; full ALFRED vintages are future work.
3. **LLM memorization:** prefer evaluation windows after the model’s training cutoff; ASS* masking reduces ticker recall but not calendar/event leakage.
4. **Ollama / APIs:** local daemon + pulled model for the default provider; Evaluate caps concurrency at **3** workers.
5. **GenPortfolio / PPO** training stacks are scaffolds — defaults are usable baselines/priors, not fully trained generative/RL policies.
6. **News & fine-tuning extras** are not wired yet (see SPEC phases P2/P3).

---

## Design principles

1. Shared environment, swappable agents.  
2. Factorial agent × feature-set comparisons.  
3. Point-in-time correctness + leakage tests.  
4. Everything is a validated config (hash + seed).  
5. Honest evaluation: mandatory baselines, costs, risk-adjusted metrics, CIs.  

Details, phasing, and GenPortfolio design notes: **[SPEC.md](SPEC.md)**.

---

## License

MIT — see `pyproject.toml`.
