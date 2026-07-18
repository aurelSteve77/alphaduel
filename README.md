# alphaduel

**A controlled duel between reasoning LLM agents and pure quant models for portfolio allocation.**

> Under *rigorously identical* conditions — same universe, same budget, same
> horizon, same costs, same walk-forward evaluation with no lookahead — does a
> reasoning language model (trained with GRPO on a verifiable P&L reward) rival
> purpose-built numerical models (supervised, deep-portfolio, deep-RL) for
> trading a portfolio?
>
> The deliverable is an honest **study**, not a promise of alpha. Negative
> results are first-class.

> **Status:** 🚧 Work in progress — **M1 (shared substrate)** landed: data pipeline,
> point-in-time features, no-lookahead splits, rule-based baselines and a
> walk-forward eval harness. Next: the quant branch (M2).

## Why this design

- **Strict parity.** A single portfolio environment and a single evaluation
  harness are driven by *both* branches, so any difference comes from the model,
  not the protocol.
- **No lookahead.** Features use information up to `t`; trades execute at
  `t+1` prices; splits are chronological.
- **Daily clock.** One step advances `rebalance_every` trading days (default 1),
  so daily-timestamped company news can be aligned to each decision.
- **Continuous actions.** The agent outputs a per-asset position delta in
  `[-1, 1]` (buy / sell / hold + amount), projected onto a long-only feasible set.

## Install

This project uses **[uv](https://docs.astral.sh/uv/)** and **Python 3.13**.

```bash
# uv provisions Python 3.13 and installs the project + dev deps into .venv
uv sync --extra dev
```

Run any command inside the environment with `uv run`, e.g. `uv run pytest`.

## Quickstart

Run a random agent against the shared environment (uses synthetic prices until
the data pipeline lands):

```python
from alphaduel import AlphaDuelEnv

env = AlphaDuelEnv(n_assets=10, horizon=252, seed=0)
obs, info = env.reset(seed=0)

done = False
while not done:
    action = env.action_space.sample()          # per-asset position deltas
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print("Final portfolio value:", info["portfolio_value"])
```

It is also registered with Gymnasium:

```python
import gymnasium as gym
import alphaduel  # noqa: F401  (registers the env id)

env = gym.make("AlphaDuel-v0")
```

### Real data + baselines

With the `data` extra installed (`uv sync --extra data`), run the rule-based
baselines out-of-sample and print a metrics table (falls back to synthetic data
if no network is available):

```bash
uv run alphaduel-baselines            # live data (yfinance -> Stooq)
uv run alphaduel-baselines --synthetic
```

Programmatic use of the shared substrate:

```python
from alphaduel import MarketData
from alphaduel.baselines import EqualWeight, Momentum
from alphaduel.eval import run_backtest, walk_forward

market = MarketData.from_synthetic(n_days=2600, n_assets=10)
test_lo, test_hi = market.split_indices("2021-12-31", "2022-12-31")["test"]
res = run_backtest(Momentum(), market, test_lo, test_hi)
print(res.metrics)
```

Run the tests:

```bash
uv run pytest
```

## Project layout

```
alphaduel/
├── pyproject.toml           # packaging + tooling (uv, hatchling, ruff, pytest)
├── configs/
│   └── default.yaml         # single source of truth: universe, budget, costs, cadence
├── src/alphaduel/
│   ├── config.py            # typed config (pydantic) + YAML loader
│   ├── cli.py               # `alphaduel-baselines` entry point
│   ├── env/
│   │   ├── portfolio.py     # framework-agnostic simulator (reset/step, costs, constraints)
│   │   └── gym_env.py       # Gymnasium wrapper (feature + synthetic modes)
│   ├── data/
│   │   ├── download.py      # yfinance -> Stooq, parquet cache
│   │   ├── features.py      # point-in-time features (<= t, cross-sectional z-score)
│   │   ├── splits.py        # chronological splits + no-lookahead asserts
│   │   └── dataset.py       # MarketData container (prices + features + splits)
│   ├── baselines/           # equal-weight, buy & hold, momentum
│   ├── eval/                # metrics, backtest, walk-forward
│   └── utils/prices.py      # synthetic GBM prices for dev/tests
└── tests/
```

## Roadmap

- **M1 — Shared substrate** ✅ env, data pipeline, point-in-time features, no-lookahead splits, baselines, walk-forward eval harness.
- **M2 — Quant branch (B):** supervised predict-then-optimize, deep-portfolio (direct Sharpe), deep-RL (PPO/SAC).
- **M3 — LLM branch (A):** verbalized state, optional SFT, GRPO (Qwen 1.5B → 3B); include a zero-shot baseline.
- **M4 — Comparison v1:** A vs B on the same protocol; the marquee figure.
- **M5 — Text/news (v2):** does language + headlines beat numeric-only? (the real "plus value")
- **M6 — UI playground** (Streamlit) + **M7 — writeup & release**.

## License

MIT — see [LICENSE](LICENSE).
