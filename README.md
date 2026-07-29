# AlphaDuel

Benchmarking **quantitative RL agents** vs **LLM agents** on a shared, realistic trading environment.

> Central question: *Does the textual reasoning of an LLM beat a purely quantitative neural network when both see the same information?*

See [SPEC.md](SPEC.md) for the full specification, methodology, and phasing.

## Quickstart

```bash
# 1. Create env and install (uv)
uv sync                      # core deps
uv sync --extra rl           # + PPO/A2C (Stable-Baselines3, torch)
uv sync --extra dashboard    # + Streamlit dashboard
uv sync --all-extras         # everything

# 2. Configure secrets
cp .env.example .env         # add FRED_API_KEY etc.

# 3. Run the P0 pipeline
uv run alphaduel download --config configs/experiment/p0_mvp.yaml
uv run alphaduel run      --config configs/experiment/p0_mvp.yaml
uv run alphaduel evaluate --config configs/experiment/p0_mvp.yaml

# 4. (P4) Dashboard
uv run alphaduel dashboard
```

## Design in one picture

```
data (yfinance / FRED / GDELT)  ->  PIT FeatureStore  ->  AlphaDuelGym
                                                              |
                                   Agents: baselines / RL / LLM  (shared interface)
                                                              |
                                Evaluation (walk-forward, costs, metrics)  ->  MLflow
```

## Status

P0 (single-asset, weight actions, quant-only + baselines) scaffold. Later phases (RL, LLM,
fine-tuning, dashboard, multi-asset) are stubbed with interfaces. See [SPEC.md](SPEC.md).

## Principles

- Shared env, swappable agents. Fair factorial comparison (agent × feature-set).
- Point-in-time correctness, enforced by a leakage test-suite.
- Everything is a validated config; reproducible from config hash + seed.
- Honest evaluation: mandatory baselines, walk-forward CV, risk-adjusted metrics, CIs.
