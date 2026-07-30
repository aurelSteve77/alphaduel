import numpy as np

from alphaduel.agents.baselines import EqualWeightAgent
from alphaduel.agents.generative import GenPortfolioAgent
from alphaduel.config.schema import EnvConfig
from alphaduel.envs.multi_asset_gym import MultiAssetGym
from alphaduel.evaluation.backtest import run_episode


def test_multi_env_shapes(synthetic_multi_panel):
    env = MultiAssetGym(synthetic_multi_panel, EnvConfig(kind="multi_asset", episode_length=30))
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (synthetic_multi_panel.n_assets,)
    assert len(info["symbols"]) == synthetic_multi_panel.n_assets


def test_multi_env_equal_weight_runs(synthetic_multi_panel):
    env = MultiAssetGym(
        synthetic_multi_panel, EnvConfig(kind="multi_asset", episode_length=25, random_start=False)
    )
    traj = run_episode(env, EqualWeightAgent(), seed=0)
    assert len(traj.rewards) == 25
    assert np.all(np.isfinite(traj.equity))


def test_genportfolio_prior_runs_and_respects_budget(synthetic_multi_panel):
    env = MultiAssetGym(
        synthetic_multi_panel, EnvConfig(kind="multi_asset", episode_length=20, random_start=False)
    )
    agent = GenPortfolioAgent(max_position_weight=0.25, use_model=False)
    agent.train(env)  # builds tokenizer, fits buckets on the train window
    obs, info = env.reset(seed=2)
    action = agent.act(obs, info)
    assert action.shape == (synthetic_multi_panel.n_assets,)
    assert action.sum() <= 1.0 + 1e-6
    assert np.all(action <= 0.25 + 1e-6)
    assert agent.last_thoughts is not None
