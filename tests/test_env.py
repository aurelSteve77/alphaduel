import numpy as np

from alphaduel.agents.baselines import BuyAndHoldAgent
from alphaduel.config.schema import EnvConfig
from alphaduel.envs.alphaduel_gym import AlphaDuelGym
from alphaduel.evaluation.backtest import run_episode


def test_env_reset_and_step(synthetic_panel):
    env = AlphaDuelGym(synthetic_panel, EnvConfig(episode_length=30))
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert "equity" in info

    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert not truncated


def test_episode_terminates(synthetic_panel):
    env = AlphaDuelGym(synthetic_panel, EnvConfig(episode_length=20, random_start=False))
    traj = run_episode(env, BuyAndHoldAgent(), seed=0)
    assert len(traj.rewards) == 20
    assert len(traj.equity) == 21


def test_buy_and_hold_tracks_price(synthetic_panel):
    env = AlphaDuelGym(synthetic_panel, EnvConfig(episode_length=50, random_start=False))
    traj = run_episode(env, BuyAndHoldAgent(), seed=0)
    # Fully invested from the first step, so equity should move with the price.
    assert traj.equity[-1] != traj.equity[0]
