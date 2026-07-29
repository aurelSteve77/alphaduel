"""Backtest runner: drive an agent through an episode and collect a trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alphaduel.agents.base import Agent
from alphaduel.envs.alphaduel_gym import AlphaDuelGym


@dataclass
class Trajectory:
    equity: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    actions: list[float] = field(default_factory=list)
    fills: list[int] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    thoughts: list[str | None] = field(default_factory=list)

    @property
    def n_transactions(self) -> int:
        return int(sum(1 for f in self.fills if f != 0))

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))


def run_episode(env: AlphaDuelGym, agent: Agent, seed: int | None = None) -> Trajectory:
    obs, info = env.reset(seed=seed)
    agent.reset()
    traj = Trajectory()
    traj.equity.append(info["equity"])
    traj.timestamps.append(info["timestamp"])

    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        traj.equity.append(info["equity"])
        traj.rewards.append(reward)
        traj.actions.append(float(np.asarray(action).ravel()[0]))
        traj.fills.append(int(info["fill_shares"]))
        traj.costs.append(float(info["fill_cost"]))
        traj.timestamps.append(info["timestamp"])
        traj.thoughts.append(getattr(agent, "last_thoughts", None))
    return traj
