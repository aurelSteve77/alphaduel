from alphaduel.config.schema import EnvConfig
from alphaduel.envs.alphaduel_gym import AlphaDuelGym
from alphaduel.leakage import assert_observation_is_causal


def test_observation_is_causal(synthetic_panel):
    env = AlphaDuelGym(synthetic_panel, EnvConfig(episode_length=30, random_start=False))
    # Perturbing the future must not change the observation at t=50.
    assert_observation_is_causal(synthetic_panel, env, t=50)
