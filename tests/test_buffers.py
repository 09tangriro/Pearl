import gym
import pytest

from anvil.buffers import ReplayBuffer

env = gym.make("CartPole-v0")


@pytest.mark.parametrize("buffer_class", [ReplayBuffer])
def test_buffer_init(buffer_class):
    action_space = env.action_space
    observation_space = env.observation_space
    buffer = buffer_class(
        buffer_size=5, observation_space=observation_space, action_space=action_space
    )

    assert buffer.observations.shape == (5, 1, 4)
    assert buffer.actions.shape == (5, 1)
    assert buffer.rewards.shape == (5, 1)
    assert buffer.dones.shape == (5, 1)
