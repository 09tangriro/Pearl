import gym
import numpy as np
import pytest
import torch as T

from anvil.buffers import ReplayBuffer
from anvil.buffers.rollout_buffer import RolloutBuffer

env = gym.make("CartPole-v0")


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_buffer_init(buffer_class):
    action_space = env.action_space
    observation_space = env.observation_space
    buffer = buffer_class(
        buffer_size=5, observation_space=observation_space, action_space=action_space
    )

    assert buffer.observations.shape == (5, 1, 4)
    assert buffer.actions.shape == (5, 1, 1)
    assert buffer.rewards.shape == (5, 1)
    assert buffer.dones.shape == (5, 1)


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_buffer_add_trajectory_and_sample(buffer_class):
    action_space = env.action_space
    observation_space = env.observation_space
    buffer = buffer_class(
        buffer_size=2, observation_space=observation_space, action_space=action_space
    )
    obs = env.reset()
    action = action_space.sample()
    next_obs, reward, done, _ = env.step(action)

    buffer.add_trajectory(
        observation=obs,
        action=action,
        reward=reward,
        next_observation=next_obs,
        done=done,
    )
    trajectory_numpy = buffer.sample(batch_size=1, dtype="numpy")
    trajectory_torch = buffer.sample(batch_size=1, dtype="torch")

    assert isinstance(trajectory_numpy.observations, np.ndarray)
    assert isinstance(trajectory_torch.observations, T.Tensor)


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_add_batch_trajectories_and_sample(buffer_class):
    action_space = env.action_space
    observation_space = env.observation_space
    buffer = buffer_class(
        buffer_size=5, observation_space=observation_space, action_space=action_space
    )

    observations = np.zeros(shape=(5, 4))
    next_observations = np.zeros(shape=(5, 4))
    actions = np.zeros(5)
    rewards = np.zeros(5)
    dones = np.zeros(5)

    obs = env.reset()
    for i in range(5):
        action = action_space.sample()
        observations[i] = obs
        actions[i] = action
        obs, reward, done, _ = env.step(action)
        next_observations[i] = obs
        rewards[i] = reward
        dones[i] = done

    buffer.add_batch_trajectories(
        observations, actions, rewards, next_observations, dones
    )

    trajectories_numpy = buffer.sample(batch_size=2, dtype="numpy")
    trajectories_torch = buffer.sample(batch_size=2, dtype="torch")

    assert isinstance(trajectories_numpy.observations, np.ndarray)
    assert isinstance(trajectories_torch.observations, T.Tensor)
