import gym
import numpy as np
import pytest
import torch as T

from anvil.buffers import ReplayBuffer
from anvil.buffers.rollout_buffer import RolloutBuffer
from anvil.common.type_aliases import Trajectories

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
    assert buffer.rewards.shape == (5, 1, 1)
    assert buffer.dones.shape == (5, 1, 1)


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

    for field in trajectory_numpy.__dataclass_fields__:
        value = getattr(trajectory_numpy, field)
        assert len(value.shape) == 3

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


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_last(buffer_class):
    num_steps = 10
    action_space = env.action_space
    observation_space = env.observation_space
    buffer = buffer_class(
        buffer_size=2, observation_space=observation_space, action_space=action_space
    )

    for _ in range(num_steps):
        obs = observation_space.sample()
        action = action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        trajectory = Trajectories(
            observations=obs[np.newaxis, np.newaxis, :],
            actions=np.array([[[action]]], dtype=np.float32),
            rewards=np.array([[[reward]]]),
            next_observations=next_obs[np.newaxis, np.newaxis, :],
            dones=np.array([[[done]]], dtype=np.float32),
        )

        buffer.add_trajectory(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )

        most_recent = buffer.last(batch_size=1)

        np.testing.assert_array_almost_equal(
            most_recent.observations, trajectory.observations
        )
        np.testing.assert_array_almost_equal(most_recent.actions, trajectory.actions)
        np.testing.assert_array_almost_equal(most_recent.rewards, trajectory.rewards)
        np.testing.assert_array_almost_equal(
            most_recent.next_observations, trajectory.next_observations
        )
        np.testing.assert_array_almost_equal(most_recent.dones, trajectory.dones)

    buffer = buffer_class(
        buffer_size=5, observation_space=observation_space, action_space=action_space
    )

    num_most_recent = 2
    obs = env.reset()
    for _ in range(num_steps):
        trajectories = [None] * num_most_recent
        for i in range(num_most_recent):
            action = action_space.sample()
            next_obs, reward, done, _ = env.step(action)

            if i == 0:
                trajectories = Trajectories(
                    observations=obs[np.newaxis, np.newaxis, :],
                    actions=np.array([[[action]]], dtype=np.float32),
                    rewards=np.array([[[reward]]]),
                    next_observations=next_obs[np.newaxis, np.newaxis, :],
                    dones=np.array([[[done]]], dtype=np.float32),
                )
            else:
                trajectories.observations = np.concatenate(
                    (trajectories.observations, obs[np.newaxis, np.newaxis, :])
                )
                trajectories.actions = np.concatenate(
                    (trajectories.actions, np.array([[[action]]], dtype=np.float32))
                )
                trajectories.rewards = np.concatenate(
                    (trajectories.rewards, np.array([[[reward]]]))
                )
                trajectories.next_observations = np.concatenate(
                    (
                        trajectories.next_observations,
                        next_obs[np.newaxis, np.newaxis, :],
                    )
                )
                trajectories.dones = np.concatenate(
                    (trajectories.dones, np.array([[[done]]], dtype=np.float32))
                )
            buffer.add_trajectory(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )

            obs = next_obs

        most_recent = buffer.last(batch_size=num_most_recent)

        np.testing.assert_array_almost_equal(
            most_recent.observations, trajectories.observations
        )
        np.testing.assert_array_almost_equal(most_recent.actions, trajectories.actions)
        np.testing.assert_array_almost_equal(most_recent.rewards, trajectories.rewards)
        np.testing.assert_array_almost_equal(
            most_recent.next_observations, trajectories.next_observations
        )
        np.testing.assert_array_almost_equal(most_recent.dones, trajectories.dones)
