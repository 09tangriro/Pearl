import gym
import numpy as np
import pytest
import torch as T

from anvilrl.buffers import ReplayBuffer
from anvilrl.buffers.rollout_buffer import RolloutBuffer
from anvilrl.common.type_aliases import Trajectories

env = gym.make("CartPole-v0")


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_buffer_init(buffer_class):
    buffer = buffer_class(env, buffer_size=5)

    assert buffer.observations.shape == (5, 4)
    assert buffer.actions.shape == (5, 1)
    assert buffer.rewards.shape == (5, 1)
    assert buffer.dones.shape == (5, 1)


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_buffer_add_trajectory_and_sample(buffer_class):
    action_space = env.action_space
    buffer = buffer_class(env, buffer_size=2)
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
        assert len(value.shape) == 2

    np.testing.assert_array_almost_equal(
        obs,
        trajectory_numpy.observations.reshape(
            4,
        ),
    )
    np.testing.assert_array_almost_equal(
        next_obs,
        trajectory_numpy.next_observations.reshape(
            4,
        ),
    )
    assert reward == trajectory_numpy.rewards.item()
    assert done == trajectory_numpy.dones.item()
    assert action == trajectory_numpy.actions.item()

    assert isinstance(trajectory_numpy.observations, np.ndarray)
    assert isinstance(trajectory_torch.observations, T.Tensor)


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_add_batch_trajectories_and_sample(buffer_class):
    action_space = env.action_space
    buffer = buffer_class(env, buffer_size=5)

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

    if buffer_class == ReplayBuffer:
        np.testing.assert_array_almost_equal(
            buffer.observations.reshape(5, 4),
            np.concatenate([[next_observations[-1]], next_observations[:-1]]),
        )

    assert isinstance(trajectories_numpy.observations, np.ndarray)
    assert isinstance(trajectories_torch.observations, T.Tensor)


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_last(buffer_class):
    num_steps = 10
    action_space = env.action_space
    observation_space = env.observation_space
    buffer = buffer_class(env, buffer_size=2)

    for _ in range(num_steps):
        obs = observation_space.sample()
        action = action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        trajectory = Trajectories(
            observations=obs[np.newaxis, :],
            actions=np.array([[action]], dtype=np.float32),
            rewards=np.array([[reward]]),
            next_observations=next_obs[np.newaxis, :],
            dones=np.array([[done]], dtype=np.float32),
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

    buffer = buffer_class(env, buffer_size=5)

    num_most_recent = 2
    obs = env.reset()
    for _ in range(num_steps):
        trajectories = [None] * num_most_recent
        for i in range(num_most_recent):
            action = action_space.sample()
            next_obs, reward, done, _ = env.step(action)

            if i == 0:
                trajectories = Trajectories(
                    observations=obs[np.newaxis, :],
                    actions=np.array([[action]], dtype=np.float32),
                    rewards=np.array([[reward]]),
                    next_observations=next_obs[np.newaxis, :],
                    dones=np.array([[done]], dtype=np.float32),
                )
            else:
                trajectories.observations = np.concatenate(
                    (trajectories.observations, obs[np.newaxis, :])
                )
                trajectories.actions = np.concatenate(
                    (trajectories.actions, np.array([[action]], dtype=np.float32))
                )
                trajectories.rewards = np.concatenate(
                    (trajectories.rewards, np.array([[reward]]))
                )
                trajectories.next_observations = np.concatenate(
                    (
                        trajectories.next_observations,
                        next_obs[np.newaxis, :],
                    )
                )
                trajectories.dones = np.concatenate(
                    (trajectories.dones, np.array([[done]], dtype=np.float32))
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


def test_flatten_env_axis():
    arr = np.random.rand(7, 3, 2)
    # First test with a single environment
    env = gym.make("CartPole-v0")
    buffer = ReplayBuffer(env, buffer_size=100)

    actual_result = buffer._flatten_env_axis(arr).shape
    expected_result = (7, 3, 2)

    assert actual_result == expected_result

    # Test with multiple environments
    env = gym.vector.make("CartPole-v0", 3)
    buffer = ReplayBuffer(env, buffer_size=100)

    actual_result = buffer._flatten_env_axis(arr).shape
    expected_result = (6, 2)

    assert actual_result == expected_result


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, RolloutBuffer])
def test_buffer_multiple_envs(buffer_class):
    env = gym.vector.make("CartPole-v0", 2)
    buffer = buffer_class(env, buffer_size=10)

    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        buffer.add_trajectory(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )

        obs = next_obs

    trajectories = buffer.last(batch_size=5)
    assert trajectories.observations.shape == (2, 5, 4)
    assert trajectories.actions.shape == (2, 5, 1)
    assert trajectories.rewards.shape == (2, 5, 1)
    assert trajectories.next_observations.shape == (2, 5, 4)
    assert trajectories.dones.shape == (2, 5, 1)
