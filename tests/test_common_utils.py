import gym
import numpy as np
import pytest
import torch as T
from gym import spaces

from pearll.common.utils import (
    extend_shape,
    filter_dataclass_by_none,
    filter_rewards,
    get_space_range,
    get_space_shape,
    set_seed,
    to_numpy,
    to_torch,
)
from pearll.settings import ExplorerSettings

numpy_data = (np.zeros(shape=(2, 2)), np.zeros(shape=(3, 3)))
torch_data = (T.zeros(2, 2), T.zeros(3, 3))
mixed_data = (np.zeros(shape=(2, 2)), T.zeros(3, 3))
one_numpy = np.zeros(shape=(2, 2))
one_torch = T.zeros(2, 2)
zero_dim_numpy = np.array(1)


@pytest.mark.parametrize("input", [numpy_data, torch_data, mixed_data])
def test_to_torch(input):
    actual_output = to_torch(*input)
    for i in range(len(actual_output)):
        assert T.equal(actual_output[i], torch_data[i])


def test_zero_dim_to_torch():
    actual_output = to_torch(zero_dim_numpy)
    expected_output = T.tensor(1)
    assert T.equal(actual_output, expected_output)


@pytest.mark.parametrize("input", [numpy_data, torch_data, mixed_data])
def test_to_numpy(input):
    actual_output = to_numpy(*input)
    for i in range(len(actual_output)):
        np.testing.assert_array_equal(actual_output[i], numpy_data[i])


def test_one_input():
    actual_output = to_numpy(one_torch)
    np.testing.assert_array_equal(actual_output, one_numpy)

    actual_output = to_torch(one_numpy)
    T.equal(actual_output, one_torch)


def test_extend_shape():
    shape = (1, 1, 1)

    actual_output = extend_shape(shape, 5)
    expected_output = (5, 1, 1)

    assert actual_output == expected_output


@pytest.mark.parametrize(
    "space",
    [
        spaces.Box(low=-1, high=1, shape=(2, 2)),
        spaces.Discrete(2),
        spaces.MultiDiscrete([2, 2]),
        spaces.MultiBinary(2),
    ],
)
def test_get_space_shape(space):
    actual_output = get_space_shape(space)

    if isinstance(space, spaces.Box):
        expected_output = (2, 2)
    elif isinstance(space, spaces.Discrete):
        expected_output = (1,)
    else:
        expected_output = (2,)
    assert actual_output == expected_output


@pytest.mark.parametrize(
    "space",
    [
        spaces.Box(low=0, high=1, shape=(2, 2)),
        spaces.Discrete(2),
        spaces.MultiDiscrete([2, 2]),
        spaces.MultiBinary(2),
    ],
)
def test_get_space_range(space):
    actual_output = get_space_range(space)

    if isinstance(space, spaces.Box):
        expected_output = (np.zeros((2, 2)), np.ones((2, 2)))
        np.testing.assert_array_equal(actual_output[0], expected_output[0])
        np.testing.assert_array_equal(actual_output[1], expected_output[1])
    elif isinstance(space, spaces.MultiDiscrete):
        expected_output = (np.zeros((2,)), np.ones((2,)))
        np.testing.assert_array_equal(actual_output[0], expected_output[0])
        np.testing.assert_array_equal(actual_output[1], expected_output[1])
    else:
        expected_output = (0, 1)
        assert actual_output == expected_output


def test_filter_dataclass_by_none():
    dataclass_example = ExplorerSettings()
    actual_output = filter_dataclass_by_none(dataclass_example)
    expected_output = {"start_steps": 1000}
    assert actual_output == expected_output

    actual_output = dataclass_example.filter_none()
    assert actual_output == expected_output


def test_set_seed():
    env = gym.make("CartPole-v0")
    seed = 0

    # Test numpy random seed
    set_seed(seed, env)
    expected_arr = np.random.rand()
    set_seed(seed, env)
    actual_arr = np.random.rand()
    np.testing.assert_array_equal(actual_arr, expected_arr)

    # Test torch random seed
    set_seed(seed, env)
    expected_arr = T.rand(2, 2)
    set_seed(seed, env)
    actual_arr = T.rand(2, 2)
    T.testing.assert_allclose(actual_arr, expected_arr)

    # Test gym random seed
    env = gym.make("CartPole-v0")
    set_seed(seed, env)
    expected_arr = env.action_space.sample()
    set_seed(seed, env)
    actual_arr = env.action_space.sample()
    np.testing.assert_array_equal(actual_arr, expected_arr)
    set_seed(seed, env)
    expected_arr = env.observation_space.sample()
    set_seed(seed, env)
    actual_arr = env.observation_space.sample()
    np.testing.assert_array_equal(actual_arr, expected_arr)


def test_filter_rewards():
    rewards = np.ones((2, 5))
    dones = np.array([[0, 0, 0, 1, 1], [0, 1, 0, 0, 0]])
    actual_output = filter_rewards(rewards, dones)
    expected_output = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    np.testing.assert_array_equal(actual_output, expected_output)

    rewards = np.ones(5)
    dones = np.array([0, 0, 0, 1, 1])
    actual_output = filter_rewards(rewards, dones)
    expected_output = np.array([1, 1, 1, 1, 0])
    np.testing.assert_array_equal(actual_output, expected_output)

    rewards = np.ones(5)
    dones = np.array([0, 1, 0, 0, 0])
    actual_output = filter_rewards(rewards, dones)
    expected_output = np.array([1, 1, 0, 0, 0])
    np.testing.assert_array_equal(actual_output, expected_output)

    rewards = np.ones(5)
    dones = np.array([0, 0, 0, 0, 0])
    actual_output = filter_rewards(rewards, dones)
    expected_output = np.array([1, 1, 1, 1, 1])
    np.testing.assert_array_equal(actual_output, expected_output)

    rewards = np.ones(5)
    dones = np.array([0, 0, 0, 0, 1])
    actual_output = filter_rewards(rewards, dones)
    expected_output = np.array([1, 1, 1, 1, 1])
    np.testing.assert_array_equal(actual_output, expected_output)
