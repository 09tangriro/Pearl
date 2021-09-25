from typing import Tuple

import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
def TD_lambda(
    rewards: np.ndarray,
    last_values: np.ndarray,
    last_dones=np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    TD(lambda) target: https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#temporal-difference-learning

    :param rewards: trajectory rewards
    :param last_values: last values of trajectories from a critic function (e.g. Q function, value function)
    :param last_dones: the done values of the last step of the trajectory, indicates whether to bootstrap
    :param gamma: the discount factor of future rewards
    """
    batch_size = rewards.shape[0]
    td_lambda = rewards.shape[1]
    last_dones = 1 - last_dones

    returns = np.zeros(shape=(batch_size,), dtype=np.float32)
    for i in range(td_lambda):
        returns += (gamma ** i) * rewards[:, i]

    returns += (gamma ** td_lambda) * last_values * last_dones

    return returns


@jit(nopython=True, parallel=True)
def TD_zero(
    rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, gamma: float = 0.99
) -> np.ndarray:
    """
    TD(0) target: https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#combining-td-and-mc-learning

    :param rewards: trajectory rewards
    :param next_values: next values of trajectories from a critic function (e.g. Q function, value function)
    :param dones: the done values of each step of the trajectory, indicates whether to bootstrap
    :param gamma: the discount factor of future rewards
    """
    return rewards + ((1 - dones) * gamma * next_values)


@jit(nopython=True, parallel=True)
def generalized_advantage_estimate(
    rewards: np.ndarray,
    old_values: np.ndarray,
    new_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized advantage estimate of a trajectory: https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737

    :param old_values: value function result with old_state input
    :param new_values: value function result with new_state input
    :param rewards: agent reward of taking actions in the environment
    :param dones: the done values of each step of the trajectory, indicates whether to bootstrap
    :param gamma: exponential mean discount
    :param gae_lambda: trajectory discount
    """
    dones = 1 - dones
    batch_size = rewards.shape[0]

    advantage = np.zeros(batch_size + 1)

    for t in np.flip(np.arange(batch_size)):
        delta = rewards[t] + (gamma * new_values[t] * dones[t]) - old_values[t]
        advantage[t] = delta + (gamma * gae_lambda * advantage[t + 1] * dones[t])

    value_target = advantage[:batch_size] + old_values

    return advantage[:batch_size], value_target
