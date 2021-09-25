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
