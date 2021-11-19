"""Methods for estimating the Value and Q functions"""

import numpy as np

from anvilrl.common.enumerations import TrajectoryType
from anvilrl.common.type_aliases import Tensor
from anvilrl.common.utils import numpy_to_torch, torch_to_numpy


def TD_lambda(
    rewards: Tensor,
    last_values: Tensor,
    last_dones=Tensor,
    gamma: float = 0.99,
    dtype: str = "torch",
) -> Tensor:
    """
    TD(lambda) target: https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#temporal-difference-learning

    :param rewards: trajectory rewards
    :param last_values: last values of trajectories from a critic function (e.g. Q function, value function)
    :param last_dones: the done values of the last step of the trajectory, indicates whether to bootstrap
    :param gamma: the discount factor of future rewards
    """
    rewards, last_values = torch_to_numpy(rewards, last_values)
    batch_size = rewards.shape[0]
    td_lambda = rewards.shape[1]
    last_dones = 1 - last_dones

    returns = np.zeros(shape=(batch_size,), dtype=np.float32)
    for i in range(td_lambda):
        returns += (gamma ** i) * rewards[:, i]

    returns += (gamma ** td_lambda) * last_values * last_dones

    if TrajectoryType(dtype.lower()) == TrajectoryType.TORCH:
        return numpy_to_torch(returns)
    elif TrajectoryType(dtype.lower()) == TrajectoryType.NUMPY:
        return returns
    else:
        raise ValueError(
            f"`dtype` flag should be 'torch' or 'numpy', but received {dtype}"
        )


def TD_zero(
    rewards: Tensor,
    next_values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    dtype: str = "torch",
) -> Tensor:
    """
    TD(0) target: https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#combining-td-and-mc-learning

    :param rewards: trajectory rewards
    :param next_values: next values of trajectories from a critic function (e.g. Q function, value function)
    :param dones: the done values of each step of the trajectory, indicates whether to bootstrap
    :param gamma: the discount factor of future rewards
    """
    returns = rewards + ((1 - dones) * gamma * next_values)
    assert (
        returns.shape == next_values.shape
    ), f"The TD returns' shape should be {next_values.shape}, instead it is {returns.shape}."
    if TrajectoryType(dtype.lower()) == TrajectoryType.TORCH:
        return numpy_to_torch(returns)
    elif TrajectoryType(dtype.lower()) == TrajectoryType.NUMPY:
        return returns
    else:
        raise ValueError(
            f"`dtype` flag should be 'torch' or 'numpy', but received {dtype}"
        )


def soft_q_target(
    rewards: Tensor,
    dones: Tensor,
    q_values: Tensor,
    log_probs: Tensor,
    alpha: float,
    gamma: float = 0.99,
    dtype: str = "torch",
) -> Tensor:
    """
    Calculate the soft Q target: https://spinningup.openai.com/en/latest/algorithms/sac.html#id1

    :param rewards: agent reward of taking actions in the environment
    :param dones: the done values of each step of the trajectory, indicates whether to bootstrap
    :param q_values: the q value target network output
    :param log_probs: the log probability of observing the next action given the next observation
    :param alpha: entropy weighting coefficient
    :param gamma: trajectory discount
    """
    returns = rewards + gamma * (1 - dones) * (q_values - (alpha * log_probs))
    assert (
        returns.shape == q_values.shape
    ), f"The advantage's shape should be {q_values.shape}, instead it is {returns.shape}."
    if TrajectoryType(dtype.lower()) == TrajectoryType.TORCH:
        return numpy_to_torch(returns)
    elif TrajectoryType(dtype.lower()) == TrajectoryType.NUMPY:
        return returns
    else:
        raise ValueError(
            f"`dtype` flag should be 'torch' or 'numpy', but received {dtype}"
        )
