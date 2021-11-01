from typing import Tuple

import numpy as np
import torch as T

from anvil_rl.common.utils import numpy_to_torch


def TD_lambda(
    rewards: np.ndarray,
    last_values: np.ndarray,
    last_dones=np.ndarray,
    gamma: float = 0.99,
) -> T.Tensor:
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

    return numpy_to_torch(returns)


def TD_zero(
    rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, gamma: float = 0.99
) -> T.Tensor:
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
    return numpy_to_torch(returns)


def generalized_advantage_estimate(
    rewards: np.ndarray,
    old_values: np.ndarray,
    new_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[T.Tensor, T.Tensor]:
    """
    Generalized advantage estimate of a trajectory: https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737

    :param old_values: value function result with old_state input
    :param new_values: value function result with new_state input
    :param rewards: agent reward of taking actions in the environment
    :param dones: the done values of each step of the trajectory, indicates whether to bootstrap
    :param gamma: trajectory discount
    :param gae_lambda: exponential mean discount
    """
    dones = 1 - dones
    batch_size = rewards.shape[0]

    advantage = np.zeros(batch_size + 1)

    for t in np.flip(np.arange(batch_size)):
        delta = rewards[t] + (gamma * new_values[t] * dones[t]) - old_values[t]
        advantage[t] = delta + (gamma * gae_lambda * advantage[t + 1] * dones[t])

    value_target = advantage[:batch_size] + old_values

    assert (
        advantage[:batch_size].shape == old_values.shape
    ), f"The advantage's shape should be {old_values.shape}, instead it is {advantage[:batch_size].shape}."
    assert (
        value_target.shape == old_values.shape
    ), f"The returns' shape should be {old_values.shape}, instead it is {value_target.shape}."

    return numpy_to_torch(advantage[:batch_size], value_target)


def soft_q_target(
    rewards: np.ndarray,
    dones: np.ndarray,
    q_values: np.ndarray,
    log_probs: np.ndarray,
    alpha: float,
    gamma: float = 0.99,
) -> T.Tensor:
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
    return numpy_to_torch(returns)


def sample_forward_kl_divergence(
    target_dist_prob: T.Tensor, sample_dist_prob: T.Tensor
) -> T.Tensor:
    """
    Sample forward KL Divergence element-wise
    https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd

    :param target_dist_prob: probabilities from target distribution, p(x)
    :param sample_dist_prob: probabilites from sample distribution, q(x)
    :return: element-wise kl approximation
    """
    ratio = target_dist_prob / sample_dist_prob
    return ratio * T.log(ratio) - (ratio - 1)


def sample_reverse_kl_divergence(
    target_dist_prob: T.Tensor, sample_dist_prob: T.Tensor
) -> T.Tensor:
    """
    Sample reverse KL Divergence element-wise
    https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd

    :param target_dist_prob: probabilities from target distribution, p(x)
    :param sample_dist_prob: probabilites from sample distribution, q(x)
    :return: element-wise kl approximation
    """
    ratio = target_dist_prob / sample_dist_prob
    return (ratio - 1) - T.log(ratio)
