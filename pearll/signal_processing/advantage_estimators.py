"""Methods for estimating the advantage function"""

from typing import Tuple

import torch as T

from pearll.common.type_aliases import Tensor


def generalized_advantage_estimate(
    rewards: Tensor,
    old_values: Tensor,
    new_values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """
    Generalized advantage estimate of a trajectory: https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737

    :param old_values: value function result with old_state input
    :param new_values: value function result with new_state input
    :param rewards: agent reward of taking actions in the environment
    :param dones: the done values of each step of the trajectory, indicates whether to bootstrap
    :param gamma: trajectory discount
    :param gae_lambda: exponential mean discount
    :return advantage: the advantage of taking actions in the environment
    :return value: the value of taking actions in the environment
    """
    dones = 1 - dones
    batch_size = rewards.shape[0]

    advantage = (
        T.zeros(batch_size + 1) if rewards.ndim == 1 else T.zeros((batch_size + 1, 1))
    )

    for t in reversed(range(batch_size)):
        delta = rewards[t] + (gamma * new_values[t] * dones[t]) - old_values[t]
        advantage[t] = delta + (gamma * gae_lambda * advantage[t + 1] * dones[t])

    value_target = advantage[:batch_size] + old_values

    assert (
        advantage[:batch_size].shape == old_values.shape
    ), f"The advantage's shape should be {old_values.shape}, instead it is {advantage[:batch_size].shape}."
    assert (
        value_target.shape == old_values.shape
    ), f"The returns' shape should be {old_values.shape}, instead it is {value_target.shape}."

    return advantage[:batch_size], value_target
