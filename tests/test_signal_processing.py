import numpy as np
import pytest
import torch as T

from anvil.signal_processing.sample_estimators import (
    TD_lambda,
    TD_zero,
    generalized_advantage_estimate,
    sample_forward_kl_divergence,
    sample_reverse_kl_divergence,
    soft_q_target,
)


@pytest.mark.parametrize(
    "kl_divergence", [sample_forward_kl_divergence, sample_reverse_kl_divergence]
)
def test_kl_divergence(kl_divergence):
    # KL Div = E(unbiased approx KL Div)
    T.manual_seed(8)
    num_samples = int(2e6)
    dist1 = T.distributions.Normal(0, 1)
    dist2 = T.distributions.Normal(1, 2)

    samples = dist2.sample((num_samples,))

    log_probs1 = dist1.log_prob(samples)
    log_probs2 = dist2.log_prob(samples)

    if kl_divergence == sample_forward_kl_divergence:
        full_kl_div = T.distributions.kl_divergence(dist1, dist2)
    else:
        full_kl_div = T.distributions.kl_divergence(dist2, dist1)

    approx_kl_div = kl_divergence(log_probs1.exp(), log_probs2.exp()).mean()

    assert T.allclose(full_kl_div, approx_kl_div, rtol=5e-3)


def test_TD_lambda():
    rewards = np.ones(shape=(2, 3))
    last_values = np.ones(shape=(2,))
    last_dones = np.zeros(shape=(2,))

    actual_returns = TD_lambda(rewards, last_values, last_dones, gamma=1)
    expected_returns = np.array([4, 4], dtype=np.float32)

    np.equal(actual_returns, expected_returns)


def test_TD_zero():
    rewards = np.ones(shape=(3,))
    next_values = np.ones(shape=(3,))
    dones = np.zeros(shape=(3,))

    actual_returns = TD_zero(rewards, next_values, dones, gamma=1)
    expected_returns = np.array([2, 2, 2], dtype=np.float32)

    np.equal(actual_returns, expected_returns)


def test_generalized_advantage_estimate():
    rewards = np.ones(shape=(3,))
    old_values = np.ones(shape=(3,))
    new_values = np.ones(shape=(3,))
    dones = np.zeros(shape=(3,))

    actual_advantages, actual_returns = generalized_advantage_estimate(
        rewards, old_values, new_values, dones, gamma=1, gae_lambda=1
    )
    expected_advantages, expected_returns = (
        np.array([3, 2, 1], dtype=np.float32),
        np.array([4, 3, 2], dtype=np.float32),
    )

    np.equal(actual_advantages, expected_advantages)
    np.equal(actual_returns, expected_returns)


def test_soft_q_target():
    rewards = np.ones(shape=(3,))
    dones = np.zeros(shape=(3,))
    q_values = np.ones(shape=(3,))
    log_probs = np.array([-1, -1, -1], dtype=np.float32)

    actual_target = soft_q_target(rewards, dones, q_values, log_probs, 1, 1)
    expected_target = np.array([3, 3, 3], dtype=np.float32)

    np.equal(actual_target, expected_target)


"""
def test_scale_normalizer():
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space
    observation = observation_space.sample()
    normalized_observation = scale_normalizer(observation, observation_space)
    np.testing.assert_array_less(
        abs(normalized_observation), [1.00001, 1.00001, 1.00001, 1.00001]
    )


def test_mean_std_normalizer():
    batch_size = 10
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space
    observations = np.zeros(shape=(batch_size, 4))
    for i in range(batch_size):
        observations[i] = observation_space.sample()

    normalized_observations = mean_std_normalizer(observations)

    np.testing.assert_almost_equal(
        normalized_observations.mean(axis=0), np.zeros(shape=(4,))
    )
    np.testing.assert_almost_equal(
        normalized_observations.std(axis=0), np.ones(shape=(4,))
    )
"""
