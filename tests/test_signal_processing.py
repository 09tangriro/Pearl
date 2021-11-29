import numpy as np
import pytest
import torch as T

from anvilrl.common.utils import numpy_to_torch
from anvilrl.signal_processing.advantage_estimators import (
    generalized_advantage_estimate,
)
from anvilrl.signal_processing.crossover_operators import crossover_one_point
from anvilrl.signal_processing.return_estimators import (
    TD_lambda,
    TD_zero,
    soft_q_target,
)
from anvilrl.signal_processing.sample_estimators import (
    sample_forward_kl_divergence,
    sample_reverse_kl_divergence,
)
from anvilrl.signal_processing.selection_operators import (
    roulette_selection,
    tournament_selection,
)


@pytest.mark.parametrize(
    "kl_divergence", [sample_forward_kl_divergence, sample_reverse_kl_divergence]
)
@pytest.mark.parametrize("dtype", ["torch", "numpy"])
def test_kl_divergence(kl_divergence, dtype):
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

    approx_kl_div = numpy_to_torch(
        kl_divergence(log_probs1.exp(), log_probs2.exp(), dtype)
    )
    mean_approx_kl_div = T.mean(approx_kl_div)

    assert T.allclose(full_kl_div, mean_approx_kl_div, rtol=5e-3)


@pytest.mark.parametrize("dtype", ["torch", "numpy"])
def test_TD_lambda(dtype):
    rewards = np.ones(shape=(2, 3))
    last_values = np.ones(shape=(2,))
    last_dones = np.zeros(shape=(2,))

    actual_returns = TD_lambda(rewards, last_values, last_dones, gamma=1, dtype=dtype)
    expected_returns = np.array([4, 4], dtype=np.float32)

    np.equal(actual_returns, expected_returns)


@pytest.mark.parametrize("dtype", ["torch", "numpy"])
def test_TD_zero(dtype):
    rewards = np.ones(shape=(3,))
    next_values = np.ones(shape=(3,))
    dones = np.zeros(shape=(3,))

    actual_returns = TD_zero(rewards, next_values, dones, gamma=1, dtype=dtype)
    expected_returns = np.array([2, 2, 2], dtype=np.float32)

    np.equal(actual_returns, expected_returns)


@pytest.mark.parametrize("dtype", ["torch", "numpy"])
def test_generalized_advantage_estimate(dtype):
    rewards = np.ones(shape=(3,))
    old_values = np.ones(shape=(3,))
    new_values = np.ones(shape=(3,))
    dones = np.zeros(shape=(3,))

    actual_advantages, actual_returns = generalized_advantage_estimate(
        rewards, old_values, new_values, dones, gamma=1, gae_lambda=1, dtype=dtype
    )
    expected_advantages, expected_returns = (
        np.array([3, 2, 1], dtype=np.float32),
        np.array([4, 3, 2], dtype=np.float32),
    )

    np.equal(actual_advantages, expected_advantages)
    np.equal(actual_returns, expected_returns)


@pytest.mark.parametrize("dtype", ["torch", "numpy"])
def test_soft_q_target(dtype):
    rewards = np.ones(shape=(3,))
    dones = np.zeros(shape=(3,))
    q_values = np.ones(shape=(3,))
    log_probs = np.array([-1, -1, -1], dtype=np.float32)

    actual_target = soft_q_target(rewards, dones, q_values, log_probs, 1, 1, dtype)
    expected_target = np.array([3, 3, 3], dtype=np.float32)

    np.equal(actual_target, expected_target)


def test_tournament_selection():
    np.random.seed(8)
    population = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    fitness = np.array([2, 1, 3])
    actual_population = tournament_selection(population, fitness, 2)
    expected_population = np.array([[1, 2, 3], [7, 8, 9], [7, 8, 9]])

    np.testing.assert_array_equal(actual_population, expected_population)


def test_roulette_selection():
    np.random.seed(8)
    population = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    fitness = np.array([2, 1, 3])
    actual_population = roulette_selection(population, fitness)
    expected_population = np.array([[7, 8, 9], [7, 8, 9], [7, 8, 9]])

    np.testing.assert_array_equal(actual_population, expected_population)


def test_one_point_crossover():
    np.random.seed(8)
    population = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    actual_population = crossover_one_point(population, 1)
    expected_population = np.array([[1, 5, 6], [4, 2, 3], [7, 8, 9]])
    np.testing.assert_array_equal(actual_population, expected_population)

    actual_population = crossover_one_point(population)
    expected_population = np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]])
    np.testing.assert_array_equal(actual_population, expected_population)
