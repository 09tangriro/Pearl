import numpy as np

from anvil.signal_processing.sample_estimators import TD_lambda, TD_zero


def test_TD_lambda():
    rewards = np.ones(shape=(2, 3))
    last_values = np.ones(shape=(2,))
    last_dones = np.zeros(shape=(2,))

    actual_returns = TD_lambda(rewards, last_values, last_dones, gamma=1)
    expected_returns = np.array([4, 4], dtype=np.float32)

    np.equal(actual_returns, expected_returns)


def test_TD_zero():
    rewards = np.ones(shape=(2, 3))
    next_values = np.ones(shape=(2, 3))
    dones = np.zeros(shape=(2, 3))

    actual_returns = TD_zero(rewards, next_values, dones, gamma=1)
    expected_returns = np.array([[2, 2, 2], [2, 2, 2]], dtype=np.float32)
    print(actual_returns)

    np.equal(actual_returns, expected_returns)
