import numpy as np

from anvilrl.common.logging_ import Logger
from anvilrl.common.type_aliases import Log


def test_init():
    Logger()


def test_reset_episode_log():
    logger = Logger()
    logger.reset_episode_log()

    assert logger.episode_rewards == []
    assert logger.episode_actor_losses == []
    assert logger.episode_critic_losses == []
    assert logger.episode_entropies == []
    assert logger.episode_kl_divergences == []


def test_make_episode_log():
    logger = Logger()
    logger.episode_rewards = [0]

    actual_log = logger._make_episode_log()
    expected_log = Log()

    assert actual_log == expected_log


def test_add_train_log():
    logger = Logger()
    train_log = Log()

    logger.add_train_log(train_log)
    assert logger.episode_actor_losses == []
    assert logger.episode_critic_losses == []
    assert logger.episode_kl_divergences == []
    assert logger.episode_entropies == []


def test_add_reward():
    logger = Logger()
    vec_logger = Logger(num_envs=2)

    logger.add_reward(1.0)
    assert logger.episode_rewards == [1.0]

    reward = np.array([1, 1])
    vec_logger.add_reward(reward)
    np.testing.assert_array_almost_equal([reward], vec_logger.episode_rewards)
